from __future__ import annotations

import hashlib
import json
import errno
import os
import shutil
import time
import urllib.request
from collections import Counter
from pathlib import Path

from PIL import Image

from .gates import bbox_valid
from .referents import leakage_reasons
from .state_db import connect, event


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_image(source_root: Path, image_id: str) -> Path | None:
    for folder in (source_root / "vg" / "images" / "VG_100K", source_root / "vg" / "images" / "VG_100K_2"):
        p = folder / f"{image_id}.jpg"
        if p.is_file(): return p
    return None


def _copy_or_download(source: Path | None, url: str | None, tmp: Path, byte_budget: int, min_free_bytes: int) -> None:
    if source is not None:
        if source.stat().st_size > byte_budget: raise OSError("MAX_WORKING_GB would be exceeded")
        with source.open("rb") as src, tmp.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk: break
                if dst.tell() + len(chunk) > byte_budget or free_bytes(tmp.parent) - len(chunk) < min_free_bytes:
                    raise OSError("working/free disk limit reached while copying")
                dst.write(chunk)
            dst.flush(); os.fsync(dst.fileno())
        return
    if not url:
        raise FileNotFoundError("no local image or URL")
    req = urllib.request.Request(url, headers={"User-Agent": "RealBind-v2-dataset-builder/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response, tmp.open("wb") as dst:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk: break
            if dst.tell() + len(chunk) > byte_budget or free_bytes(tmp.parent) - len(chunk) < min_free_bytes:
                raise OSError("working/free disk limit reached while downloading")
            dst.write(chunk)
        dst.flush(); os.fsync(dst.fileno())


def _verify(path: Path, image_row, families: list[dict]) -> tuple[int, list[tuple[str, list[str]]]]:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            width, height = im.size
            im.load()
            if width != int(image_row["width"]) or height != int(image_row["height"]):
                return 0, [(f["candidate_family_id"], ["DIMENSION_MISMATCH"]) for f in families]
    except Exception:
        return 0, [(f["candidate_family_id"], ["IMAGE_CORRUPTED"]) for f in families]
    valid, rejected = 0, []
    with Image.open(path) as check:
        for fam in families:
            reasons = leakage_reasons(fam["referent_A"], fam["color_A"]) + leakage_reasons(fam["referent_B"], fam["color_B"])
            for bbox_key in ("bbox_A", "bbox_B"):
                x, y, w, h = fam[bbox_key]
                box = (int(x), int(y), int(x + w), int(y + h))
                if not bbox_valid([x, y, w, h], width, height):
                    reasons.append("INVALID_BBOX")
                elif check.crop(box).width <= 0 or check.crop(box).height <= 0:
                    reasons.append("INVALID_BBOX")
            if reasons: rejected.append((fam["candidate_family_id"], sorted(set(reasons))))
            else: valid += 1
    return valid, rejected


def process(root: Path, source_root: Path, max_working_gb: float = 10, min_free_gb: float = 10,
            max_images: int | None = None, retry_limit: int = 3, target_families: int = 1400) -> dict:
    db = connect(root / "state" / "construction_state.sqlite")
    working = root / "working"; accepted = root / "accepted" / "images"
    working.mkdir(parents=True, exist_ok=True); accepted.mkdir(parents=True, exist_ok=True)
    # Crash recovery: partial tmp files are not valid; reset interrupted state to pending.
    stale = list(working.glob("*.tmp"))
    for p in stale: p.unlink(missing_ok=True)
    interrupted = db.execute("SELECT image_id,working_path FROM images WHERE status IN ('DOWNLOADING','PROCESSING','DOWNLOADED')").fetchall()
    with db:
        for item in interrupted:
            p = Path(item["working_path"]) if item["working_path"] else None
            if p and p.is_file():
                db.execute("UPDATE images SET status='DOWNLOADED' WHERE image_id=?", (item["image_id"],))
            elif (accepted / f"{item['image_id']}.jpg").is_file():
                # Recover a crash between atomic move and accepted-state commit.
                os.replace(accepted / f"{item['image_id']}.jpg", working / f"{item['image_id']}.jpg")
                db.execute("UPDATE images SET status='DOWNLOADED',working_path=? WHERE image_id=?", (str(working / f"{item['image_id']}.jpg"), item["image_id"]))
            else:
                db.execute("UPDATE images SET status='PENDING_DOWNLOAD',working_path=NULL WHERE image_id=?", (item["image_id"],))
    # If a crash happened after a rejection commit but before unlink, complete
    # the already-authorized cleanup only after confirming the commit event.
    rejected = db.execute("SELECT image_id,working_path FROM images WHERE status='AUTO_REJECTED' AND working_path IS NOT NULL").fetchall()
    for item in rejected:
        committed = db.execute("SELECT 1 FROM events WHERE image_id=? AND event IN ('IMAGE_AUTO_REJECTED_COMMITTED','IMAGE_CORRUPTED_COMMITTED') LIMIT 1", (item["image_id"],)).fetchone()
        p = Path(item["working_path"])
        if committed and p.exists(): p.unlink()
        if committed:
            with db:
                db.execute("UPDATE images SET working_path=NULL WHERE image_id=?", (item["image_id"],))
                event(db, item["image_id"], None, "REJECTED_IMAGE_DELETED_RECOVERY", {"path": str(p), "database_commit_preceded_delete": True})
    done = 0
    failed_this_run: set[str] = set()
    while max_images is None or done < max_images:
        auto_valid = db.execute("SELECT COUNT(*) FROM families WHERE status='AUTO_VALID'").fetchone()[0]
        if auto_valid >= target_families: break
        rows = db.execute("SELECT * FROM images WHERE status IN ('PENDING_DOWNLOAD','DOWNLOADED') ORDER BY CASE status WHEN 'DOWNLOADED' THEN 0 ELSE 1 END,image_id").fetchall()
        row = next((r for r in rows if r["image_id"] not in failed_this_run), None)
        if row is None: break
        image_id = row["image_id"]
        fam_rows = db.execute("SELECT candidate_family_id,payload_json FROM families WHERE image_id=? AND status='METADATA_VALID' ORDER BY candidate_family_id", (image_id,)).fetchall()
        families = [json.loads(x["payload_json"]) for x in fam_rows]
        final_tmp = working / f"{image_id}.jpg"
        tmp = working / f"{image_id}.tmp"
        with db:
            if row["status"] == "PENDING_DOWNLOAD":
                db.execute("UPDATE images SET status='DOWNLOADING',working_path=? WHERE image_id=?", (str(final_tmp), image_id))
        if not final_tmp.is_file():
            if free_bytes(root) < min_free_gb * 1024**3:
                with db:
                    db.execute("UPDATE images SET status='PENDING_DOWNLOAD',working_path=NULL WHERE image_id=?", (image_id,))
                    event(db, image_id, None, "DISK_STOP", {"reason": "MIN_FREE_GB", "free_bytes": free_bytes(root)})
                break
            working_size = sum(p.stat().st_size for p in working.iterdir() if p.is_file())
            if working_size >= max_working_gb * 1024**3:
                break
            source = _source_image(source_root, image_id)
            try:
                used_bytes = sum(p.stat().st_size for p in working.iterdir() if p.is_file())
                byte_budget = int(max_working_gb * 1024**3) - used_bytes
                _copy_or_download(source, row["url"], tmp, byte_budget, int(min_free_gb * 1024**3))
                # Verify before atomic rename. A completed but undecodable image
                # is a semantic image rejection, not a network retry.
                try:
                    with Image.open(tmp) as im: im.verify()
                except Exception as corrupt_exc:
                    with db:
                        for fam in families:
                            db.execute("UPDATE families SET status='AUTO_REJECTED',reject_reasons=? WHERE candidate_family_id=?", (json.dumps(["IMAGE_CORRUPTED"]), fam["candidate_family_id"]))
                        db.execute("UPDATE images SET status='AUTO_REJECTED',working_path=?,last_error=? WHERE image_id=?", (str(tmp), str(corrupt_exc), image_id))
                        event(db, image_id, None, "IMAGE_CORRUPTED_COMMITTED", {"error": str(corrupt_exc), "delete_after_commit": True})
                    db.execute("PRAGMA wal_checkpoint(FULL)")
                    for dbfile in (root / "state" / "construction_state.sqlite", root / "state" / "construction_state.sqlite-wal"):
                        if dbfile.exists():
                            fd = os.open(dbfile, os.O_RDONLY); os.fsync(fd); os.close(fd)
                    tmp.unlink(missing_ok=True)
                    with db:
                        db.execute("UPDATE images SET working_path=NULL WHERE image_id=?", (image_id,))
                        event(db, image_id, None, "REJECTED_IMAGE_DELETED", {"path": str(tmp), "database_commit_preceded_delete": True})
                    done += 1
                    continue
                projected = sum(p.stat().st_size for p in working.iterdir() if p.is_file()) + tmp.stat().st_size
                if projected > max_working_gb * 1024**3:
                    tmp.unlink(missing_ok=True)
                    with db:
                        db.execute("UPDATE images SET status='PENDING_DOWNLOAD',working_path=NULL WHERE image_id=?", (image_id,))
                        event(db, image_id, None, "WORKING_LIMIT_STOP", {"projected_bytes": projected, "max_working_bytes": int(max_working_gb * 1024**3)})
                    break
                os.replace(tmp, final_tmp)
                with db:
                    db.execute("UPDATE images SET status='DOWNLOADED',working_path=?,retry_count=0,last_error=NULL WHERE image_id=?", (str(final_tmp), image_id))
                    event(db, image_id, None, "DOWNLOAD_COMPLETE", {"source": str(source) if source else row["url"], "bytes": final_tmp.stat().st_size})
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                if "limit reached" in str(exc) or "MAX_WORKING_GB" in str(exc) or getattr(exc, "errno", None) == errno.ENOSPC or free_bytes(root) < min_free_gb * 1024**3:
                    with db:
                        db.execute("UPDATE images SET status='PENDING_DOWNLOAD',working_path=NULL WHERE image_id=?", (image_id,))
                        event(db, image_id, None, "DISK_STOP", {"reason": str(exc), "free_bytes": free_bytes(root)})
                    break
                with db:
                    n = int(row["retry_count"]) + 1
                    db.execute("UPDATE images SET status='PENDING_DOWNLOAD',retry_count=?,last_error=? WHERE image_id=?", (n, str(exc), image_id))
                    event(db, image_id, None, "DOWNLOAD_RETRY" if n <= retry_limit else "DOWNLOAD_FAILED", {"retry": n, "error": str(exc)})
                if n <= retry_limit: time.sleep(min(2**(n-1), 8)); continue
                failed_this_run.add(image_id)
                done += 1; continue
        with db:
            db.execute("UPDATE images SET status='PROCESSING' WHERE image_id=?", (image_id,))
        valid_count, rejected = _verify(final_tmp, row, families)
        digest = sha256(final_tmp)
        duplicate = db.execute("SELECT image_id,accepted_path FROM images WHERE sha256=? AND image_id<>? AND accepted_path IS NOT NULL LIMIT 1", (digest, image_id)).fetchone()
        if duplicate:
            valid_count = 0
            rejected.extend((f["candidate_family_id"], ["DUPLICATE_IMAGE"]) for f in families)
        if valid_count:
            dest = accepted / f"{image_id}.jpg"
            os.replace(final_tmp, dest)
            with db:
                for fam in families:
                    reasons = next((r for fid, r in rejected if fid == fam["candidate_family_id"]), None)
                    if reasons:
                        db.execute("UPDATE families SET status='AUTO_REJECTED',reject_reasons=? WHERE candidate_family_id=?", (json.dumps(reasons), fam["candidate_family_id"]))
                    else:
                        db.execute("UPDATE families SET status='AUTO_VALID' WHERE candidate_family_id=?", (fam["candidate_family_id"],))
                db.execute("UPDATE images SET status='AUTO_ACCEPTED',accepted_path=?,working_path=NULL,sha256=? WHERE image_id=?", (str(dest), digest, image_id))
                event(db, image_id, None, "IMAGE_AUTO_ACCEPTED", {"accepted_families": valid_count, "rejected_families": len(rejected), "sha256": digest})
            done += 1
        else:
            # Record and fsync rejection before removing the only temporary copy.
            with db:
                for fam in families:
                    reasons = next((r for fid, r in rejected if fid == fam["candidate_family_id"]), ["NO_VALID_PAIR"])
                    db.execute("UPDATE families SET status='AUTO_REJECTED',reject_reasons=? WHERE candidate_family_id=?", (json.dumps(reasons), fam["candidate_family_id"]))
                db.execute("UPDATE images SET status='AUTO_REJECTED',working_path=?,sha256=? WHERE image_id=?", (str(final_tmp), digest, image_id))
                event(db, image_id, None, "IMAGE_AUTO_REJECTED_COMMITTED", {"rejected_families": len(families), "reasons": [r for _, rs in rejected for r in rs], "sha256": digest, "delete_after_commit": True})
            db.execute("PRAGMA wal_checkpoint(FULL)")
            for dbfile in (root / "state" / "construction_state.sqlite", root / "state" / "construction_state.sqlite-wal"):
                if dbfile.exists():
                    fd = os.open(dbfile, os.O_RDONLY); os.fsync(fd); os.close(fd)
            final_tmp.unlink(missing_ok=True)
            with db:
                db.execute("UPDATE images SET working_path=NULL WHERE image_id=?", (image_id,))
                event(db, image_id, None, "REJECTED_IMAGE_DELETED", {"path": str(final_tmp), "database_commit_preceded_delete": True})
            done += 1
    return {"processed_images_this_run": done, "working_bytes": sum(p.stat().st_size for p in working.iterdir() if p.is_file()),
            "accepted_bytes": sum(p.stat().st_size for p in accepted.iterdir() if p.is_file()), "free_bytes": free_bytes(root),
            "stop_reason": "AUTO_ACCEPTED_TARGET" if db.execute("SELECT COUNT(*) FROM families WHERE status='AUTO_VALID'").fetchone()[0] >= target_families else "NO_PENDING_IMAGE_OR_DISK_LIMIT"}
