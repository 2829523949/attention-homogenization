#!/usr/bin/env python3
"""Build the self-contained 500-image RealBind2 facts/image package and ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]  # realbind2-vlm
EXPECTED = 500
PACKAGE_NAME = "RealBind2_FactsImage500_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected object at {path}:{line_no}")
                rows.append(value)
    return rows


def copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_contact_sheet(cases: list[dict[str, Any]], images_root: Path, output: Path) -> None:
    from PIL import Image, ImageDraw

    columns, rows = 20, 25
    cell_w, cell_h = 160, 112
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for index, case in enumerate(cases):
        source = images_root / Path(str(case["image_file"])).name
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((cell_w - 8, cell_h - 26), Image.Resampling.LANCZOS)
            x = (index % columns) * cell_w + 4
            y = (index // columns) * cell_h + 4
            canvas.paste(image, (x, y))
            draw.text((x, y + cell_h - 20), f"{case['case_id']} | {case['image_id']}", fill=(20, 20, 20))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="JPEG", quality=88, optimize=True)


def write_checksums(root: Path, output: Path) -> int:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != output:
            files.append((path.relative_to(root).as_posix(), sha256_file(path)))
    output.write_text("".join(f"{digest}  {relative}\n" for relative, digest in files), encoding="utf-8")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, default=ROOT / "experiments" / "facts_image_500" / "output" / "cohort_v1")
    parser.add_argument("--package-root", type=Path, default=ROOT / "experiments" / "facts_image_500" / "output" / "package")
    parser.add_argument("--name", default=PACKAGE_NAME)
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    cohort = args.cohort_dir.expanduser().resolve()
    package_parent = args.package_root.expanduser().resolve()
    staging = package_parent / args.name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    experiment_dir = ROOT / "experiments" / "facts_image_500"

    cases_path = cohort / "cases.jsonl"
    manifest_path = cohort / "manifest.json"
    audit_path = cohort / "selection_audit.json"
    csv_path = cohort / "selected_images.csv"
    cases = read_jsonl(cases_path)
    if len(cases) != EXPECTED:
        raise ValueError(f"expected {EXPECTED} cases, found {len(cases)}")
    if len({str(case["case_id"]) for case in cases}) != EXPECTED:
        raise ValueError("duplicate case IDs")
    if len({str(case["image_id"]) for case in cases}) != EXPECTED:
        raise ValueError("duplicate image IDs")
    if sum(str(case["qa_type"]) == "REL2C_LEFT" for case in cases) != EXPECTED // 2:
        raise ValueError("left/right imbalance")

    for relative in ("manifest.json", "selection_audit.json", "selected_images.csv", "SELECTION_REPORT.md"):
        copy_file(cohort / relative, staging / relative)
    copy_file(cases_path, staging / "cases.jsonl")
    copy_file(experiment_dir / "README.md", staging / "README.md")
    copy_file(experiment_dir / "PROTOCOL.md", staging / "PROTOCOL.md")
    copy_file(experiment_dir / "package_requirements.txt", staging / "requirements-main-vlm.txt")

    # Scripts, configs, wrappers.
    for name in (
        "select_500_cases.py",
        "run_facts_image.py",
        "summarize_facts_image.py",
        "audit_package.py",
        "merge_results.py",
        "custom_adapter_template.py",
        "build_package.py",
    ):
        copy_file(experiment_dir / "scripts" / name, staging / "scripts" / name)
    for config in sorted((experiment_dir / "configs").glob("*.yaml")):
        copy_file(config, staging / "configs" / config.name)
    for wrapper in sorted(experiment_dir.glob("run_*.sh")) + [experiment_dir / "summarize_one_model.sh"]:
        copy_file(wrapper, staging / wrapper.name)

    # Minimal framework required by the packaged runner.
    copy_file(ROOT / "src/__init__.py", staging / "src/__init__.py")
    for relative in ("src/common", "src/models"):
        shutil.copytree(
            ROOT / relative,
            staging / relative,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
    (staging / "outputs").mkdir(exist_ok=True)

    # Copy the selected images and verify every original hash.
    images_dir = staging / "images"
    images_dir.mkdir()
    images_source_root = Path("/mnt/d/datasets/RealBind_v2/frozen")
    for index, case in enumerate(cases, start=1):
        source = images_source_root / str(case["image_file"])
        if not source.is_file():
            raise FileNotFoundError(source)
        if sha256_file(source) != str(case["image_sha256"]):
            raise ValueError(f"source image hash mismatch: {source}")
        destination = images_dir / source.name
        if destination.exists():
            raise ValueError(f"duplicate destination image: {destination}")
        shutil.copy2(source, destination)
        if index % 100 == 0:
            print(f"copied images {index}/{EXPECTED}", flush=True)

    contact = staging / "contact_sheet.jpg"
    build_contact_sheet(cases, images_dir, contact)
    print(f"contact sheet: {contact} ({contact.stat().st_size} bytes)")

    # Run the package auditor so the audit JSON is shipped inside the package.
    package_audit_path = staging / "PACKAGE_AUDIT.json"
    subprocess.run(
        [
            sys.executable,
            str(staging / "scripts" / "audit_package.py"),
            "--cases", str(staging / "cases.jsonl"),
            "--images-root", str(staging / "images"),
            "--output", str(package_audit_path),
            "--expected", str(EXPECTED),
            "--package-root", str(staging),
        ],
        check=True,
    )

    checksums = staging / "SHA256SUMS"
    count = write_checksums(staging, checksums)
    print(f"SHA256SUMS entries: {count}")

    package_info = {
        "package_name": args.name,
        "case_count": EXPECTED,
        "conditions": ["facts_only", "image_plus_facts"],
        "images": EXPECTED,
        "contact_sheet": "contact_sheet.jpg",
        "cases_sha256": sha256_file(staging / "cases.jsonl"),
        "package_audit_passed": json.loads(package_audit_path.read_text(encoding="utf-8"))["passed"],
        "package_audit_sha256": sha256_file(package_audit_path),
    }
    if args.no_zip:
        (staging / "PACKAGE_INFO.json").write_text(json.dumps(package_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(package_info, indent=2, sort_keys=True))
        return 0

    # Add PACKAGE_INFO.json, then regenerate checksums so the ZIP contents are self-consistent.
    (staging / "PACKAGE_INFO.json").write_text(json.dumps(package_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_checksums(staging, checksums)
    zip_path = package_parent / f"{args.name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    base_name = zip_path.with_suffix("")
    archive = shutil.make_archive(str(base_name), "zip", root_dir=staging.parent, base_dir=staging.name)
    final_zip = Path(archive)
    package_info["zip_path"] = str(final_zip)
    package_info["zip_sha256"] = sha256_file(final_zip)
    package_info["zip_bytes"] = final_zip.stat().st_size
    info_path = package_parent / f"{args.name}.PACKAGE_INFO.json"
    info_path.write_text(json.dumps(package_info, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(package_info, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
