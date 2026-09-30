from __future__ import annotations

import json
import sqlite3
from pathlib import Path

IMAGE_STATES = ("METADATA_REJECTED", "PENDING_DOWNLOAD", "DOWNLOADING", "DOWNLOADED", "PROCESSING", "AUTO_ACCEPTED", "AUTO_REJECTED", "HUMAN_PENDING", "HUMAN_ACCEPTED", "HUMAN_REJECTED", "FROZEN")
FAMILY_STATES = ("METADATA_VALID", "AUTO_VALID", "AUTO_REJECTED", "HUMAN_PENDING", "HUMAN_ACCEPTED", "HUMAN_REJECTED", "FROZEN")


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS images (
      image_id TEXT PRIMARY KEY, source TEXT NOT NULL, width INTEGER, height INTEGER,
      url TEXT, status TEXT NOT NULL CHECK(status IN ('METADATA_REJECTED','PENDING_DOWNLOAD','DOWNLOADING','DOWNLOADED','PROCESSING','AUTO_ACCEPTED','AUTO_REJECTED','HUMAN_PENDING','HUMAN_ACCEPTED','HUMAN_REJECTED','FROZEN')),
      local_source TEXT, working_path TEXT, accepted_path TEXT, sha256 TEXT,
      retry_count INTEGER NOT NULL DEFAULT 0, last_error TEXT, metadata_json TEXT NOT NULL,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS families (
      candidate_family_id TEXT PRIMARY KEY, image_id TEXT NOT NULL REFERENCES images(image_id),
      status TEXT NOT NULL CHECK(status IN ('METADATA_VALID','AUTO_VALID','AUTO_REJECTED','HUMAN_PENDING','HUMAN_ACCEPTED','HUMAN_REJECTED','FROZEN')),
      reject_reasons TEXT NOT NULL DEFAULT '[]', payload_json TEXT NOT NULL,
      annotator TEXT, decision_at TEXT, note TEXT, split TEXT,
      UNIQUE(image_id,candidate_family_id));
    CREATE INDEX IF NOT EXISTS images_status_idx ON images(status,image_id);
    CREATE INDEX IF NOT EXISTS families_image_idx ON families(image_id,status);
    CREATE TABLE IF NOT EXISTS events (event_id INTEGER PRIMARY KEY, at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      image_id TEXT, candidate_family_id TEXT, event TEXT NOT NULL, details_json TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    return db


def event(db: sqlite3.Connection, image_id: str | None, family_id: str | None, name: str, details: dict) -> None:
    db.execute("INSERT INTO events(image_id,candidate_family_id,event,details_json) VALUES(?,?,?,?)", (image_id, family_id, name, json.dumps(details, sort_keys=True)))
