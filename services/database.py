"""SQLite persistence with parameterized queries."""
from __future__ import annotations
import sqlite3
from pathlib import Path
from flask import current_app, g

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 analysis_uuid TEXT NOT NULL UNIQUE,
 image_filename TEXT NOT NULL,
 prediction TEXT NOT NULL CHECK(prediction IN ('Fresh','Stale')),
 confidence REAL NOT NULL,
 fresh_probability REAL NOT NULL,
 stale_probability REAL NOT NULL,
 gradcam_filename TEXT,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at DESC);
"""

def get_db():
    if "db" not in g:
        path = Path(current_app.config["DATABASE"]); path.parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(path); g.db.row_factory = sqlite3.Row
    return g.db

def close_db(_error=None):
    db = g.pop("db", None)
    if db is not None: db.close()

def init_db():
    db=get_db(); db.executescript(SCHEMA)
    existing={row[1] for row in db.execute("PRAGMA table_info(analyses)")}
    for name,definition in {"produce_name":"TEXT", "produce_confidence":"REAL", "freshness_coverage":"TEXT"}.items():
        if name not in existing: db.execute(f"ALTER TABLE analyses ADD COLUMN {name} {definition}")
    db.commit()

def save_analysis(record: dict):
    get_db().execute("""INSERT INTO analyses
      (analysis_uuid,image_filename,prediction,confidence,fresh_probability,stale_probability,gradcam_filename,produce_name,produce_confidence,freshness_coverage)
      VALUES (?,?,?,?,?,?,?,?,?,?)""", (record["analysis_uuid"], record["image_filename"], record["prediction"], record["confidence"], record["fresh_probability"], record["stale_probability"], record.get("gradcam_filename"),record.get("produce_name"),record.get("produce_confidence"),record.get("freshness_coverage")))
    get_db().commit()

def get_analysis(analysis_uuid: str):
    return get_db().execute("SELECT * FROM analyses WHERE analysis_uuid=?", (analysis_uuid,)).fetchone()

def list_analyses():
    return get_db().execute("SELECT * FROM analyses ORDER BY created_at DESC").fetchall()

def delete_analysis(analysis_uuid: str):
    row = get_analysis(analysis_uuid)
    if row: get_db().execute("DELETE FROM analyses WHERE analysis_uuid=?", (analysis_uuid,)); get_db().commit()
    return row

def dashboard_stats():
    return get_db().execute("""SELECT COUNT(*) total,
      SUM(CASE WHEN prediction='Fresh' THEN 1 ELSE 0 END) fresh,
      SUM(CASE WHEN prediction='Stale' THEN 1 ELSE 0 END) stale,
      COALESCE(AVG(confidence),0) average_confidence FROM analyses""").fetchone()
