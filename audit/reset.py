"""Clear history: remove every recording and everything derived from it, to start fresh.

Kept: the original production prompt (prompts/), .env and settings.
Removed: recordings, transcripts, audits, batches, corrections, prompt versions, test cases, test runs,
human verdicts, and the files behind them in data/.
With backup=True (default) the database and those files are first copied to data/archive/<timestamp>/.
"""
import shutil
import sqlite3
from datetime import datetime

from . import config, db

# every table that holds recording-derived data (children first)
TABLES = ["sim_results", "test_runs", "test_cases", "sim_runs", "fix_tests", "call_root_causes", "fixes",
          "root_causes", "analysis_runs", "prompt_versions", "human_labels", "grades", "signals",
          "transcripts", "calls", "batches"]
DIRS = [config.AUDIO_DIR, config.TRANSCRIPT_DIR, config.TRANSCRIPT_COPY_DIR, config.FIX_DIR,
        config.PROMPT_VERSION_DIR, config.TESTCASE_DIR, config.INBOX_DIR]
FILES = ["judge_status.json", "pipeline.step"]


def counts() -> dict:
    return {t: (db.one(f"SELECT COUNT(*) AS n FROM {t}") or {"n": 0})["n"] for t in ("calls", "batches",
            "grades", "prompt_versions", "test_runs", "human_labels")}


def clear_history(backup: bool = True) -> dict:
    archive = None
    if backup:
        archive = config.DATA_DIR / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
        archive.mkdir(parents=True, exist_ok=True)
        src = sqlite3.connect(config.AUDIT_DB)
        dst = sqlite3.connect(archive / "audit.db")
        src.backup(dst)  # consistent copy even while the dashboard has the DB open
        dst.close()
        src.close()
    before = counts()
    with db.connect() as c:
        existing = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        c.execute("PRAGMA foreign_keys=OFF")
        for t in TABLES:
            if t in existing:
                c.execute(f"DELETE FROM {t}")
        if "sqlite_sequence" in existing:
            c.execute("DELETE FROM sqlite_sequence")  # batch / run numbers restart at 1
    for d in DIRS:
        if d.exists():
            if archive:
                shutil.move(str(d), str(archive / d.name))
            else:
                shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        (config.DATA_DIR / f).unlink(missing_ok=True)
    log = config.LOG_DIR / "pipeline.log"
    if log.exists():
        if archive:
            shutil.copy2(log, archive / "pipeline.log")
        log.write_text("", encoding="utf-8")
    v = sqlite3.connect(config.AUDIT_DB, isolation_level=None)  # VACUUM can't run inside a transaction
    v.execute("VACUUM")
    v.close()
    return {"removed": before, "backup": str(archive) if archive else None}
