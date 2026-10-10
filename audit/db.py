"""SQLite storage. One connection per call site; WAL so the dashboard can read while the
pipeline writes."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    call_id        TEXT PRIMARY KEY,          -- sha1(audio)[:12]
    filename       TEXT NOT NULL,             -- original file name
    src_path       TEXT,
    audio_path     TEXT NOT NULL,             -- data/audio/<call_id>.mp3
    duration_sec   REAL,
    channels       INTEGER,
    sample_rate    INTEGER,
    status         TEXT NOT NULL DEFAULT 'new', -- new|transcribing|transcribed|graded|error
    error          TEXT,
    added_at       TEXT NOT NULL,
    transcribed_at TEXT,
    graded_at      TEXT
);

CREATE TABLE IF NOT EXISTS transcripts (
    call_id     TEXT PRIMARY KEY REFERENCES calls(call_id),
    language    TEXT,
    bot_speaker TEXT,                         -- diarization speaker_id mapped to BOT
    turns_json  TEXT NOT NULL,                -- [{idx, speaker_id, role, start, end, text}]
    raw_path    TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    call_id      TEXT PRIMARY KEY REFERENCES calls(call_id),
    signals_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS grades (
    call_id          TEXT PRIMARY KEY REFERENCES calls(call_id),
    grade            TEXT NOT NULL,           -- FATAL | NON_FATAL (final, after rules)
    llm_grade        TEXT,                    -- what the LLM said before rule checks
    grade_reason     TEXT,
    confidence       REAL,
    flow             TEXT,
    outcome          TEXT,
    summary          TEXT,
    root_cause_hypothesis TEXT,
    issues_json      TEXT,                    -- [{code, severity, layer, turn_index, timestamp_sec, quote, explanation, verified, source}]
    risk_flags_json  TEXT,
    requirement_json TEXT,                    -- {stated:{...}, captured:{...}}
    ask_counts_json  TEXT,
    needs_review     INTEGER NOT NULL DEFAULT 0,
    model            TEXT,
    created_at       TEXT NOT NULL,
    raw_json         TEXT
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    n_audited  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS root_causes (
    rc_id       TEXT PRIMARY KEY,             -- R<run>-RC<nn>
    run_id      INTEGER NOT NULL REFERENCES analysis_runs(run_id),
    name        TEXT NOT NULL,
    description TEXT,
    layer       TEXT,                         -- prompt|asr|knowledge|flow|tts_audio|infra
    n_calls     INTEGER,
    n_fatal     INTEGER,
    frequency   REAL,
    severity    REAL,
    priority    REAL,
    rank        INTEGER,
    found_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS call_root_causes (
    call_id   TEXT NOT NULL REFERENCES calls(call_id),
    rc_id     TEXT NOT NULL REFERENCES root_causes(rc_id),
    is_fatal  INTEGER NOT NULL,
    issue_idx TEXT,                           -- json list of indexes into grades.issues_json
    PRIMARY KEY (call_id, rc_id)
);

CREATE TABLE IF NOT EXISTS fixes (
    fix_id        TEXT PRIMARY KEY,           -- <rc_id>-F<n>
    rc_id         TEXT NOT NULL REFERENCES root_causes(rc_id),
    created_at    TEXT NOT NULL,
    summary       TEXT,
    rationale     TEXT,
    edits_json    TEXT,
    diff          TEXT,
    revised_prompt_path TEXT,
    expected_call_ids_json TEXT,
    regression_risk TEXT,
    status        TEXT NOT NULL DEFAULT 'proposed' -- proposed|accepted|rejected|tested
);

-- Filled by the testing phase (Samvaad CHAT-mode replay). Schema fixed now so the
-- dashboard can already show the audit -> issue -> fix -> test timeline.
CREATE TABLE IF NOT EXISTS fix_tests (
    test_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    fix_id          TEXT NOT NULL REFERENCES fixes(fix_id),
    method          TEXT,                     -- e.g. 'replay-chat', 'held-out'
    started_at      TEXT,
    finished_at     TEXT,
    n_cases         INTEGER,
    before_fail_rate REAL,
    after_fail_rate  REAL,
    details_json    TEXT
);

-- One simulated conversation: a case (failed real call or scripted scenario) replayed
-- against the baseline or the fixed prompt, then re-graded.
CREATE TABLE IF NOT EXISTS sim_runs (
    sim_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id     INTEGER NOT NULL REFERENCES fix_tests(test_id),
    fix_id      TEXT NOT NULL,
    case_key    TEXT NOT NULL,                -- call:<call_id> | scenario:<n>
    case_title  TEXT,
    variant     TEXT NOT NULL,                -- baseline | fixed
    turns_json  TEXT,                         -- [{idx, role, text, tools, vars}]
    end_reason  TEXT,
    grade       TEXT,
    rc_present  INTEGER,                      -- did the targeted root cause recur?
    rc_evidence TEXT,
    issues_json TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL
);

-- Step 5: one corrected copy of the prompt, combining the chosen fixes.
CREATE TABLE IF NOT EXISTS prompt_versions (
    version_id  TEXT PRIMARY KEY,             -- v1, v2, ...
    created_at  TEXT NOT NULL,
    fix_ids     TEXT,                         -- json list of fixes applied
    path        TEXT NOT NULL,                -- data/prompts/<version>.md
    diff        TEXT,
    notes       TEXT
);

-- Step 6: a replayable test built from one recording's transcript.
CREATE TABLE IF NOT EXISTS test_cases (
    case_id     TEXT PRIMARY KEY,             -- = call_id
    created_at  TEXT NOT NULL,
    title       TEXT,
    case_json   TEXT NOT NULL,                -- {context, buyer_script, buyer_behaviour, checks}
    path        TEXT
);

-- Step 8: one run of the test cases against original vs corrected prompt.
CREATE TABLE IF NOT EXISTS test_runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id  TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL,                -- running | done | error
    n_cases     INTEGER,
    summary_json TEXT
);

CREATE TABLE IF NOT EXISTS sim_results (
    sim_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES test_runs(run_id),
    case_id     TEXT NOT NULL,
    variant     TEXT NOT NULL,                -- original | corrected
    turns_json  TEXT,
    end_reason  TEXT,
    grade       TEXT,
    scorecard_json TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS human_labels (
    call_id    TEXT PRIMARY KEY,
    label      TEXT NOT NULL,                 -- FATAL | NON_FATAL
    notes      TEXT,
    labeler    TEXT,
    created_at TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    conn = sqlite3.connect(config.AUDIT_DB, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


MIGRATIONS = [
    "ALTER TABLE fixes ADD COLUMN samvaad_version INTEGER",
    "ALTER TABLE fix_tests ADD COLUMN status TEXT",
    "ALTER TABLE fix_tests ADD COLUMN backend TEXT",
    "ALTER TABLE grades ADD COLUMN scorecard_json TEXT",
    "ALTER TABLE fixes ADD COLUMN selected INTEGER DEFAULT 1",
]


def init():
    with connect() as c:
        c.executescript(SCHEMA)
        for m in MIGRATIONS:
            try:
                c.execute(m)
            except sqlite3.OperationalError:
                pass  # column already exists


def rows(sql: str, params=()) -> list[dict]:
    with connect() as c:
        return [dict(r) for r in c.execute(sql, params).fetchall()]


def one(sql: str, params=()) -> dict | None:
    r = rows(sql, params)
    return r[0] if r else None


def execute(sql: str, params=()):
    with connect() as c:
        c.execute(sql, params)


def set_status(call_id: str, status: str, error: str | None = None, **stamps):
    cols = ["status=?", "error=?"] + [f"{k}=?" for k in stamps]
    with connect() as c:
        c.execute(f"UPDATE calls SET {', '.join(cols)} WHERE call_id=?",
                  (status, error, *stamps.values(), call_id))


def loads(s, default=None):
    if s is None:
        return default
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return default
