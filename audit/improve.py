"""Auto-improve: fix the most frequent error first, keep the fix only if it helps without side effects.

For each error, most frequent first (an audit-sheet parameter failing in the most calls):
  1. Correct the CURRENT BASE prompt for that error (fixes.propose).
  2. Replay the calls that show the error (+ a couple that don't, to catch side effects) with the
     base prompt and with the candidate, and score both with the audit sheet.
  3. Keep the candidate as the new base if
        - the target error happens in fewer replays,
        - no other parameter fails more often, and
        - there are no more fatal calls than before.
     Otherwise flag it for manual review and keep the base unchanged.
  4. Move to the next error, starting from whichever prompt is now the base.
"""
import difflib
import json
import logging
from pathlib import Path

from . import config, db, fixes, rubric, simulate
from .grader import format_transcript

log = logging.getLogger(__name__)

NOT_PROMPT_FIXABLE = {"DEAD_AIR": "latency / audio", "WER": "speech recognition"}
REGRESSION_CALLS = 2  # calls without the error, replayed to catch side effects

SCHEMA = """
CREATE TABLE IF NOT EXISTS base_prompts (
    base_id    TEXT PRIMARY KEY,              -- base_v0 (original), base_v1, ...
    created_at TEXT NOT NULL,
    path       TEXT NOT NULL,
    parent_id  TEXT,
    step_id    INTEGER,
    notes      TEXT
);
CREATE TABLE IF NOT EXISTS improve_sessions (
    session_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL,               -- running | done | error
    top_n       INTEGER,
    start_base  TEXT,
    end_base    TEXT
);
CREATE TABLE IF NOT EXISTS improve_steps (
    step_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER NOT NULL,
    rank         INTEGER NOT NULL,
    param        TEXT NOT NULL,
    n_failing    INTEGER,
    n_fatal      INTEGER,
    failing_calls_json TEXT,
    base_id      TEXT,
    candidate_path TEXT,
    summary      TEXT,
    rationale    TEXT,
    diff         TEXT,
    run_id       INTEGER,
    metrics_json TEXT,
    status       TEXT NOT NULL,              -- correcting | testing | kept | manual_review | approved | rejected | error
    reason       TEXT,
    new_base_id  TEXT,
    created_at   TEXT NOT NULL,
    decided_at   TEXT
);
"""


def init():
    with db.connect() as c:
        c.executescript(SCHEMA)
        if not c.execute("SELECT 1 FROM base_prompts").fetchone():
            c.execute("INSERT INTO base_prompts (base_id, created_at, path, notes) VALUES ('base_v0', ?, ?, ?)",
                      (db.now(), str(config.BOT_PROMPT_PATH), "Original production prompt"))


def current_base() -> dict:
    init()
    return db.one("SELECT * FROM base_prompts ORDER BY CAST(SUBSTR(base_id, 7) AS INTEGER) DESC LIMIT 1")


def _new_base(lines: list[str], parent: str, step_id: int | None, notes: str) -> str:
    n = int(current_base()["base_id"].split("_v")[1]) + 1
    bid = f"base_v{n}"
    path = config.PROMPT_VERSION_DIR / f"vani_prompt_{bid}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    db.execute("INSERT INTO base_prompts (base_id, created_at, path, parent_id, step_id, notes) VALUES (?,?,?,?,?,?)",
               (bid, db.now(), str(path), parent, step_id, notes))
    return bid


# ───────────────────────────── what to fix first ─────────────────────────────

def error_frequency() -> list[dict]:
    """Audit-sheet parameters ranked by how many audited calls fail them (fatal first on ties)."""
    stats: dict[str, dict] = {}
    for r in db.rows("SELECT call_id, scorecard_json FROM grades WHERE scorecard_json IS NOT NULL"):
        for c in db.loads(r["scorecard_json"], []):
            if c["status"] in ("fatal", "non_fatal"):
                s = stats.setdefault(c["param"], {"param": c["param"], "calls": [], "n_fatal": 0, "notes": []})
                s["calls"].append(r["call_id"])
                s["n_fatal"] += c["status"] == "fatal"
                s["notes"].append(c.get("note", ""))
    out = []
    for s in stats.values():
        s.update(name=rubric.PARAM_NAMES.get(s["param"], s["param"]), n_failing=len(s["calls"]),
                 fixable=s["param"] not in NOT_PROMPT_FIXABLE, why_not=NOT_PROMPT_FIXABLE.get(s["param"], ""))
        out.append(s)
    return sorted(out, key=lambda s: (s["fixable"], s["n_failing"], s["n_fatal"]), reverse=True)


def _examples(param: str, call_ids: list[str], k: int = 4) -> str:
    blocks = []
    for cid in call_ids[:k]:
        g = db.one("SELECT summary, scorecard_json FROM grades WHERE call_id=?", (cid,))
        t = db.one("SELECT turns_json FROM transcripts WHERE call_id=?", (cid,))
        c = next((x for x in db.loads(g["scorecard_json"], []) if x["param"] == param), {})
        turns = db.loads(t["turns_json"], [])
        ti = c.get("turn_index")
        keep = set(range(ti - 4, ti + 3)) if isinstance(ti, int) else set(range(30))
        blocks.append(f"### Call {cid} ({c.get('status')})\nSummary: {g['summary']}\n"
                      f"Audit note: {c.get('note', '')}\nTranscript excerpt:\n"
                      f"{format_transcript([x for x in turns if x['idx'] in keep])}")
    return "\n\n".join(blocks)


# ───────────────────────────── one step ─────────────────────────────

def _fails(card: list[dict], key: str) -> bool:
    return any(c["param"] == key and c["status"] in ("fatal", "non_fatal") for c in card or [])


def evaluate(run_id: int, param: str) -> dict:
    sims = db.rows("SELECT case_id, variant, grade, scorecard_json, error FROM sim_results WHERE run_id=?", (run_id,))
    pairs = {}
    for s in sims:
        pairs.setdefault(s["case_id"], {})[s["variant"]] = s
    ok = {cid: p for cid, p in pairs.items() if all(v in p and not p[v]["error"] for v in ("original", "corrected"))}
    m = {"cases": len(pairs), "usable": len(ok), "target": {"before": 0, "after": 0},
         "fatal": {"before": 0, "after": 0}, "params": {}, "per_case": []}
    for cid, p in ok.items():
        b, a = db.loads(p["original"]["scorecard_json"], []), db.loads(p["corrected"]["scorecard_json"], [])
        m["target"]["before"] += _fails(b, param)
        m["target"]["after"] += _fails(a, param)
        m["fatal"]["before"] += p["original"]["grade"] == "FATAL"
        m["fatal"]["after"] += p["corrected"]["grade"] == "FATAL"
        for k in rubric.PARAM_KEYS:
            d = m["params"].setdefault(k, {"before": 0, "after": 0})
            d["before"] += _fails(b, k)
            d["after"] += _fails(a, k)
        m["per_case"].append({"case_id": cid, "before": p["original"]["grade"], "after": p["corrected"]["grade"],
                              "target_before": _fails(b, param), "target_after": _fails(a, param)})
    m["regressions"] = [k for k, d in m["params"].items() if k != param and d["after"] > d["before"]]
    return m


def decide(m: dict, param: str) -> tuple[bool, str]:
    name = rubric.PARAM_NAMES.get(param, param)
    if m["usable"] == 0:
        return False, "No simulated call completed — check the log."
    if m["target"]["before"] == 0:
        return False, (f"The simulator did not reproduce '{name}' with the current base prompt, so the fix "
                       "could not be verified.")
    reasons = []
    if m["target"]["after"] >= m["target"]["before"]:
        reasons.append(f"'{name}' did not drop ({m['target']['before']} → {m['target']['after']} replays)")
    if m["regressions"]:
        reasons.append("got worse: " + ", ".join(
            f"{rubric.PARAM_NAMES[k]} {m['params'][k]['before']}→{m['params'][k]['after']}" for k in m["regressions"]))
    if m["fatal"]["after"] > m["fatal"]["before"]:
        reasons.append(f"fatal calls {m['fatal']['before']} → {m['fatal']['after']}")
    if reasons:
        return False, "; ".join(reasons)
    return True, (f"'{name}' {m['target']['before']} → {m['target']['after']} replays; no other parameter got worse; "
                  f"fatal {m['fatal']['before']} → {m['fatal']['after']}.")


def run_step(session_id: int, rank: int, err: dict) -> dict:
    base = current_base()
    base_lines = Path(base["path"]).read_text(encoding="utf-8").splitlines()
    with db.connect() as c:
        step_id = c.execute(
            "INSERT INTO improve_steps (session_id, rank, param, n_failing, n_fatal, failing_calls_json, base_id,"
            " status, created_at) VALUES (?,?,?,?,?,?,?, 'correcting', ?)",
            (session_id, rank, err["param"], err["n_failing"], err["n_fatal"], json.dumps(err["calls"]),
             base["base_id"], db.now())).lastrowid

    def update(**kw):
        db.execute(f"UPDATE improve_steps SET {', '.join(f'{k}=?' for k in kw)} WHERE step_id=?",
                   (*kw.values(), step_id))

    try:
        notes = "; ".join(sorted({n for n in err["notes"] if n})[:6])
        p = fixes.propose(err["name"], f"Audit parameter '{err['name']}' fails in {err['n_failing']} call(s). "
                          f"Auditor notes: {notes}",
                          f"fails in {err['n_failing']} calls, {err['n_fatal']} fatal",
                          _examples(err["param"], err["calls"]), base_lines, tag=err["param"])
        if not any(e.get("applied") for e in p["edits"]):
            update(status="manual_review", reason="No applicable change could be made to the prompt.",
                   summary=p["out"].get("summary"), decided_at=db.now())
            return db.one("SELECT * FROM improve_steps WHERE step_id=?", (step_id,))
        cand = config.PROMPT_VERSION_DIR / f"candidate_s{session_id}_{rank}_{err['param'].lower()}.md"
        cand.write_text("\n".join(p["revised"]) + "\n", encoding="utf-8")
        update(status="testing", candidate_path=str(cand), summary=p["out"].get("summary"),
               rationale=p["out"].get("rationale"), diff=p["diff"])

        # cases: calls showing the error + a couple without it (side-effect check)
        failing = err["calls"][:config.SIM_CASES_PER_FIX]
        others = [r["call_id"] for r in db.rows("SELECT call_id, scorecard_json FROM grades WHERE scorecard_json IS NOT NULL")
                  if r["call_id"] not in err["calls"]][:REGRESSION_CALLS]
        case_ids = failing + others
        missing = [c for c in case_ids if not simulate.load_case(c)]
        if missing:
            simulate.build_test_cases(missing)
        run_id = simulate.run_test(f"step{step_id}", case_ids,
                                   texts={"original": "\n".join(base_lines), "corrected": "\n".join(p["revised"])})
        m = evaluate(run_id, err["param"])
        keep, reason = decide(m, err["param"])
        new_base = _new_base(p["revised"], base["base_id"], step_id,
                             f"Fixed '{err['name']}' (step {step_id})") if keep else None
        update(run_id=run_id, metrics_json=json.dumps(m), status="kept" if keep else "manual_review",
               reason=reason, new_base_id=new_base, decided_at=db.now())
        log.info("step %s %s: %s — %s", rank, err["param"], "KEPT" if keep else "MANUAL REVIEW", reason)
    except Exception as e:
        log.exception("improve step failed for %s", err["param"])
        update(status="error", reason=str(e)[:500], decided_at=db.now())
    return db.one("SELECT * FROM improve_steps WHERE step_id=?", (step_id,))


def run_session(top_n: int = 3) -> int:
    init()
    errs = [e for e in error_frequency() if e["fixable"]][:top_n]
    with db.connect() as c:
        sid = c.execute("INSERT INTO improve_sessions (started_at, status, top_n, start_base) VALUES (?, 'running', ?, ?)",
                        (db.now(), top_n, current_base()["base_id"])).lastrowid
    try:
        for rank, err in enumerate(errs, 1):
            log.info("── error %d/%d: %s (%d calls)", rank, len(errs), err["name"], err["n_failing"])
            (config.DATA_DIR / "pipeline.step").write_text(
                f"Auto-improve {rank}/{len(errs)}: {err['name']}", encoding="utf-8")
            run_step(sid, rank, err)
        db.execute("UPDATE improve_sessions SET finished_at=?, status='done', end_base=? WHERE session_id=?",
                   (db.now(), current_base()["base_id"], sid))
    except Exception:
        log.exception("improve session failed")
        db.execute("UPDATE improve_sessions SET finished_at=?, status='error' WHERE session_id=?", (db.now(), sid))
    return sid


# ───────────────────────────── manual review ─────────────────────────────

def _carry_over(step_base: list[str], candidate: list[str], current: list[str]) -> tuple[list[str], int]:
    """Re-apply the step's changes (made against an older base) onto the current base."""
    out, skipped = list(current), 0
    ops = [op for op in difflib.SequenceMatcher(a=step_base, b=candidate, autojunk=False).get_opcodes()
           if op[0] != "equal"]
    for tag, i1, i2, j1, j2 in reversed(ops):
        old, new = step_base[i1:i2], candidate[j1:j2]
        if old:
            pos = next((k for k in range(len(out) - len(old) + 1) if out[k:k + len(old)] == old), None)
            if pos is None:
                skipped += 1
                continue
            out[pos:pos + len(old)] = new
        else:
            anchor = step_base[i1 - 1] if i1 > 0 else None
            pos = (out.index(anchor) + 1) if anchor in out else (0 if anchor is None else None)
            if pos is None:
                skipped += 1
                continue
            out[pos:pos] = new
    return out, skipped


def approve(step_id: int) -> str:
    s = db.one("SELECT * FROM improve_steps WHERE step_id=?", (step_id,))
    if not s["candidate_path"]:
        raise ValueError("this step has no corrected prompt to approve")
    step_base = Path(db.one("SELECT path FROM base_prompts WHERE base_id=?", (s["base_id"],))["path"])
    cur = current_base()
    merged, skipped = _carry_over(step_base.read_text(encoding="utf-8").splitlines(),
                                  Path(s["candidate_path"]).read_text(encoding="utf-8").splitlines(),
                                  Path(cur["path"]).read_text(encoding="utf-8").splitlines())
    bid = _new_base(merged, cur["base_id"], step_id,
                    f"Approved manually: '{rubric.PARAM_NAMES.get(s['param'], s['param'])}' (step {step_id})"
                    + (f"; {skipped} change(s) could not be carried over" if skipped else ""))
    db.execute("UPDATE improve_steps SET status='approved', new_base_id=?, decided_at=? WHERE step_id=?",
               (bid, db.now(), step_id))
    return bid


def reject(step_id: int):
    db.execute("UPDATE improve_steps SET status='rejected', decided_at=? WHERE step_id=?", (db.now(), step_id))
