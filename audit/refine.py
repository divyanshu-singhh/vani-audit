"""Test → compare with the recordings → if the corrected prompt is not better, revise it and re-test.

Score per call (lower is better): Fatal 10, Non-fatal 3, Pass 0, plus 1 per failing audit parameter
(dead air and WER excluded — a text replay cannot measure them).

    better  corrected total < recording total            → stop, keep this version
    worse   corrected total > recording total            → revise the prompt using what got worse, re-test
    same    equal totals while the recordings had issues → revise too (the goal is improvement)

Each revision becomes a new prompt version (v4, v5, …) and a new test run linked to the previous one,
up to REFINE_MAX_ROUNDS runs in total.
"""
import json
import logging
from pathlib import Path

from . import config, db, fixes, rubric, simulate
from .grader import format_transcript

log = logging.getLogger(__name__)

GRADE_POINTS = {"FATAL": 10, "NON_FATAL": 3, "PASS": 0}
AUDIO_ONLY = {"DEAD_AIR", "WER"}


def _failing(card: list[dict]) -> set[str]:
    return {c["param"] for c in card or [] if c["status"] in ("fatal", "non_fatal") and c["param"] not in AUDIO_ONLY}


def _score(grade: str | None, card: list[dict]) -> int:
    return GRADE_POINTS.get(grade, 0) + len(_failing(card))


def compare_run(run_id: int) -> dict:
    sims = db.rows("SELECT case_id, grade, scorecard_json, turns_json FROM sim_results"
                   " WHERE run_id=? AND variant='corrected' AND error IS NULL", (run_id,))
    cases, rec_total, cor_total = [], 0, 0
    worse_params: dict[str, int] = {}
    for s in sims:
        g = db.one("SELECT grade, scorecard_json FROM grades WHERE call_id=?", (s["case_id"],)) or {}
        rc, cc = db.loads(g.get("scorecard_json"), []) or [], db.loads(s["scorecard_json"], []) or []
        r, c = _score(g.get("grade"), rc), _score(s["grade"], cc)
        rec_total, cor_total = rec_total + r, cor_total + c
        new_fails = sorted(_failing(cc) - _failing(rc))
        for p in new_fails:
            worse_params[p] = worse_params.get(p, 0) + 1
        cases.append({"case_id": s["case_id"], "recording": g.get("grade"), "corrected": s["grade"],
                      "rec_score": r, "cor_score": c, "new_fails": new_fails,
                      "still_failing": sorted(_failing(cc) & _failing(rc)), "fixed": sorted(_failing(rc) - _failing(cc))})
    if not cases:
        verdict = "error"
    elif cor_total < rec_total:
        verdict = "better"
    elif cor_total > rec_total:
        verdict = "worse"
    else:
        verdict = "same"
    return {"verdict": verdict, "recording_score": rec_total, "corrected_score": cor_total, "cases": cases,
            "worse_params": worse_params}


def _feedback(run_id: int, cmp: dict, diff: str) -> str:
    """What went wrong with the corrected prompt, for the revision request."""
    blocks = [f"## Changes already made to the prompt in this version (diff vs original)\n{diff[:4000]}"]
    for c in cmp["cases"]:
        if c["cor_score"] < c["rec_score"]:
            continue
        s = db.one("SELECT scorecard_json, turns_json FROM sim_results WHERE run_id=? AND case_id=? AND "
                   "variant='corrected'", (run_id, c["case_id"]))
        card = db.loads(s["scorecard_json"], []) or []
        turns = db.loads(s["turns_json"], []) or []
        bad = [x for x in card if x["param"] in set(c["new_fails"]) | set(c["still_failing"])]
        notes = "\n".join(f"  - {rubric.PARAM_NAMES[x['param']]} ({x['status']}) @turn {x.get('turn_index')}: "
                          f"{x.get('note', '')}" for x in bad)
        focus = {x["turn_index"] for x in bad if isinstance(x.get("turn_index"), int)}
        keep = {j for f in focus for j in range(f - 3, f + 3)} if focus else set(range(24))
        title = (db.one("SELECT filename FROM calls WHERE call_id=?", (c["case_id"],)) or {}).get("filename", c["case_id"])
        blocks.append(f"### {title}: recording {c['recording']} → corrected prompt {c['corrected']}\n"
                      f"New problems caused by the correction: {', '.join(rubric.PARAM_NAMES[p] for p in c['new_fails']) or 'none'}\n"
                      f"Still failing: {', '.join(rubric.PARAM_NAMES[p] for p in c['still_failing']) or 'none'}\n"
                      f"Audit notes:\n{notes}\nSimulated conversation with the corrected prompt (excerpt):\n"
                      f"{format_transcript([t for t in turns if t['idx'] in keep])}")
    return "\n\n".join(blocks)


def refine_version(version_id: str, run_id: int, cmp: dict) -> str | None:
    v = db.one("SELECT * FROM prompt_versions WHERE version_id=?", (version_id,))
    lines = Path(v["path"]).read_text(encoding="utf-8").splitlines()
    worse = ", ".join(f"{rubric.PARAM_NAMES[p]} ({n})" for p, n in sorted(cmp["worse_params"].items(), key=lambda x: -x[1]))
    name = f"Corrected prompt {version_id} tested {cmp['verdict']} than the recordings"
    desc = (f"Replaying the recorded calls with prompt {version_id} scored {cmp['corrected_score']} vs "
            f"{cmp['recording_score']} for the real recordings (lower is better). "
            + (f"New failures introduced: {worse}. " if worse else "")
            + "Revise the prompt so these problems stop, keep the intended improvements, and you may change or "
              "undo lines that an earlier correction added if they caused the regression.")
    p = fixes.propose(name, desc, f"run {run_id}", _feedback(run_id, cmp, v["diff"] or ""), lines, tag=f"refine-{version_id}")
    if not any(e.get("applied") for e in p["edits"]):
        log.warning("refinement of %s produced no applicable change", version_id)
        return None
    note = f"Refined from {version_id} after test run {run_id} ({cmp['verdict']}): {p['out'].get('summary', '')}"
    return fixes.save_version(p["revised"], db.loads(v["fix_ids"], []), note)


def _needs_refinement(cmp: dict) -> bool:
    return cmp["verdict"] == "worse" or (cmp["verdict"] == "same" and cmp["recording_score"] > 0)


def test_and_refine(version_id: str, case_ids: list[str], max_rounds: int | None = None) -> list[int]:
    """Run the test; while the corrected prompt is not better than the recordings, revise and re-test."""
    max_rounds = max(1, max_rounds or config.REFINE_MAX_ROUNDS)
    runs, parent = [], None
    for rnd in range(1, max_rounds + 1):
        (config.DATA_DIR / "pipeline.step").write_text(
            f"Testing {version_id} (round {rnd}/{max_rounds}, {len(case_ids)} case(s))", encoding="utf-8")
        run_id = simulate.run_test(version_id, case_ids)
        cmp = compare_run(run_id)
        db.execute("UPDATE test_runs SET round=?, parent_run=?, verdict=?, compare_json=? WHERE run_id=?",
                   (rnd, parent, cmp["verdict"], json.dumps(cmp), run_id))
        runs.append(run_id)
        log.info("round %d: %s → %s (recording score %d, corrected %d)", rnd, version_id, cmp["verdict"],
                 cmp["recording_score"], cmp["corrected_score"])
        if not _needs_refinement(cmp) or rnd == max_rounds:
            break
        (config.DATA_DIR / "pipeline.step").write_text(f"Revising {version_id} (it tested {cmp['verdict']})",
                                                       encoding="utf-8")
        new_v = refine_version(version_id, run_id, cmp)
        if not new_v:
            break
        version_id, parent = new_v, run_id
    return runs


def chain(run_id: int) -> list[dict]:
    """All runs of one refinement loop, oldest first, for the run given."""
    r = db.one("SELECT * FROM test_runs WHERE run_id=?", (run_id,))
    while r and r.get("parent_run"):
        r = db.one("SELECT * FROM test_runs WHERE run_id=?", (r["parent_run"],))
    out = []
    while r:
        out.append(r)
        r = db.one("SELECT * FROM test_runs WHERE parent_run=? ORDER BY run_id LIMIT 1", (r["run_id"],))
    return out
