"""Root-cause grouping and ranking across all graded calls.

1. Taxonomy: one LLM pass over every failing call's issues + hypotheses proposes 5-12
   named root causes, each tagged with a layer (prompt / asr / knowledge / flow / ...).
2. Assignment: batched LLM passes map each issue of each call to one root cause.
3. Ranking (deterministic, see PRIORITY_FORMULA).
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from . import config, db, llm, rubric

log = logging.getLogger(__name__)

LOST_OUTCOMES = {"no_requirement_closed", "buyer_dropped", "bot_ended_abruptly", "connectivity_end", "other"}
PRIORITY_FORMULA = (
    "frequency = calls affected ÷ calls audited\n"
    "call severity = 1.0 if the root cause made the call FATAL, else 0.4;\n"
    "                +0.3 if a lead was lost (buyer stated a product but the call ended without transfer/WhatsApp);\n"
    "                +0.2 if the buyer showed frustration  (capped at 1.5)\n"
    "severity = mean call severity over affected calls\n"
    "priority = 100 × frequency × severity"
)

TAXONOMY_SCHEMA = {
    "type": "object",
    "properties": {"root_causes": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "RC01, RC02, ..."},
            "name": {"type": "string", "description": "≤8 words"},
            "description": {"type": "string", "description": "one line: what goes wrong and why"},
            "layer": {"type": "string", "enum": list(rubric.LAYERS)},
        },
        "required": ["key", "name", "description", "layer"]}}},
    "required": ["root_causes"],
}

ASSIGN_SCHEMA = {
    "type": "object",
    "properties": {"assignments": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "call_id": {"type": "string"},
            "links": {"type": "array", "items": {
                "type": "object",
                "properties": {"rc_key": {"type": "string"},
                               "issue_indexes": {"type": "array", "items": {"type": "integer"}}},
                "required": ["rc_key", "issue_indexes"]}},
        },
        "required": ["call_id", "links"]}}},
    "required": ["assignments"],
}

TAXONOMY_SYSTEM = f"""You analyse audit results of an AI voice bot (IndiaMART buyer-side bot VANI).
Group the failures below into 5-12 ROOT CAUSES — the underlying reasons, not symptoms.
Good: "No handling for buyers who give a budget instead of quantity". Bad: "Bot asked quantity many times".
Separate causes that need different fixes. Tag each with the layer where the fix belongs:
{json.dumps(rubric.LAYERS, indent=1)}
Known contradictions inside the bot prompt (a likely source of 'prompt' root causes):
{chr(10).join('- ' + c for c in rubric.PROMPT_CONFLICTS)}
Return JSON only."""

ASSIGN_SYSTEM = """Map every listed issue of every call to exactly one of the given root causes (by key).
Use rc_key "NONE" only for trivial issues that fit no root cause. Return JSON only."""


def _graded_calls():
    return db.rows("SELECT g.call_id, g.grade, g.flow, g.outcome, g.root_cause_hypothesis,"
                   " g.issues_json, g.risk_flags_json, g.requirement_json FROM grades g")


def _call_line(g: dict, with_index: bool) -> str:
    issues = db.loads(g["issues_json"], [])
    parts = [f"{'i' + str(i) + ' ' if with_index else ''}{x['code']}({x['severity']},{x.get('layer')}): "
             f"{(x.get('explanation') or '')[:160]}" for i, x in enumerate(issues)]
    return (f"CALL {g['call_id']} [{g['grade']}, flow={g['flow']}, outcome={g['outcome']}]\n"
            f"  hypothesis: {g['root_cause_hypothesis']}\n  " + "\n  ".join(parts))


def _assign_batch(batch: list[dict], taxonomy: list[dict]) -> list[dict]:
    rc_list = "\n".join(f"- {r['key']}: {r['name']} — {r['description']}" for r in taxonomy)
    user = f"## Root causes\n{rc_list}\n\n## Calls\n" + "\n\n".join(_call_line(g, True) for g in batch)
    try:
        return llm.chat_json(ASSIGN_SYSTEM, user, ASSIGN_SCHEMA, "rc_assign",
                             reasoning=config.GRADER_REASONING)["assignments"]
    except Exception:
        log.exception("assignment batch failed")
        return []


def run() -> int | None:
    graded = _graded_calls()
    failing = [g for g in graded if db.loads(g["issues_json"], [])]
    if not failing:
        log.info("no issues to analyse")
        return None

    # Fatal calls first so they always make it into the taxonomy prompt if we must cap.
    failing.sort(key=lambda g: g["grade"] != "FATAL")
    sample = failing[:600]
    taxonomy = llm.chat_json(TAXONOMY_SYSTEM, "\n\n".join(_call_line(g, False) for g in sample),
                             TAXONOMY_SCHEMA, "rc_taxonomy", reasoning=config.ANALYSIS_REASONING,
                             max_tokens=12000)["root_causes"]
    keys = {r["key"] for r in taxonomy}

    batches = [failing[i:i + 20] for i in range(0, len(failing), 20)]
    with ThreadPoolExecutor(max_workers=config.GRADER_WORKERS) as pool:
        assignments = [a for part in pool.map(lambda b: _assign_batch(b, taxonomy), batches) for a in part]

    by_call = {g["call_id"]: g for g in graded}
    links: dict[tuple[str, str], list[int]] = {}
    for a in assignments:
        if a.get("call_id") not in by_call:
            continue
        for link in a.get("links", []):
            if link.get("rc_key") in keys:
                links.setdefault((a["call_id"], link["rc_key"]), []).extend(link.get("issue_indexes") or [])

    n_audited = len(graded)
    found_at = db.now()
    with db.connect() as c:
        run_id = c.execute("INSERT INTO analysis_runs (created_at, n_audited) VALUES (?,?)",
                           (found_at, n_audited)).lastrowid
        stats = []
        for r in taxonomy:
            rc_id = f"R{run_id}-{r['key']}"
            sev, n_fatal, members = [], 0, []
            for (cid, key), idxs in links.items():
                if key != r["key"]:
                    continue
                g = by_call[cid]
                issues = db.loads(g["issues_json"], [])
                is_fatal = any(0 <= i < len(issues) and issues[i]["severity"] == "fatal" for i in idxs)
                flags = db.loads(g["risk_flags_json"], {})
                stated = (db.loads(g["requirement_json"], {}) or {}).get("stated") or {}
                s = 1.0 if is_fatal else 0.4
                s += 0.3 if (g["outcome"] in LOST_OUTCOMES and stated.get("product")) else 0
                s += 0.2 if flags.get("frustration") else 0
                sev.append(min(s, 1.5))
                n_fatal += is_fatal
                members.append((cid, rc_id, int(is_fatal), json.dumps(sorted(set(idxs)))))
            freq = len(members) / n_audited if n_audited else 0
            severity = sum(sev) / len(sev) if sev else 0
            stats.append([rc_id, run_id, r["name"], r["description"], r["layer"], len(members),
                          n_fatal, freq, severity, 100 * freq * severity, found_at, members])
        stats = [s for s in stats if s[5] > 0]  # drop causes no audited call actually showed
        stats.sort(key=lambda x: x[9], reverse=True)
        for rank, s in enumerate(stats, 1):
            members = s.pop()
            c.execute("INSERT INTO root_causes (rc_id, run_id, name, description, layer, n_calls,"
                      " n_fatal, frequency, severity, priority, found_at, rank) VALUES"
                      " (?,?,?,?,?,?,?,?,?,?,?,?)", (*s, rank))
            c.executemany("INSERT OR REPLACE INTO call_root_causes VALUES (?,?,?,?)", members)
    log.info("analysis run %s: %d root causes over %d audited calls", run_id, len(stats), n_audited)
    return run_id


def latest_run_id() -> int | None:
    r = db.one("SELECT MAX(run_id) AS r FROM analysis_runs")
    return r["r"] if r else None
