"""Per-call grading against the IndiaMART audit sheet (audit/rubric.py PARAMETERS).

1. N independent LLM scorecards in parallel (self-consistency) → majority status per parameter.
2. Rule checks from timings and counts: dead air (> 5 s before a bot turn), looping (> 1 + 2 asks),
   prompt-text leaks.
3. A second reviewer re-checks every FATAL finding against the transcript and drops/downgrades
   unsupported ones.
4. Overall: FATAL if any parameter is fatal, NON_FATAL if any is non-fatal, else PASS.
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from rapidfuzz import fuzz

from . import config, db, llm, rubric

log = logging.getLogger(__name__)

STATUSES = ["ok", "non_fatal", "fatal", "na"]
SEVERITY_ORDER = {"na": 0, "ok": 1, "non_fatal": 2, "fatal": 3}
REQ_FIELDS = {k: {"type": ["string", "null"]} for k in
              ("product", "quantity", "quantity_unit", "spec1", "spec2", "spec3", "buyer_name", "city", "state")}

SCHEMA = {
    "type": "object",
    "properties": {
        "flow": {"type": "string", "enum": list(rubric.FLOWS)},
        "language": {"type": "string"},
        "summary": {"type": "string", "description": "2-3 sentences: what happened on the call"},
        "buyer_intent": {"type": "string", "enum": ["buying", "not_buying", "unclear"]},
        "requirement": {
            "type": "object",
            "properties": {
                "stated": {"type": "object", "properties": REQ_FIELDS, "description": "What the BUYER actually said"},
                "captured": {"type": "object", "properties": REQ_FIELDS, "description": "What the BOT confirmed/acted on"},
            },
        },
        "outcome": {"type": "string", "enum": [
            "transferred_to_live_seller", "whatsapp_details_promised", "no_requirement_closed",
            "callback_original_seller", "buyer_dropped", "bot_ended_abruptly", "connectivity_end", "other"]},
        "ask_counts": {
            "type": "object",
            "properties": {k: {"type": "integer"} for k in rubric.FIELD_LIMITS},
            "description": "How many times the bot asked/confirmed each field, counting rephrasings. "
                           "specification_max = the most times any ONE specification was asked.",
        },
        "scorecard": {
            "type": "array",
            "description": "Exactly one entry per audit parameter of the sheet, in the sheet's order",
            "items": {
                "type": "object",
                "properties": {
                    "param": {"type": "string", "enum": rubric.PARAM_KEYS},
                    "status": {"type": "string", "enum": STATUSES},
                    "turn_index": {"type": ["integer", "null"]},
                    "quote": {"type": "string", "description": "Verbatim words from that turn (original script), '' if ok/na"},
                    "layer": {"type": "string", "enum": list(rubric.LAYERS)},
                    "note": {"type": "string", "description": "≤30 words, English: what was right or wrong"},
                    "confidence": {"type": "number", "description": "0-1: how sure you are of THIS status"},
                },
                "required": ["param", "status", "note"],
            },
        },
        "additional_observation": {"type": "string", "description": "Anything else (pronunciation, tone…), '' if none"},
        "uncovered_issues": {
            "type": "array",
            "description": "Serious problems in this call that NO audit parameter above covers (e.g. abusive or rude "
                           "bot, sharing someone's personal data, legal / safety / consent problem, a call mishandled "
                           "in a way none of the parameters captures). Empty for almost every call.",
            "items": {"type": "object", "properties": {
                "description": {"type": "string"}, "severity": {"type": "string", "enum": ["major", "minor"]},
                "turn_index": {"type": ["integer", "null"]}, "quote": {"type": "string"}},
                "required": ["description", "severity"]},
        },
        "risk_flags": {
            "type": "object",
            "properties": {k: {"type": "boolean"} for k in
                           ("wrong_requirement", "frustration", "loop", "low_confidence", "human_request")},
        },
        "root_cause_hypothesis": {"type": "string", "description": "One generalisable line: WHY the main failure happened"},
        "confidence": {"type": "number"},
    },
    "required": ["flow", "summary", "buyer_intent", "requirement", "outcome", "ask_counts", "scorecard",
                 "risk_flags", "root_cause_hypothesis", "confidence"],
}

SYSTEM = """You are a senior call-quality auditor at IndiaMART, auditing ONE call of the buyer-side AI voice bot "VANI"
with the standard audit sheet below. Return JSON only.

How to audit:
1. From the BUYER turns, work out what the buyer actually wanted (requirement.stated). From the BOT turns, what the bot
   confirmed/acted on (requirement.captured).
2. Score EVERY parameter of the sheet: ok, non_fatal, fatal, or na (did not come up). Use the sheet's own FATAL and
   NON-FATAL definitions. For non-ok statuses give the turn_index and a SHORT verbatim quote from that turn (original
   script, not translated) and the layer where the fix belongs.
3. The transcript is machine speech-to-text of 8 kHz phone audio. If an error is plausibly a transcript artefact rather
   than something the buyer experienced, it is at most non_fatal (layer asr). Speaker labels may be swapped on a line.
4. Do not invent problems. A clean call scores ok/na everywhere.
5. risk_flags: wrong_requirement (captured ≠ stated), frustration (annoyed, repeats self), loop (bot repeats/circles),
   low_confidence (many unclear/garbled buyer turns), human_request (buyer asks for a person / refuses the bot).
6. Give every non-na status a confidence (0-1): how sure you are of THAT status. Decide every parameter yourself —
   use your best judgement; a low confidence is only for genuinely unclear evidence (e.g. inaudible or garbled turns).
   uncovered_issues: only serious problems no parameter covers; leave empty otherwise. Do not repeat parameter issues there.
7. root_cause_hypothesis: the underlying reason for the worst problem, phrased so it applies to other calls
   (e.g. "No instruction for buyers who answer the quantity question with a budget").
"""


def _fmt_ts(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def format_transcript(turns: list[dict]) -> str:
    return "\n".join(f"[{t['idx']:02d}] {_fmt_ts(t['start'])} {t['role']}: {t['text']}" for t in turns)


def _signal_hints(sig: dict) -> str:
    lines = [f"duration={sig.get('duration_sec')}s turns={sig.get('n_turns')} end_state={sig.get('end_state')}"]
    for k, evs in (sig.get("evidence") or {}).items():
        for e in evs[:4]:
            lines.append(f"- {k} @turn {e['turn']}: {e['why']}")
    return "\n".join(lines)


def _verify(item: dict, turns: list[dict]) -> dict:
    """Check the quote really is in the transcript; snap turn/timestamp to where it is."""
    q = (item.get("quote") or "").strip()
    ti = item.get("turn_index")
    item["verified"] = False
    if q and isinstance(ti, int) and 0 <= ti < len(turns) and fuzz.partial_ratio(q, turns[ti]["text"]) >= 70:
        item["verified"] = True
    elif q:
        best = max(turns, key=lambda t: fuzz.partial_ratio(q, t["text"]), default=None)
        if best and fuzz.partial_ratio(q, best["text"]) >= 80:
            item["turn_index"], item["verified"] = best["idx"], True
    ti = item.get("turn_index")
    if isinstance(ti, int) and 0 <= ti < len(turns):
        item["timestamp_sec"] = round(turns[ti]["start"], 1)
    return item


def _consensus(system: str, user: str) -> dict:
    """Self-consistency: N scorecards in parallel; majority status per parameter."""
    n = max(1, config.GRADER_SAMPLES)

    def sample(_):
        try:
            return llm.judge_json(system, user, SCHEMA, "call_grade", reasoning=config.GRADER_REASONING,
                                 temperature=0.0 if n == 1 else 0.3)
        except Exception:
            log.exception("grading sample failed")
            return None

    with ThreadPoolExecutor(max_workers=n) as pool:
        outs = [o for o in pool.map(sample, range(n)) if o]
    if not outs:
        raise RuntimeError("all grading samples failed")

    card = []
    for key in rubric.PARAM_KEYS:
        entries = [next((e for e in o.get("scorecard", []) if e.get("param") == key), None) for o in outs]
        entries = [e for e in entries if e]
        statuses = [e.get("status", "na") if e.get("status") in STATUSES else "na" for e in entries] or ["na"]
        top = max(statuses.count(s) for s in set(statuses))
        tied = [s for s in set(statuses) if statuses.count(s) == top]
        # 3-way tie (ok / non_fatal / fatal) → the middle ground
        status = tied[0] if len(tied) == 1 else ("non_fatal" if "non_fatal" in tied else
                                                 max(tied, key=SEVERITY_ORDER.get))
        pick = next((e for e in entries if e.get("status") == status), {"note": ""})
        confs = [float(e["confidence"]) for e in entries
                 if e.get("status") == status and isinstance(e.get("confidence"), (int, float))]
        card.append({"param": key, "status": status, "turn_index": pick.get("turn_index"),
                     "quote": pick.get("quote", ""), "layer": pick.get("layer", "prompt"),
                     "note": pick.get("note", ""), "votes": statuses,
                     "confidence": round(min(confs), 2) if confs else None})

    need = (len(outs) + 1) // 2
    flags = {k: sum(bool((o.get("risk_flags") or {}).get(k)) for o in outs) >= need
             for k in ("wrong_requirement", "frustration", "loop", "low_confidence", "human_request")}
    asks = {}
    for f in rubric.FIELD_LIMITS:
        vals = sorted(int((o.get("ask_counts") or {}).get(f) or 0) for o in outs)
        asks[f] = vals[len(vals) // 2]
    base = outs[0]
    uncovered, seen = [], set()
    for o in outs:
        for u in o.get("uncovered_issues") or []:
            key = (u.get("description") or "").strip().lower()[:80]
            if key and key not in seen:
                seen.add(key)
                uncovered.append(u)
    return dict(base, scorecard=card, risk_flags=flags, ask_counts=asks, samples=len(outs),
                uncovered_issues=uncovered)


def _apply_rules(card: list[dict], out: dict, sig: dict, turns: list[dict]):
    """Deterministic checks that override or upgrade the LLM's scorecard."""
    by = {c["param"]: c for c in card}

    def upgrade(key, status, turn, quote, note):
        c = by[key]
        if SEVERITY_ORDER[status] > SEVERITY_ORDER[c["status"]]:
            c.update(status=status, turn_index=turn, quote=quote, note=note, source="rule",
                     layer=c.get("layer") or "flow")

    # Dead air: silence before a BOT turn after the buyer spoke. Measured, so it replaces the LLM's guess.
    gaps = [(cur["start"] - prev["end"], cur) for prev, cur in zip(turns, turns[1:])
            if cur["role"] == "BOT" and prev["role"] == "BUYER" and prev["end"] > 0]
    worst = max(gaps, key=lambda g: g[0], default=(0, None))
    if worst[1] is not None and worst[0] > rubric.DEAD_AIR_SEC:
        by["DEAD_AIR"].update(status="non_fatal", turn_index=worst[1]["idx"], quote="", source="rule",
                              layer="tts_audio",
                              note=f"{worst[0]:.1f}s silence before the bot replied "
                                   f"({sum(g > rubric.DEAD_AIR_SEC for g, _ in gaps)} gap(s) > {rubric.DEAD_AIR_SEC:.0f}s)")
    elif turns and turns[-1]["end"] > 0:
        by["DEAD_AIR"].update(status="ok", note="No bot-side silence above 5 s", source="rule")

    # Looping: any field asked more than 1 + 2 times.
    over = {f: n for f, n in (out.get("ask_counts") or {}).items() if isinstance(n, int) and n > rubric.LOOP_LIMIT}
    if over:
        f, n = max(over.items(), key=lambda x: x[1])
        sev = "fatal" if (out.get("risk_flags") or {}).get("frustration") else "non_fatal"
        upgrade("LOOPING", sev, None, "", f"Rule: '{f}' asked {n}× (more than 1 + 2).")

    # Internal variables / prompt text spoken aloud.
    for e in (sig.get("evidence") or {}).get("prompt_leak", [])[:1]:
        upgrade("COMPLIANCE", "fatal", e["turn"], e["quote"][:120], f"Rule: bot spoke internal content ({e['why']}).")

    for c in card:  # the sheet says some parameters can only be one way
        if c["param"] in rubric.NO_NON_FATAL and c["status"] == "non_fatal":
            c["status"] = "ok"
        if c["param"] in rubric.NO_FATAL and c["status"] == "fatal":
            c["status"] = "non_fatal"


REVIEW_SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object",
        "properties": {"param": {"type": "string"},
                       "not_issue_rule": {"type": "integer",
                                          "description": "number of the NOT-issue rule this finding matches, 0 if none"},
                       "keep": {"type": "boolean"},
                       "status": {"type": "string", "enum": ["fatal", "non_fatal"]},
                       "reason": {"type": "string"}},
        "required": ["param", "not_issue_rule", "keep", "status", "reason"]}}},
    "required": ["verdicts"],
}

REVIEW_SYSTEM = """You are a strict SECOND reviewer of a call-quality audit of IndiaMART's voice bot VANI.
A first-pass auditor marked the parameters below FATAL. Check each one against the transcript and the audit-sheet
definition, independently.
- First set not_issue_rule: the number of the NOT-issue rule that covers this finding (then keep=false), else 0.
- keep=false if it matches a NOT-issue, misreads the transcript, is unsupported by what was said, or is speculation
  (e.g. assuming a live seller existed when the transcript never shows one).
- keep=true, status=non_fatal if real but it did not mislead the buyer or corrupt the buy lead.
- keep=true, status=fatal only for a clear, evidenced match with the parameter's FATAL definition.
Return JSON only."""


def _second_review(card: list[dict], turns: list[dict]):
    fatal = [c for c in card if c["status"] == "fatal" and c.get("source") != "rule"]
    if not fatal:
        return
    defs = {k: (d, f) for k, _, d, f, _ in rubric.PARAMETERS}
    listing = "\n".join(f"- {c['param']} ({rubric.PARAM_NAMES[c['param']]}) @turn {c.get('turn_index')}: {c['note']}"
                        f" | quote: {c.get('quote', '')}\n    sheet FATAL definition: {defs[c['param']][1]}"
                        for c in fatal)
    numbered = "\n".join(f"{n}. {line[2:]}" for n, line in
                         enumerate([x for x in rubric.NOT_ISSUES.splitlines() if x.startswith("- ")], 1))
    user = (f"## Bot rules\n{rubric.BOT_RULES_DIGEST}\n## NOT-issue rules (numbered)\n{numbered}\n\n"
            f"## Transcript\n{format_transcript(turns)}\n\n## Parameters marked FATAL\n{listing}")
    try:
        verdicts = llm.judge_json(REVIEW_SYSTEM, user, REVIEW_SCHEMA, "issue_review",
                                 reasoning=config.GRADER_REASONING, max_tokens=3000)["verdicts"]
    except Exception:
        log.exception("second review failed; keeping first-pass verdicts")
        return
    by = {v.get("param"): v for v in verdicts}
    for c in fatal:
        v = by.get(c["param"])
        if not v:
            continue
        c["review"] = v.get("reason", "")
        if not v.get("keep") or int(v.get("not_issue_rule") or 0) > 0:
            c.update(status="ok", note=f"(rejected on review) {c['note']}", disputed="fatal → ok")
        elif v.get("status") == "non_fatal" and c["param"] not in rubric.NO_NON_FATAL:
            c.update(status="non_fatal", disputed="fatal → non-fatal")


def overall(card: list[dict]) -> str:
    statuses = {c["status"] for c in card}
    return "FATAL" if "fatal" in statuses else "NON_FATAL" if "non_fatal" in statuses else "PASS"


def issues_from(card: list[dict]) -> list[dict]:
    """Non-ok parameters as issue records (used by root-cause analysis and fixes)."""
    return [{"code": c["param"], "severity": c["status"], "layer": c.get("layer") or "prompt",
             "turn_index": c.get("turn_index"), "quote": c.get("quote", ""), "explanation": c.get("note", ""),
             "verified": c.get("verified", c.get("source") == "rule"), "timestamp_sec": c.get("timestamp_sec"),
             "source": c.get("source", "llm"), "review": c.get("review", "")}
            for c in card if c["status"] in ("fatal", "non_fatal")]


def grade_turns(turns: list[dict], sig: dict, note: str = "") -> dict:
    """Grade a transcript (real or simulated). Pure: returns the result, stores nothing."""
    system = SYSTEM + "\n" + rubric.rubric_text()
    if config.GRADER_INCLUDE_FULL_PROMPT:
        system += "\n## Full production prompt\n" + config.BOT_PROMPT_PATH.read_text(encoding="utf-8")
    user = (note + f"## Automatic signal hints (may be noisy)\n{_signal_hints(sig)}\n\n"
            f"## Transcript (turn index, mm:ss, speaker)\n{format_transcript(turns) or '(empty)'}")

    if not turns:
        out = {"flow": "unknown", "summary": "Empty transcript (no speech recognised).", "requirement": {},
               "outcome": "other", "ask_counts": {}, "risk_flags": {"low_confidence": True},
               "root_cause_hypothesis": "", "confidence": 0.0, "buyer_intent": "unclear",
               "scorecard": [{"param": k, "status": "na", "note": "no speech"} for k in rubric.PARAM_KEYS]}
    else:
        out = _consensus(system, user)

    card = out["scorecard"]
    for c in card:
        if c["status"] in ("fatal", "non_fatal"):
            _verify(c, turns)
    _apply_rules(card, out, sig, turns)
    _second_review(card, turns)

    grade = overall(card)
    issues = issues_from(card)
    fatal = [i for i in issues if i["severity"] == "fatal"]
    minor = [i for i in issues if i["severity"] == "non_fatal"]
    reason = ("Fatal: " + "; ".join(f"{rubric.PARAM_NAMES[i['code']]} — {i['explanation']}" for i in fatal) if fatal
              else "Non-fatal: " + "; ".join(rubric.PARAM_NAMES[i["code"]] for i in minor) if minor
              else "All audit parameters OK.")
    out["grade_reason"] = reason

    hints = sig.get("risk_hints") or {}
    flags = {k: bool((out.get("risk_flags") or {}).get(k) or hints.get(k)) for k in
             ("wrong_requirement", "frustration", "loop", "low_confidence", "human_request")}
    conf = float(out.get("confidence") or 0)
    unverified = any(not i["verified"] for i in fatal)
    needs_review = grade == "FATAL" or any(flags.values()) or conf < 0.6 or unverified
    manual = manual_review_reasons(card, out)
    return {"out": out, "scorecard": card, "issues": issues, "grade": grade, "flags": flags,
            "confidence": conf, "needs_review": needs_review, "manual_review": manual}


def manual_review_reasons(card: list[dict], out: dict) -> list[str]:
    """A human is needed only when the bot cannot stand behind its verdict:
      1. a MAJOR problem was found that no audit parameter covers, or
      2. the call's result rests entirely on low-confidence findings (every finding that makes it Fatal /
         Non-fatal is below REVIEW_CONFIDENCE).
    Everything else — including findings the second reviewer overturned — is decided by the bot."""
    reasons = []
    for u in out.get("uncovered_issues") or []:
        if u.get("severity") == "major" and (u.get("description") or "").strip():
            reasons.append(f"Not covered by the audit sheet: {u['description'].strip()}")
    grade = overall(card)
    deciding = [c for c in card if c["status"] == {"FATAL": "fatal", "NON_FATAL": "non_fatal"}.get(grade)
                and c.get("source") != "rule"]
    confs = [c.get("confidence") for c in deciding]
    if deciding and all(isinstance(x, (int, float)) and x < config.REVIEW_CONFIDENCE for x in confs):
        found = ", ".join(f"{rubric.PARAM_NAMES.get(c['param'], c['param'])} ({c['confidence']:.0%})" for c in deciding)
        reasons.append(f"The {grade.replace('_', '-').lower()} result rests only on low-confidence findings: {found}")
    return reasons


def refresh_review(call_id: str) -> list[str]:
    """Re-apply the manual-review rule to a stored grade (no new LLM call)."""
    g = db.one("SELECT scorecard_json, raw_json FROM grades WHERE call_id=?", (call_id,))
    if not g:
        return []
    reasons = manual_review_reasons(db.loads(g["scorecard_json"], []) or [], db.loads(g["raw_json"], {}) or {})
    db.execute("UPDATE grades SET manual_review=?, review_reason=? WHERE call_id=?",
               (int(bool(reasons)), "\n".join(reasons), call_id))
    return reasons


def grade_call(call_id: str) -> dict:
    t = db.one("SELECT turns_json FROM transcripts WHERE call_id=?", (call_id,))
    s = db.one("SELECT signals_json FROM signals WHERE call_id=?", (call_id,))
    turns = db.loads(t["turns_json"], [])
    sig = db.loads(s["signals_json"], {}) if s else {}
    r = grade_turns(turns, sig)
    out = r["out"]
    with db.connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO grades (call_id, grade, llm_grade, grade_reason, confidence, flow,"
            " outcome, summary, root_cause_hypothesis, issues_json, risk_flags_json, requirement_json,"
            " ask_counts_json, needs_review, model, created_at, raw_json, scorecard_json, manual_review,"
            " review_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (call_id, r["grade"], r["grade"], out.get("grade_reason"), r["confidence"], out.get("flow"),
             out.get("outcome"), out.get("summary"), out.get("root_cause_hypothesis"),
             json.dumps(r["issues"], ensure_ascii=False), json.dumps(r["flags"]),
             json.dumps(out.get("requirement") or {}, ensure_ascii=False),
             json.dumps(out.get("ask_counts") or {}), int(r["needs_review"]), llm.judge_label(),
             db.now(), json.dumps(out, ensure_ascii=False), json.dumps(r["scorecard"], ensure_ascii=False),
             int(bool(r["manual_review"])), "\n".join(r["manual_review"])),
        )
    db.set_status(call_id, "graded", graded_at=db.now())
    return out


def _safe_grade(call_id: str) -> bool:
    try:
        grade_call(call_id)
        return True
    except Exception as e:
        log.exception("grading failed for %s", call_id)
        db.set_status(call_id, "error", f"grade: {e}")
        return False


def grade_pending() -> int:
    from . import signals
    ids = [r["call_id"] for r in db.rows("SELECT call_id FROM calls WHERE status='transcribed'")]
    for cid in ids:
        signals.compute_for(cid)
    with ThreadPoolExecutor(max_workers=config.GRADER_WORKERS) as pool:
        return sum(pool.map(_safe_grade, ids))
