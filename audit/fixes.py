"""Suggest a concrete, reviewable prompt change per root cause.

The LLM sees the full production prompt with line numbers (sarvam-105b has 128K context)
plus example failing calls, and returns line-anchored edits. We apply them ourselves and
produce a unified diff + the full revised prompt (the input to the testing phase).
"""
import difflib
import re
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from rapidfuzz import fuzz

from . import analysis, config, db, llm, rubric
from .grader import format_transcript

log = logging.getLogger(__name__)

SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "array", "description": "FIRST: the exact prompt lines that cause (or fail to "
                      "prevent) this failure, and how the bot's behaviour in the examples follows from them",
                      "items": {"type": "object", "properties": {
                          "line_start": {"type": "integer"}, "line_end": {"type": "integer"},
                          "quote": {"type": "string"}, "why": {"type": "string"}},
                          "required": ["line_start", "quote", "why"]}},
        "summary": {"type": "string", "description": "One line: what the change does"},
        "rationale": {"type": "string", "description": "Why this fixes the root cause, citing the example calls"},
        "edits": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["replace", "insert_after", "delete"]},
                "line_start": {"type": "integer"},
                "line_end": {"type": "integer", "description": "inclusive; = line_start for one line / insert_after"},
                "old_text_excerpt": {"type": "string", "description": "first ~80 chars of the current text at line_start, copied exactly"},
                "new_text": {"type": "string", "description": "replacement or inserted text (markdown, same style as the prompt); empty for delete"},
                "why": {"type": "string"},
            },
            "required": ["op", "line_start", "line_end", "old_text_excerpt", "new_text", "why"]}},
        "expected_fixed_call_ids": {"type": "array", "items": {"type": "string"}},
        "regression_risk": {"type": "string", "description": "What could break; what to watch in testing"},
        "test_scenarios": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "buyer_turns": {"type": "array", "items": {"type": "string"},
                                "description": "What a simulated buyer says, in order, in the call's language"},
                "expected_bot_behavior": {"type": "string"},
            },
            "required": ["title", "buyer_turns", "expected_bot_behavior"]}},
    },
    "required": ["diagnosis", "summary", "rationale", "edits", "expected_fixed_call_ids", "regression_risk",
                 "test_scenarios"],
}

_STOP = set("the a an and or to of in on for with is are be by as at it this that not no if when then bot buyer "
            "call calls seller prompt instruction instructions without".split())


def _sections(lines: list[str]) -> list[tuple[int, int]]:
    """Split the prompt into heading-delimited chunks: [(start, end_exclusive)], 0-based."""
    heads = [i for i, l in enumerate(lines)
             if l.startswith(("## ", "**", "{%")) or l.strip() in ("Enrichment \\- ", "Redial\\_AST")]
    heads = sorted(set([0] + heads + [len(lines)]))
    out, start = [], 0
    for h in heads[1:]:
        if h - start >= 3 or h == len(lines):
            out.append((start, h))
            start = h
    return out


def ranked_sections(lines: list[str], query: str) -> list[tuple[float, int, int]]:
    words = {w for w in re.findall(r"[a-z_]{3,}", query.lower()) if w not in _STOP}
    scored = []
    for s, e in _sections(lines):
        text = " ".join(lines[s:e]).lower().replace("\\_", "_")
        score = sum(text.count(w) for w in words) / (1 + (e - s) ** 0.5)
        scored.append((score, s, e))
    return sorted(scored, reverse=True)


def relevant_excerpt(lines: list[str], query: str, budget_lines: int = 260) -> str:
    """Keyword-retrieve the prompt sections most related to the root cause, keeping L-numbers."""
    picked, used = [], 0
    for score, s, e in ranked_sections(lines, query):
        if score <= 0 or used >= budget_lines:
            break
        picked.append((s, min(e, s + 80)))
        used += min(e, s + 80) - s
    picked.sort()
    return "\n...\n".join("\n".join(f"L{i + 1:04d}| {lines[i]}" for i in range(s, e)) for s, e in picked)

SYSTEM = f"""You are a prompt engineer for IndiaMART's buyer-side voice bot VANI (Sarvam Samvaad agent).
Given ONE root cause found by auditing real calls, the failing examples, and the relevant sections of the production prompt
(line-numbered), propose the SMALLEST precise change to the prompt that fixes the root cause.

Rules:
- Edits are line-anchored: line_start/line_end refer to the L-numbers shown. old_text_excerpt must be copied
  exactly from line_start so we can verify the anchor. Prefer 1-3 surgical edits over rewrites.
- Keep the prompt's style (markdown bullets, Jinja {{% if %}} blocks, ﻿@ variable﻿ notation). Never break a Jinja block.
- If the prompt contradicts itself on this topic, resolve the contradiction explicitly.
- Write instructions the model can follow turn-by-turn: concrete trigger → concrete action, with a short
  example utterance in Hinglish where helpful. No generic advice ("be more careful").
- If the root cause is ASR/audio and cannot be fixed in the prompt, say so in the rationale and give the
  best prompt-side mitigation (e.g. confirm-back for low-confidence product names).
- test_scenarios: 2-4 short simulated-buyer scripts that reproduce the failure, for replay testing.
Known prompt contradictions:
{chr(10).join('- ' + c for c in rubric.PROMPT_CONFLICTS)}
Return JSON only."""


def _numbered(lines: list[str]) -> str:
    return "\n".join(f"L{i + 1:04d}| {line}" for i, line in enumerate(lines))


def _examples(rc_id: str, k: int = 5) -> str:
    members = db.rows("SELECT cr.call_id, cr.is_fatal, cr.issue_idx, g.summary, g.issues_json,"
                      " t.turns_json FROM call_root_causes cr JOIN grades g USING(call_id)"
                      " JOIN transcripts t USING(call_id) WHERE cr.rc_id=?"
                      " ORDER BY cr.is_fatal DESC LIMIT ?", (rc_id, k))
    blocks = []
    for m in members:
        issues = db.loads(m["issues_json"], [])
        idxs = [i for i in db.loads(m["issue_idx"], []) if 0 <= i < len(issues)]
        turns = db.loads(m["turns_json"], [])
        focus = {issues[i]["turn_index"] for i in idxs if isinstance(issues[i].get("turn_index"), int)}
        keep = {j for f in focus for j in range(f - 4, f + 3)} if focus else set(range(30))
        excerpt = format_transcript([t for t in turns if t["idx"] in keep])
        iss = "\n".join(f"  - {issues[i]['code']} @turn {issues[i].get('turn_index')}: {issues[i]['explanation']}"
                        for i in idxs)
        blocks.append(f"### Call {m['call_id']} ({'FATAL' if m['is_fatal'] else 'non-fatal'})\n"
                      f"Summary: {m['summary']}\nIssues:\n{iss}\nTranscript excerpt:\n{excerpt}")
    return "\n\n".join(blocks)


def apply_edits(lines: list[str], edits: list[dict]) -> tuple[list[str], list[dict]]:
    """Apply line-anchored edits bottom-up. Edits whose anchor text doesn't match are skipped."""
    out = list(lines)
    for e in edits:  # models often miscount by a few lines: snap to where the quoted anchor really is
        anchor = (e.get("old_text_excerpt") or "").strip()[:200]
        s = e.get("line_start", 0) - 1
        if anchor:
            near = range(max(0, s - 8), min(len(lines), s + 9))
            best = max(near, key=lambda i: fuzz.partial_ratio(anchor, lines[i]), default=s)
            if best != s and fuzz.partial_ratio(anchor, lines[best]) >= 85:
                shift = best - s
                e["line_start"] = best + 1
                e["line_end"] = e.get("line_end", s + 1) + shift
    for e in sorted(edits, key=lambda e: e.get("line_start", 0), reverse=True):
        s, end = e.get("line_start", 0) - 1, e.get("line_end", e.get("line_start", 0)) - 1
        end = max(end, s)
        ok = 0 <= s <= end < len(lines)
        anchor = (e.get("old_text_excerpt") or "").strip()
        if ok and anchor:
            window = " ".join(lines[s:min(end + 1, s + 3)])
            ok = fuzz.partial_ratio(anchor[:200], window) >= 70
        e["applied"] = bool(ok)
        if not ok:
            continue
        new = (e.get("new_text") or "").splitlines()
        if e["op"] == "insert_after" and new and fuzz.partial_ratio(new[0].strip(), lines[end].strip()) >= 90:
            new = new[1:]  # model repeated the anchor line before the inserted text
        if e["op"] == "insert_after":
            out[end + 1:end + 1] = new
        elif e["op"] == "delete":
            del out[s:end + 1]
        else:
            out[s:end + 1] = new
    return out, edits


def generate(rc_id: str) -> str:
    rc = db.one("SELECT * FROM root_causes WHERE rc_id=?", (rc_id,))
    prompt_lines = config.BOT_PROMPT_PATH.read_text(encoding="utf-8").splitlines()
    p = propose(rc["name"], rc["description"], f"layer: {rc['layer']}, affects {rc['n_calls']} calls, "
                f"{rc['n_fatal']} fatal", _examples(rc_id), prompt_lines, tag=rc_id)
    out, revised, edits = p["out"], p["revised"], p["edits"]
    n = (db.one("SELECT COUNT(*) AS n FROM fixes WHERE rc_id=?", (rc_id,)) or {"n": 0})["n"] + 1
    fix_id = f"{rc_id}-F{n}"
    path = config.FIX_DIR / f"{fix_id}.md"
    path.write_text("\n".join(revised) + "\n", encoding="utf-8")
    db.execute(
        "INSERT INTO fixes (fix_id, rc_id, created_at, summary, rationale, edits_json, diff,"
        " revised_prompt_path, expected_call_ids_json, regression_risk, status) VALUES"
        " (?,?,?,?,?,?,?,?,?,?, 'proposed')",
        (fix_id, rc_id, db.now(), out.get("summary"), out.get("rationale"),
         json.dumps({"edits": edits, "test_scenarios": out.get("test_scenarios", []),
                     "diagnosis": out.get("diagnosis", [])}, ensure_ascii=False),
         p["diff"], str(path), json.dumps(out.get("expected_fixed_call_ids", [])), out.get("regression_risk")),
    )
    return fix_id


def propose(name: str, description: str, stats: str, examples: str, prompt_lines: list[str],
            tag: str = "") -> dict:
    """Ask for a correction of one problem against a given prompt; return the revised prompt + diff."""
    excerpt = relevant_excerpt(prompt_lines, f"{name} {description} {examples}")
    user = (f"## Root cause\n{name} ({stats})\n{description}\n\n## Failing examples\n{examples}\n\n"
            f"## Relevant sections of the production prompt (line-numbered; '...' = omitted)\n{excerpt}")
    existing = [l.strip() for l in prompt_lines if len(l.strip()) > 30]
    rc = {"name": name, "description": description}
    rc_id = tag or name

    def attempt(feedback: str = ""):
        out = llm.chat_json(SYSTEM, user + feedback, SCHEMA, "prompt_fix",
                            reasoning=config.ANALYSIS_REASONING, max_tokens=8000)
        edits = out.get("edits", [])
        for e in edits:  # an edit that only re-adds text already in the prompt is not a fix
            new = [l.strip() for l in (e.get("new_text") or "").splitlines() if len(l.strip()) > 30]
            if e.get("op") != "delete" and new and all(
                    max((fuzz.ratio(n.replace("\\", ""), x.replace("\\", "").replace("﻿", ""))
                         for x in existing), default=0) >= 90 for n in new):
                e["skip_reason"] = "duplicates existing prompt text"
        revised, applied = apply_edits(prompt_lines, [e for e in edits if not e.get("skip_reason")])
        return out, revised, applied + [e for e in edits if e.get("skip_reason")]

    out, revised, edits = attempt()
    if not any(e.get("applied") for e in edits):
        bad = "; ".join(f"L{e.get('line_start')} '{(e.get('old_text_excerpt') or '')[:60]}'"
                        + (f" ({e['skip_reason']})" if e.get("skip_reason") else "") for e in edits)
        log.warning("%s: no edit anchored (%s); retrying with feedback", rc_id, bad)
        out, revised, edits = attempt(
            f"\n\n## Your previous answer could not be applied\nAnchors that failed: {bad}.\n"
            "old_text_excerpt must be copied character-for-character from an L-numbered line of the PROMPT "
            "excerpt above (never from the root cause or examples), and line_start must be that L-number. "
            "new_text must add or change an instruction, not repeat existing text.")
    if not any(e.get("applied") for e in edits):
        # Last resort: keep the proposed instruction, placed at the end of the most relevant section.
        block = next((e for e in edits if (e.get("new_text") or "").strip() and not e.get("skip_reason")), None)
        top = ranked_sections(prompt_lines, f"{rc['name']} {rc['description']}")
        if block and top:
            _, s, e_ = top[0]
            new = [l for l in block["new_text"].splitlines()
                   if fuzz.partial_ratio(rc["description"], l) < 90 and fuzz.partial_ratio(rc["name"], l) < 90]
            end = e_ - 1
            while end > s and not prompt_lines[end].strip():
                end -= 1
            revised = prompt_lines[:end + 1] + new + prompt_lines[end + 1:]
            block.update(applied=True, placed="end of most relevant section", line_start=end + 1,
                         line_end=end + 1, op="insert_after")
            log.warning("%s: placed fix at end of section L%d-L%d", rc_id, s + 1, e_)
    diff = "\n".join(difflib.unified_diff(prompt_lines, revised, "current_prompt.md",
                                          "proposed_prompt.md", lineterm="", n=2))
    return {"out": out, "revised": revised, "edits": edits, "diff": diff}


def generate_top(k: int = 5) -> list[str]:
    run_id = analysis.latest_run_id()
    if run_id is None:
        return []
    rcs = db.rows("SELECT rc_id FROM root_causes WHERE run_id=? AND n_calls>0 AND rc_id NOT IN"
                  " (SELECT rc_id FROM fixes) ORDER BY rank LIMIT ?", (run_id, k))

    def safe(rc_id):
        try:
            return generate(rc_id)
        except Exception:
            log.exception("fix generation failed for %s", rc_id)

    with ThreadPoolExecutor(max_workers=3) as pool:
        return [f for f in pool.map(safe, [r["rc_id"] for r in rcs]) if f]


def build_version(fix_ids: list[str] | None = None, notes: str = "") -> str:
    """Step 5: one corrected copy of the prompt with the chosen fixes merged in.

    Each fix is stored as a full revised prompt; we take its changed regions against the original
    and apply all of them bottom-up. Overlapping regions keep the higher-ranked fix and skip the rest.
    """
    if fix_ids is None:
        fix_ids = [r["fix_id"] for r in db.rows(
            "SELECT f.fix_id FROM fixes f JOIN root_causes r USING(rc_id)"
            " WHERE f.status != 'rejected' AND COALESCE(f.selected, 1) = 1"
            " AND r.run_id = (SELECT MAX(run_id) FROM analysis_runs) ORDER BY r.rank")]
    original = config.BOT_PROMPT_PATH.read_text(encoding="utf-8").splitlines()
    changes, used, skipped = [], [], []
    for fid in fix_ids:
        f = db.one("SELECT revised_prompt_path FROM fixes WHERE fix_id=?", (fid,))
        if not f or not Path(f["revised_prompt_path"] or "").exists():
            continue
        revised = Path(f["revised_prompt_path"]).read_text(encoding="utf-8").splitlines()
        ops = [op for op in difflib.SequenceMatcher(a=original, b=revised, autojunk=False).get_opcodes()
               if op[0] != "equal"]
        if any(not (i2 <= a1 or i1 >= a2) or (i1 == i2 == a1 == a2)
               for (_, i1, i2, _, _) in ops for (a1, a2, _) in changes):
            skipped.append(fid)
            continue
        changes += [(i1, i2, revised[j1:j2]) for (_, i1, i2, j1, j2) in ops]
        used.append(fid)
    merged = list(original)
    for i1, i2, new in sorted(changes, key=lambda c: (c[0], c[1]), reverse=True):
        merged[i1:i2] = new
    note = notes + (f" Skipped overlapping: {', '.join(skipped)}." if skipped else "")
    vid = save_version(merged, used, note.strip())
    log.info("prompt %s built from %s (skipped %s)", vid, used, skipped)
    return vid


def save_version(lines: list[str], fix_ids: list[str], notes: str = "") -> str:
    """Store a full prompt copy as the next version (v1, v2, …); the diff is always against the original."""
    original = config.BOT_PROMPT_PATH.read_text(encoding="utf-8").splitlines()
    n = (db.one("SELECT COUNT(*) AS n FROM prompt_versions") or {"n": 0})["n"] + 1
    vid = f"v{n}"
    path = config.PROMPT_VERSION_DIR / f"vani_prompt_{vid}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    diff = "\n".join(difflib.unified_diff(original, lines, "original_prompt.md", path.name, lineterm="", n=2))
    db.execute("INSERT INTO prompt_versions (version_id, created_at, fix_ids, path, diff, notes) VALUES (?,?,?,?,?,?)",
               (vid, db.now(), json.dumps(fix_ids), str(path), diff, notes))
    return vid


def latest_version() -> dict | None:
    return db.one("SELECT * FROM prompt_versions ORDER BY created_at DESC, version_id DESC LIMIT 1")
