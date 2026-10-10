"""Offline end-to-end check of the whole loop with stubbed Sarvam STT + LLM (no API key, no credits).

    python scripts/smoke_test.py            # uses a throwaway DB in data/smoke/
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / "data" / "smoke"
shutil.rmtree(SMOKE, ignore_errors=True)
SMOKE.mkdir(parents=True)
os.environ["AUDIT_DB"] = str(SMOKE / "audit.db")
os.environ["INBOX_DIR"] = str(SMOKE / "inbox")
sys.path.insert(0, str(ROOT))

from audit import config  # noqa: E402

for name in ("FIX_DIR", "AUDIO_DIR", "TRANSCRIPT_COPY_DIR", "PROMPT_VERSION_DIR", "TESTCASE_DIR"):
    d = SMOKE / name.lower()
    d.mkdir()
    setattr(config, name, d)  # keep smoke output out of the real data folders

from audit import agreement, analysis, db, fixes, grader, ingest, llm, rubric, simulate, transcribe  # noqa: E402

FAKE_STT = {
    "language_code": "hi-IN",
    "diarized_transcript": {"entries": [
        {"speaker_id": "1", "start_time_seconds": 0.5, "end_time_seconds": 4.0, "transcript": "नमस्ते, क्या आपको सोलर पैनल की requirement है?"},
        {"speaker_id": "0", "start_time_seconds": 4.5, "end_time_seconds": 5.5, "transcript": "हां जी"},
        {"speaker_id": "1", "start_time_seconds": 6.0, "end_time_seconds": 11.0, "transcript": "सेलर अभी बिज़ी हैं. आपको कितने पीस चाहिए?"},
        {"speaker_id": "0", "start_time_seconds": 12.0, "end_time_seconds": 13.0, "transcript": "पचास हज़ार का बजट है"},
        {"speaker_id": "1", "start_time_seconds": 13.5, "end_time_seconds": 16.0, "transcript": "ठीक है, आपको कितने पीस चाहिए?"},
        {"speaker_id": "0", "start_time_seconds": 16.5, "end_time_seconds": 17.5, "transcript": "बोला ना बजट पचास हज़ार"},
        {"speaker_id": "1", "start_time_seconds": 25.0, "end_time_seconds": 27.5, "transcript": "जी, आपको कितने पीस चाहिए?"},
        {"speaker_id": "0", "start_time_seconds": 28.0, "end_time_seconds": 29.0, "transcript": "किसी इंसान से बात कराओ"},
        {"speaker_id": "1", "start_time_seconds": 29.5, "end_time_seconds": 32.0, "transcript": "आपका नाम क्या है?"},
    ]},
}
FIX_MARKER = "If the buyer answers with a budget"


def card(**bad):
    return [{"param": k, "status": bad.get(k, ("ok", ""))[0], "turn_index": 6, "quote": bad.get(k, ("", ""))[1],
             "layer": "prompt", "note": f"note for {k}"} for k in rubric.PARAM_KEYS]


def fake_json(system, user, schema, name, **kw):
    if name == "call_grade":
        looping = "कितने पीस" in user and user.count("कितने पीस") >= 3
        sim_fixed = "मात्रा आप सेलर से" in user
        bad = {}
        if looping and not sim_fixed:
            bad["LOOPING"] = ("fatal", "आपको कितने पीस चाहिए")
            bad["QTY_PROBING"] = ("non_fatal", "आपको कितने पीस चाहिए")
        bad["WRONG_INFO"] = ("fatal", "this quote is not in the transcript")
        return {"flow": "helpdesk", "language": "hi", "summary": "Buyer gives budget; bot loops on quantity.",
                "buyer_intent": "buying", "requirement": {"stated": {"product": "solar panel"}, "captured": {}},
                "outcome": "buyer_dropped", "ask_counts": {"quantity": 3}, "scorecard": card(**bad),
                "risk_flags": {"frustration": True}, "root_cause_hypothesis": "No rule for budget answers.",
                "confidence": 0.8}
    if name == "issue_review":  # reject the made-up WRONG_INFO, keep LOOPING
        return {"verdicts": [{"param": "WRONG_INFO", "keep": False, "status": "fatal", "reason": "not supported"},
                             {"param": "LOOPING", "keep": True, "status": "fatal", "reason": "3 asks + irritation"}]}
    if name == "rc_taxonomy":
        return {"root_causes": [{"key": "RC01", "name": "Budget given instead of quantity",
                                 "description": "No rule for budget answers.", "layer": "prompt"}]}
    if name == "rc_assign":
        ids = [line.split()[1] for line in user.splitlines() if line.startswith("CALL ")]
        return {"assignments": [{"call_id": c, "links": [{"rc_key": "RC01", "issue_indexes": [0, 1]}]} for c in ids]}
    if name == "prompt_fix":
        return {"diagnosis": [{"line_start": 897, "quote": "On a valid response", "why": "no budget branch"}],
                "summary": "Treat a budget answer as quantity ambiguity once, then defer.",
                "rationale": "Calls loop because budget answers are re-asked verbatim.",
                "edits": [{"op": "insert_after", "line_start": 899, "line_end": 899,  # off by 2: must snap to 897
                           "old_text_excerpt": "* On a valid response, update",
                           "new_text": "* " + FIX_MARKER + ", acknowledge it once and ask for an approximate count; "
                                       "if they repeat the budget, defer quantity to the seller.", "why": "budget"}],
                "expected_fixed_call_ids": [], "regression_risk": "May under-collect quantity.", "test_scenarios": []}
    if name == "test_case":
        return {"flow": "helpdesk", "product_name": "Solar Panel", "language_name": "hindi",
                "live_seller_available": False, "quantity_unit_options": ["Piece"],
                "specification_options": {"Type": ["Mono", "Poly"]}, "buyer_behaviour": "Gives budget, not quantity.",
                "checks": ["Defer quantity to seller after a repeated budget answer"]}
    raise AssertionError(name)


def fake_msgs(messages, schema, name, **kw):
    system, convo = messages[0]["content"], messages[1:]
    if name == "sim_buyer":
        n_said = sum(1 for m in convo if m["role"] == "assistant")
        assert "Buyer script" in system and "पचास हज़ार" in system, "buyer script missing"
        if n_said >= 3:
            return {"say": "ठीक है", "hang_up": True}
        return {"say": ["हां जी", "पचास हज़ार का बजट है", "बोला ना बजट पचास हज़ार"][n_said], "hang_up": False}
    if name == "sim_bot":
        assert "{%" not in system and "﻿" not in system, "prompt not fully rendered"
        n_buyer = sum(1 for m in convo if m["role"] == "user") - 1
        if FIX_MARKER in system and n_buyer >= 2:
            return {"say": "कोई बात नहीं, मात्रा आप सेलर से डिस्कस कर लीजिए. आपका नाम क्या है?",
                    "variable_updates": {"@call_step": "LOOP_DETECTED"}, "tool_calls": []}
        if n_buyer == 1:
            return {"say": "सेलर अभी बिज़ी हैं. आपको कितने पीस चाहिए?", "variable_updates": {"identity_revealed": "1"},
                    "tool_calls": [{"name": "requested_another_product_tool", "args": {"product_name": "Solar Panel"}}]}
        return {"say": "जी, आपको कितने पीस चाहिए?", "variable_updates": {}, "tool_calls": []}
    if name in ("call_grade", "issue_review", "test_case", "prompt_fix", "rc_taxonomy", "rc_assign"):
        return fake_json(system, messages[-1]["content"], schema, name, **kw)
    raise AssertionError(name)


llm.chat_json = fake_json
llm.chat_json_messages = fake_msgs
config.GRADER_SAMPLES = 1
db.init()

# ① upload  ② STT + saved copy
samples = sorted((Path.home() / "Downloads").glob("call_*.mp3"))[:2]
assert samples, "no sample mp3s in Downloads"
for s in samples:
    print("register", ingest.register(s))
for c in db.rows("SELECT call_id FROM calls"):
    transcribe.store_transcript(c["call_id"], FAKE_STT, None)
assert list(config.TRANSCRIPT_COPY_DIR.glob("*.txt")), "transcript copy not saved"

# ③ issues
print("audited", grader.grade_pending())
g = db.one("SELECT * FROM grades")
sc = {c["param"]: c for c in db.loads(g["scorecard_json"])}
print("grade", g["grade"], "|", g["grade_reason"])
assert g["grade"] == "FATAL" and sc["LOOPING"]["status"] == "fatal"
assert sc["WRONG_INFO"]["status"] == "ok", "second review should have rejected WRONG_INFO"
assert sc["DEAD_AIR"]["status"] == "non_fatal", sc["DEAD_AIR"]  # 7.5 s gap before turn 6

# ④ corrections  ⑤ corrected prompt
analysis.run()
fid = fixes.generate_top(1)
print("fix", fid)
vid = fixes.build_version()
v = db.one("SELECT * FROM prompt_versions WHERE version_id=?", (vid,))
print("version", vid, "\n", v["diff"][:500])
assert FIX_MARKER in Path(v["path"]).read_text(encoding="utf-8")
assert "@@ -896" in v["diff"] or "@@ -897" in v["diff"], "edit did not snap to the anchor line"

# ⑥ test setup  ⑧ run  ⑨ compare
print("test cases", simulate.build_test_cases())
case = simulate.load_case(db.one("SELECT case_id FROM test_cases")["case_id"])
assert case["buyer_script"][1] == "पचास हज़ार का बजट है" and case["context"]["initial_message"].startswith("नमस्ते")
config.SIM_MAX_TURNS = 6
run_id = simulate.run_test(vid, [r["case_id"] for r in db.rows("SELECT case_id FROM test_cases")])
tr = db.one("SELECT * FROM test_runs WHERE run_id=?", (run_id,))
summary = db.loads(tr["summary_json"])
print("run", tr["status"], summary["original"]["param_fails"]["LOOPING"], "→", summary["corrected"]["param_fails"]["LOOPING"])
assert tr["status"] == "done" and summary["errors"] == 0
assert summary["original"]["param_fails"]["LOOPING"] > summary["corrected"]["param_fails"]["LOOPING"] == 0

# ⑦ manual test engine
bot = simulate.new_manual_bot("corrected", vid, case["context"])
turns = [{"idx": 0, "role": "BOT", "text": bot.opener()["text"]}, {"idx": 1, "role": "BUYER", "text": "हां जी"}]
r = bot.reply(turns)
assert r["text"] and bot.vars["identity_revealed"] == "1"

cid = db.one("SELECT call_id FROM calls")["call_id"]
agreement.set_label(cid, "FATAL")
print("agreement", {k: v for k, v in agreement.metrics().items() if k != "disagreements"})
print("SMOKE OK")
