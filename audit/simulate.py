"""Steps 6-9: test setup from recordings, simulated replay, before/after comparison.

Test case (step 6), built once per recording from its saved transcript:
    context       — what the bot knew at call start (product, flow, live seller, units, specs, opener)
    buyer_script  — the buyer's own lines from the recording, in order
    buyer_behaviour / checks — how the buyer behaved and what the bot must get right

Test run (step 8): every chosen case is replayed twice — ORIGINAL prompt and CORRECTED prompt —
by a simulated buyer that follows the buyer_script, then both conversations are scored with the
same audit sheet as the real calls (step 9 compares them parameter by parameter).

Bot under test (config.SIM_BACKEND):
  local   — we render the production prompt for the call (Jinja + variables), run it on a
            Sarvam LLM turn by turn, apply its @variable updates, and stub its tools.
  samvaad — your real Sarvam agent over the Conv AI SDK in CHAT mode (one committed app version per prompt).
"""
import asyncio
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, db, grader, llm, prompt_render, rubric, signals

log = logging.getLogger(__name__)

# ───────────────────────────── step 6: test setup ─────────────────────────────

CASE_SCHEMA = {
    "type": "object",
    "properties": {
        "flow": {"type": "string", "enum": list(prompt_render.FLOW_SECTIONS)},
        "product_name": {"type": "string"},
        "seller_name": {"type": "string"},
        "language_name": {"type": "string", "enum": ["hindi", "english", "telugu"]},
        "live_seller_available": {"type": "boolean"},
        "known_buyer_name": {"type": "string"},
        "known_buyer_city": {"type": "string"},
        "quantity_unit_options": {"type": "array", "items": {"type": "string"}},
        "specification_options": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "string"}}},
        "buyer_behaviour": {"type": "string", "description": "2-3 lines: how this buyer behaves (what they ask, how they answer, quirks)"},
        "checks": {"type": "array", "items": {"type": "string"},
                   "description": "3-6 things the bot must do right in this conversation, from the audit findings and the call"},
    },
    "required": ["flow", "product_name", "language_name", "live_seller_available", "quantity_unit_options",
                 "specification_options", "buyer_behaviour", "checks"],
}

CASE_SYSTEM = """You turn a real IndiaMART voice-bot call (bot = VANI) into a replayable test case.
From the transcript and the audit findings infer:
- flow: helpdesk (buyer's call to a seller was redirected), enrichment (buyer already has a buy lead), redial_no_name_city, redial_ast
- product_name: the product in context when the call started (English), "" if the bot had none
- seller_name: the seller the buyer was calling, "" if unknown
- language_name: hindi | english | telugu
- live_seller_available: true ONLY if the bot offered to connect to a seller on the line or a transfer happened
- known_buyer_name / known_buyer_city: only if the bot already knew them without asking, else ""
- quantity_unit_options: 1-3 sensible units for this product (e.g. ["Piece"], ["Kilogram","Ton"])
- specification_options: 0-2 key specs for this product with 2-4 options each, e.g. {"Thickness": ["4 inch","6 inch"]}
- buyer_behaviour: how this buyer talks and behaves (asks price first, gives district instead of city, short answers…)
- checks: what the bot must get right with THIS buyer (include each audit finding as a check, phrased as expected behaviour)
Return JSON only."""


def build_test_case(call_id: str) -> dict:
    c = db.one("SELECT filename FROM calls WHERE call_id=?", (call_id,))
    t = db.one("SELECT turns_json FROM transcripts WHERE call_id=?", (call_id,))
    g = db.one("SELECT grade, issues_json, flow FROM grades WHERE call_id=?", (call_id,))
    turns = db.loads(t["turns_json"], [])
    issues = db.loads(g["issues_json"], []) if g else []
    findings = "\n".join(f"- {rubric.PARAM_NAMES.get(i['code'], i['code'])} ({i['severity']}): {i['explanation']}"
                         for i in issues) or "- none (clean call)"
    ctx = llm.chat_json(CASE_SYSTEM, f"## Transcript\n{grader.format_transcript(turns)}\n\n## Audit findings\n{findings}",
                        CASE_SCHEMA, "test_case", reasoning="off", max_tokens=3000)
    first_bot = next((x["text"] for x in turns if x["role"] == "BOT"), "")
    case = {
        "case_id": call_id, "title": c["filename"], "call_id": call_id,
        "original_grade": g["grade"] if g else None,
        "context": {
            "flow": (g and g["flow"]) or ctx.get("flow") or "helpdesk",
            "product_name": ctx.get("product_name", ""), "seller_name": ctx.get("seller_name", ""),
            "language_name": ctx.get("language_name", "hindi"),
            "live_seller_available": bool(ctx.get("live_seller_available")),
            "known_buyer_name": ctx.get("known_buyer_name", ""), "known_buyer_city": ctx.get("known_buyer_city", ""),
            "quantity_unit_options": ctx.get("quantity_unit_options") or [],
            "specification_options": ctx.get("specification_options") or {},
            "initial_message": first_bot,  # the bot's real opener from the recording
        },
        "buyer_script": [x["text"] for x in turns if x["role"] == "BUYER" and x["text"].strip()],
        "buyer_behaviour": ctx.get("buyer_behaviour", ""),
        "checks": ctx.get("checks", []),
    }
    path = config.TESTCASE_DIR / f"{Path(c['filename']).stem}.json"
    path.write_text(json.dumps(case, ensure_ascii=False, indent=1), encoding="utf-8")
    db.execute("INSERT OR REPLACE INTO test_cases (case_id, created_at, title, case_json, path) VALUES (?,?,?,?,?)",
               (call_id, db.now(), c["filename"], json.dumps(case, ensure_ascii=False), str(path)))
    return case


def build_test_cases(call_ids: list[str] | None = None) -> int:
    if call_ids is None:  # every audited call that does not have a test case yet
        call_ids = [r["call_id"] for r in db.rows(
            "SELECT call_id FROM grades WHERE call_id NOT IN (SELECT case_id FROM test_cases)")]
    ok = 0
    with ThreadPoolExecutor(max_workers=config.GRADER_WORKERS) as pool:
        for r in pool.map(lambda cid: _safe(build_test_case, cid), call_ids):
            ok += r is not None
    return ok


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception:
        log.exception("%s failed for %s", fn.__name__, a)
        return None


def load_case(case_id: str) -> dict | None:
    r = db.one("SELECT case_json FROM test_cases WHERE case_id=?", (case_id,))
    return db.loads(r["case_json"]) if r else None


def initial_variables(ctx: dict) -> dict:
    live = bool(ctx.get("live_seller_available"))
    return {
        "bot_name": "Vani", "product_name": ctx.get("product_name", ""),
        "seller_name": ctx.get("seller_name", ""),
        "quantity_unit_options": ctx.get("quantity_unit_options") or [],
        "specification_options": ctx.get("specification_options") or {},
        "buyer_name": ctx.get("known_buyer_name", ""), "updated_buyer_name": "",
        "buyer_city": ctx.get("known_buyer_city", ""), "updated_city_name": "", "updated_state_name": "",
        "triangulation_response_value": "0", "triangulation_response_city": "",
        "is_enrich": "1" if ctx.get("flow") == "enrichment" else "0",
        "ast_seller_pns": "9999999999" if live else "", "ast_flow_live": "true" if live else "false",
        "lead_type": "LIVE" if live else "", "ast_seller_company": "", "ast_seller_city": "",
        "identity_revealed": "0", "bl_converted": "0", "identity_challenge_count": "0",
        "off_topic_counter": "0", "seller_ask_count": "0", "pcd_pitch": "false",
        "already_bought_handled": "0", "language_name": ctx.get("language_name", "hindi"),
        "initial_message": ctx.get("initial_message", ""), "current_phase": "REQUIREMENT CONFIRMATION",
        "turn_intent": "", "quantity": "", "quantity_unit": "", "specifications": "",
        "additional_details": "", "image_reference": "False", "call_step": "",
    }

# ───────────────────────────── simulated buyer ─────────────────────────────

BUYER_SCHEMA = {"type": "object", "properties": {"say": {"type": "string"}, "hang_up": {"type": "boolean"}},
                "required": ["say", "hang_up"]}

BUYER_SYSTEM = """You role-play a BUYER on a phone call with IndiaMART's AI assistant, to test the assistant.
Stay in character as the buyer of the recorded call below:
- Follow the buyer's script: say the same things, in the same order, in the same language/register, when the
  assistant gives you the chance. Adapt the wording only so it fits what the assistant just said.
- Keep the buyer's behaviour (e.g. asks price first, answers with a district instead of a city, short answers).
- Never volunteer information the real buyer did not give. Speak like a phone caller: short, one sentence.
- hang_up=true when the assistant has closed or transferred the call, or the script is finished and the call is over.
Return JSON only: {"say": "...", "hang_up": false}"""


def buyer_step(case: dict, turns: list[dict]) -> dict:
    persona = ("## Buyer script (from the recording, in order)\n"
               + "\n".join(f"{n + 1}. {line}" for n, line in enumerate(case.get("buyer_script", [])))
               + f"\n\n## Buyer behaviour\n{case.get('buyer_behaviour', '')}")
    msgs = [{"role": "system", "content": BUYER_SYSTEM + "\n\n" + persona}]
    for t in turns:  # from the buyer's point of view the bot is the 'user'
        msgs.append({"role": "user" if t["role"] == "BOT" else "assistant", "content": t["text"] or "…"})
    out = llm.chat_json_messages(msgs, BUYER_SCHEMA, "sim_buyer", model=config.SIM_BUYER_MODEL,
                                 reasoning="off", max_tokens=1500, temperature=0.3)
    return {"say": (out.get("say") or "").strip(), "hang_up": bool(out.get("hang_up"))}

# ───────────────────────────── bot under test: local ─────────────────────────────

BOT_SCHEMA = {
    "type": "object",
    "properties": {
        "variable_updates": {"type": "object", "additionalProperties": {"type": "string"}},
        "tool_calls": {"type": "array", "items": {
            "type": "object", "properties": {"name": {"type": "string"}, "args": {"type": "object"}},
            "required": ["name"]}},
        "say": {"type": "string"},
        "end_interaction": {"type": "boolean"},
    },
    "required": ["say"],
}

RUNTIME = """

## RUNTIME PROTOCOL (simulation harness — not part of your persona)
You are live on this phone call. For every turn reply with JSON only:
{{"variable_updates": {{"var": "value"}}, "tool_calls": [{{"name": "tool", "args": {{}}}}], "say": "...", "end_interaction": false}}
- variable_updates: every `@variable` your instructions tell you to set or increment this turn (string values).
- tool_calls: tools you invoke this turn. Available: {tools}. Results appear below on your next turn.
- say: exactly what you speak to the buyer (follow the TTS rules). May be empty only if you are waiting for a tool result.
- end_interaction: true when your instructions say to invoke end_interaction.

Current variable values: {vars}
{tool_results}"""

TRACKED_VARS = [
    "product_name", "current_phase", "identity_revealed", "bl_converted", "identity_challenge_count",
    "seller_ask_count", "off_topic_counter", "quantity", "quantity_unit", "quantity_asked_count",
    "specifications", "updated_buyer_name", "updated_city_name", "updated_state_name", "buyer_disposition",
    "call_outcome", "connect_ast_call", "pcd_pitch", "turn_intent", "call_step", "additional_details",
    "ast_seller_company", "ast_seller_city",
]


def _run_tool(name: str, args: dict, vars_: dict, ctx: dict) -> str:
    args = args or {}
    first = next(iter(args.values()), "") if args else ""
    if name == "requested_another_product_tool":
        product = str(args.get("product_name") or args.get("product") or first or vars_.get("product_name"))
        same = product.lower() == str(ctx.get("product_name", "")).lower()
        vars_["product_name"] = product
        if not same:
            vars_["quantity_unit_options"], vars_["specification_options"] = ["Piece"], {}
        return (f"requested_another_product_tool → product loaded: {product}; quantity_unit_options="
                f"{json.dumps(vars_['quantity_unit_options'])}; specification_options="
                f"{json.dumps(vars_['specification_options'], ensure_ascii=False)}")
    if name == "update_ast_buy_variables_by_city_tool":
        city = str(args.get("city") or args.get("city_name") or first)
        if ctx.get("live_seller_available"):
            vars_["ast_seller_company"], vars_["ast_seller_city"] = "a verified IndiaMART seller", city
            return f"update_ast_buy_variables_by_city_tool → live seller available near {city}"
        vars_["ast_seller_pns"] = ""
        return "update_ast_buy_variables_by_city_tool → no live seller available for this location"
    if name == "ast_buy_confirmed_tool":
        return "ast_buy_confirmed_tool → call transfer to live seller initiated"
    if name == "end_interaction":
        return "end_interaction → call ended"
    return f"{name} → (not available in simulation)"


def _bot_messages(system: str, turns: list[dict]) -> list[dict]:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": "(call connected)"}]
    for t in turns:
        msgs.append({"role": "assistant" if t["role"] == "BOT" else "user", "content": t["text"] or "…"})
    return msgs


class LocalBot:
    def __init__(self, prompt_text: str, ctx: dict):
        self.prompt, self.ctx = prompt_text, ctx
        self.vars = initial_variables(ctx)
        self.tools = prompt_render.tool_names(prompt_text)
        self.tool_results: list[str] = []
        self.ended = self.transferred = False

    def opener(self) -> dict:
        return {"text": self.vars["initial_message"], "tools": [], "vars": {}}

    def reply(self, turns: list[dict]) -> dict:
        said, tools_used, updates = "", [], {}
        for _ in range(3):  # allow tool-only steps before the bot speaks
            system = prompt_render.render(self.prompt, self.ctx["flow"], self.vars) + RUNTIME.format(
                tools=", ".join(self.tools),
                vars=json.dumps({k: self.vars.get(k, "") for k in TRACKED_VARS}, ensure_ascii=False),
                tool_results=("Last tool results:\n" + "\n".join(self.tool_results)) if self.tool_results else "")
            out = llm.chat_json_messages(_bot_messages(system, turns), BOT_SCHEMA, "sim_bot",
                                         model=config.SIM_BOT_MODEL, reasoning=config.SIM_BOT_REASONING,
                                         max_tokens=4000, temperature=0.3)
            for k, v in (out.get("variable_updates") or {}).items():
                k = k.strip().lstrip("@").strip("`").strip()  # the prompt writes setters as `@name`
                self.vars[k] = updates[k] = str(v)
            self.tool_results = []
            for call in out.get("tool_calls") or []:
                name = call.get("name", "")
                self.tool_results.append(_run_tool(name, call.get("args") or {}, self.vars, self.ctx))
                tools_used.append({"name": name, "args": call.get("args") or {}})
                self.ended |= name in ("ast_buy_confirmed_tool", "end_interaction")
                self.transferred |= name == "ast_buy_confirmed_tool"
            self.ended |= bool(out.get("end_interaction"))
            said = (out.get("say") or "").strip()
            if said or self.ended:
                break
        return {"text": said, "tools": tools_used, "vars": updates}

# ───────────────────────────── bot under test: Samvaad ─────────────────────────────


async def _samvaad_conversation(case: dict, ctx: dict, version) -> tuple[list[dict], str]:
    from pydantic import SecretStr
    from sarvam_conv_ai_sdk import (AsyncSamvaadAgent, InteractionConfig, InteractionType, MsgStatus,
                                    ServerMsgType, ServerTextChunkMsg)
    from sarvam_conv_ai_sdk.messages.types import UserIdentifierType

    inbox: asyncio.Queue = asyncio.Queue()
    ended = asyncio.Event()
    chunks: list[str] = []
    events: list[dict] = []
    last_chunked = {"text": None}

    async def on_text(m):
        if isinstance(m, ServerTextChunkMsg):
            chunks.append(m.text)
            if m.status == MsgStatus.COMPLETED:
                last_chunked["text"] = "".join(chunks)
                await inbox.put(last_chunked["text"])
                chunks.clear()
        elif m.text != last_chunked["text"]:  # skip a full message duplicating streamed chunks
            await inbox.put(m.text)

    async def on_event(e):
        if e.type == ServerMsgType.INTERACTION_END:
            ended.set()
        elif e.type in (ServerMsgType.TOOL_CALL, ServerMsgType.VARIABLE_UPDATE, ServerMsgType.STATE_TRANSITION):
            events.append(json.loads(e.model_dump_json()))

    async def bot_reply(quiet: float = 3.0, first_wait: float = 60.0) -> str:
        texts = []
        while True:
            try:
                texts.append(await asyncio.wait_for(inbox.get(), quiet if texts else first_wait))
            except asyncio.TimeoutError:
                return " ".join(texts).strip()
            if ended.is_set() and inbox.empty():
                return " ".join(texts).strip()

    variables = {k: prompt_render._value(v) for k, v in initial_variables(ctx).items()}
    cfg = InteractionConfig(
        user_identifier_type=UserIdentifierType.CUSTOM, user_identifier=f"qa-sim-{uuid.uuid4().hex[:8]}",
        org_id=config.SAMVAAD_ORG_ID, workspace_id=config.SAMVAAD_WORKSPACE_ID, app_id=config.SAMVAAD_APP_ID,
        version=int(version) if version else None, interaction_type=InteractionType.CHAT, sample_rate=16000,
        agent_variables=variables, initial_bot_message=variables["initial_message"] or None,
    )
    agent = AsyncSamvaadAgent(api_key=SecretStr(config.SARVAM_API_KEY), config=cfg, text_callback=on_text,
                              event_callback=on_event, base_url=config.SAMVAAD_BASE_URL)
    turns: list[dict] = []
    end = "max_turns"
    loop = asyncio.get_running_loop()
    await agent.start()
    try:
        if not await agent.wait_for_connect(timeout=20):
            raise RuntimeError("Samvaad agent did not connect")

        def add(role, text, extra=None):
            turns.append({"idx": len(turns), "role": role, "text": text, **(extra or {})})

        add("BOT", await bot_reply())
        for _ in range(config.SIM_MAX_TURNS):
            b = await loop.run_in_executor(None, buyer_step, case, turns)
            if b["say"]:
                add("BUYER", b["say"])
            if b["hang_up"]:
                end = "buyer_hung_up"
                break
            await agent.send_text(b["say"] or "…")
            text = await bot_reply()
            add("BOT", text, {"tools": list(events)})
            events.clear()
            if ended.is_set():
                end = "bot_ended"
                break
    finally:
        await agent.stop()
    return turns, end

# ───────────────────────────── run + judge ─────────────────────────────


def converse_local(case: dict, ctx: dict, prompt_text: str) -> tuple[list[dict], str]:
    bot = LocalBot(prompt_text, ctx)
    o = bot.opener()
    turns = [{"idx": 0, "role": "BOT", "text": o["text"], "tools": [], "vars": {}}]
    end = "max_turns"
    for _ in range(config.SIM_MAX_TURNS):
        b = buyer_step(case, turns)
        if b["say"]:
            turns.append({"idx": len(turns), "role": "BUYER", "text": b["say"]})
        if b["hang_up"]:
            end = "buyer_hung_up"
            break
        r = bot.reply(turns)
        turns.append({"idx": len(turns), "role": "BOT", "text": r["text"], "tools": r["tools"], "vars": r["vars"]})
        if bot.ended:
            end = "transferred" if bot.transferred else "bot_ended"
            break
    return turns, end



def _timed(turns: list[dict]) -> list[dict]:
    """Give simulated turns pseudo timestamps (5 s apart, no gaps) so the grader works unchanged."""
    return [dict(t, idx=i, start=i * 5.0, end=i * 5.0 + 5.0) for i, t in enumerate(turns)]


SIM_NOTE = ("NOTE: simulated TEXT conversation used to test a prompt change — no speech recognition errors, "
            "no timing (score DEAD_AIR and WER as na).\n\n")


def grade_sim(turns: list[dict]) -> dict:
    turns = _timed(turns)
    r = grader.grade_turns(turns, signals.compute(turns, None), note=SIM_NOTE)
    for c in r["scorecard"]:
        if c["param"] in ("DEAD_AIR", "WER"):
            c["status"] = "na"
    r["grade"] = grader.overall(r["scorecard"])
    return r


def prompt_text(variant: str, version_id: str) -> str:
    if variant == "original":
        return config.BOT_PROMPT_PATH.read_text(encoding="utf-8")
    v = db.one("SELECT path FROM prompt_versions WHERE version_id=?", (version_id,))
    return Path(v["path"]).read_text(encoding="utf-8")


def run_one(run_id: int, version_id: str, case: dict, variant: str, text: str | None = None) -> dict:
    row = {"run_id": run_id, "case_id": case["case_id"], "variant": variant, "created_at": db.now()}
    try:
        ctx = case["context"]
        if config.SIM_BACKEND == "samvaad":
            version = config.SAMVAAD_BASELINE_VERSION if variant == "original" else config.SAMVAAD_CORRECTED_VERSION
            if not version:
                raise RuntimeError(f"no Samvaad app version set for the {variant} prompt")
            turns, end = asyncio.run(_samvaad_conversation(case, ctx, version))
        else:
            turns, end = converse_local(case, ctx, text or prompt_text(variant, version_id))
        g = grade_sim(turns)
        row.update(turns_json=json.dumps(_timed(turns), ensure_ascii=False), end_reason=end, grade=g["grade"],
                   scorecard_json=json.dumps(g["scorecard"], ensure_ascii=False))
    except Exception as e:
        log.exception("simulation failed: %s %s", case["case_id"], variant)
        row["error"] = str(e)[:500]
    db.execute(f"INSERT INTO sim_results ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
               tuple(row.values()))
    return row


def summarize(run_id: int) -> dict:
    rows = db.rows("SELECT case_id, variant, grade, scorecard_json, error FROM sim_results WHERE run_id=?", (run_id,))
    out = {}
    for variant in ("original", "corrected"):
        ok = [r for r in rows if r["variant"] == variant and not r["error"]]
        cards = [db.loads(r["scorecard_json"], []) for r in ok]
        out[variant] = {
            "n": len(ok),
            "fatal": sum(r["grade"] == "FATAL" for r in ok),
            "non_fatal": sum(r["grade"] == "NON_FATAL" for r in ok),
            "pass": sum(r["grade"] == "PASS" for r in ok),
            "param_fails": {k: sum(1 for c in cards for e in c if e["param"] == k and e["status"] in ("fatal", "non_fatal"))
                            for k in rubric.PARAM_KEYS},
        }
    out["errors"] = sum(1 for r in rows if r["error"])
    return out


def run_test(version_id: str, case_ids: list[str], texts: dict | None = None) -> int:
    """Replay cases with two prompts. texts={'original': ..., 'corrected': ...} overrides the defaults
    (original production prompt vs prompt version `version_id`)."""
    with db.connect() as c:
        run_id = c.execute("INSERT INTO test_runs (version_id, started_at, status, n_cases) VALUES (?,?, 'running', ?)",
                           (version_id, db.now(), len(case_ids))).lastrowid
    try:
        cases = [x for x in (load_case(cid) for cid in case_ids) if x]
        if not cases:
            raise RuntimeError("no test cases — generate the test setup first")
        jobs = [(case, v) for case in cases for v in ("original", "corrected")]
        with ThreadPoolExecutor(max_workers=config.SIM_WORKERS) as pool:
            list(pool.map(lambda j: run_one(run_id, version_id, *j, text=(texts or {}).get(j[1])), jobs))
        db.execute("UPDATE test_runs SET finished_at=?, status='done', summary_json=? WHERE run_id=?",
                   (db.now(), json.dumps(summarize(run_id)), run_id))
    except Exception as e:
        log.exception("test run %s failed", run_id)
        db.execute("UPDATE test_runs SET finished_at=?, status='error', summary_json=? WHERE run_id=?",
                   (db.now(), json.dumps({"error": str(e)[:500]}), run_id))
    return run_id


# ───────────────────────────── step 7: manual testing ─────────────────────────────

def new_manual_bot(variant: str, version_id: str | None, ctx: dict) -> "LocalBot":
    """A bot for the Manual test tab: same engine as the automated runs; the user plays the buyer."""
    return LocalBot(prompt_text(variant, version_id or ""), dict(ctx))
