"""Sarvam chat-completions wrapper returning parsed JSON.

Tries Structured Outputs (json_schema) first, then falls back to json_object, then to
plain text with brace extraction — so a model/endpoint that rejects one format still works.
"""
import json
import logging
import re
import time

from sarvamai import SarvamAI

from . import config

log = logging.getLogger(__name__)
_client: SarvamAI | None = None
# Sarvam's strict json_schema mode can degenerate into endless whitespace, so we use json_object
# with the schema written into the system prompt.
_FORMATS = ["json_object", None]
_format_ok: dict[str, str | None] = {}  # remembers which format worked per schema name


def client() -> SarvamAI:
    global _client
    if _client is None:
        if not config.SARVAM_API_KEY:
            raise RuntimeError("SARVAM_API_KEY is not set (copy .env.example to .env)")
        _client = SarvamAI(api_subscription_key=config.SARVAM_API_KEY, timeout=300)
    return _client


def _extract_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except ValueError:
        start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _response_format(kind, name, schema):
    if kind == "json_schema":
        return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": False}}
    if kind == "json_object":
        return {"type": "json_object"}
    return None


def chat_json(system: str, user: str, schema: dict, name: str,
              reasoning: str | None = None, max_tokens: int = 8000, retries: int = 3,
              temperature: float = 0.0):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return chat_json_messages(messages, schema, name, reasoning=reasoning,
                              max_tokens=max_tokens, retries=retries, temperature=temperature)


def chat_json_messages(messages: list[dict], schema: dict, name: str, model: str | None = None,
                       reasoning: str | None = None, max_tokens: int = 8000, retries: int = 3,
                       temperature: float = 0.1):
    """reasoning: 'off' (sends reasoning_effort=null — fast, no thinking tokens), 'low', 'medium'
    (server default), 'high', 'max'. If thinking eats the whole token budget and no answer is
    produced, we retry once with reasoning off instead of burning more retries."""
    kinds = [_format_ok[name]] if name in _format_ok else _FORMATS
    messages = [dict(m) for m in messages]
    sys_i = next((n for n, m in enumerate(messages) if m["role"] == "system"), None)
    schema_note = ("\n\n## Output: one JSON object matching this JSON schema\n"
                   + json.dumps(schema, ensure_ascii=False))
    if sys_i is None:
        messages.insert(0, {"role": "system", "content": schema_note.strip()})
    else:
        messages[sys_i]["content"] += schema_note
    last_err = None
    for kind in kinds:
        for attempt in range(retries):
            try:
                kw = dict(messages=messages, model=model or config.LLM_MODEL,
                          temperature=temperature, max_tokens=max_tokens)
                if reasoning in (None, "off", "none"):
                    kw["reasoning_effort"] = None
                elif reasoning != "medium":  # medium is the server default
                    kw["reasoning_effort"] = reasoning
                rf = _response_format(kind, name, schema)
                if rf:
                    kw["response_format"] = rf
                resp = client().chat.completions(**kw)
                choice = resp.choices[0]
                content = choice.message.content or ""
                if not content.strip() and choice.finish_reason == "length":
                    if reasoning not in (None, "off", "none"):
                        log.warning("%s: reasoning used all %d tokens with no answer; retrying with reasoning off",
                                    name, max_tokens)
                        reasoning = "off"
                        continue
                    raise RuntimeError(f"{name}: output truncated at max_tokens={max_tokens}")
                out = _extract_json(content)
                _format_ok[name] = kind
                return out
            except ValueError as e:  # unparseable output: retry same format
                last_err = e
                log.warning("%s: bad JSON (attempt %d): %s", name, attempt + 1, e)
            except Exception as e:
                last_err = e
                msg = str(e)
                if any(code in msg for code in ("429", "500", "502", "503", "504", "timed out", "Timeout")):
                    time.sleep(min(60, 4 * 2 ** attempt))
                    continue
                if "400" in msg or "422" in msg:  # format not supported → try the next one
                    log.warning("%s: format %s rejected: %s", name, kind, msg[:200])
                    break
                raise
    raise RuntimeError(f"{name}: LLM call failed: {last_err}")


# ───────────────────────── judge: IM LLM gateway (GPT / Sonnet) ─────────────────────────
_judge_http = None
_judge_warned = False
_judge_auth_failed = ""  # set once the gateway rejects the key → Sarvam for the rest of the run
JUDGE_STATUS_FILE = config.DATA_DIR / "judge_status.json"  # last gateway result, shown on the dashboard


def judge_ready() -> bool:
    return bool(config.JUDGE_API_KEY) and not _judge_auth_failed


def _record_status(ok: bool, detail: str):
    try:
        JUDGE_STATUS_FILE.write_text(json.dumps({"ok": ok, "model": config.JUDGE_MODEL, "detail": detail[:200],
                                                 "at": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")
    except OSError:
        pass


def judge_status() -> dict | None:
    try:
        return json.loads(JUDGE_STATUS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def judge_label() -> str:
    if not config.JUDGE_API_KEY:
        return f"{config.LLM_MODEL} (Sarvam fallback — no JUDGE_API_KEY)"
    s = judge_status()
    if _judge_auth_failed or (s and not s["ok"] and s.get("model") == config.JUDGE_MODEL):
        return f"{config.LLM_MODEL} (Sarvam fallback — gateway rejected the key)"
    return config.JUDGE_MODEL


def judge_json(system: str, user: str, schema: dict, name: str, max_tokens: int = 8000, retries: int = 3,
               temperature: float = 0.0, **_ignored):
    """Grading decisions (errors / fatal / non-fatal) on the IM LLM gateway — an OpenAI-compatible
    /chat/completions endpoint. Falls back to Sarvam when no JUDGE_API_KEY is configured."""
    global _judge_http, _judge_warned, _judge_auth_failed
    if not judge_ready():
        if not _judge_warned:
            log.warning("judge unavailable (%s) — grading with Sarvam %s",
                        _judge_auth_failed or "no JUDGE_API_KEY", config.LLM_MODEL)
            _judge_warned = True
        return chat_json(system, user, schema, name, reasoning=config.GRADER_REASONING,
                         max_tokens=max_tokens, temperature=temperature)
    import httpx
    if _judge_http is None:
        _judge_http = httpx.Client(timeout=httpx.Timeout(180, connect=20))
    messages = [
        {"role": "system", "content": system + "\n\n## Output: one JSON object matching this JSON schema\n"
         + json.dumps(schema, ensure_ascii=False) + "\nRespond with valid JSON only. No markdown, no code fences."},
        {"role": "user", "content": user},
    ]
    body = {"model": config.JUDGE_MODEL, "messages": messages}
    openrouter = "openrouter.ai" in config.JUDGE_BASE_URL
    if openrouter:
        # OpenRouter: unified `reasoning` object, plain max_tokens, and it ignores sampling params a model rejects.
        body["max_tokens"] = max_tokens + (8000 if _is_reasoning_model(config.JUDGE_MODEL) else 0)
        if _is_reasoning_model(config.JUDGE_MODEL) and config.JUDGE_REASONING:
            body["reasoning"] = {"effort": config.JUDGE_REASONING, "exclude": True}
        else:
            body["temperature"] = temperature
    elif _is_reasoning_model(config.JUDGE_MODEL):
        # GPT-5 / o-series: no custom temperature, max_completion_tokens (covers reasoning + answer),
        # and a reasoning-effort knob instead.
        body["max_completion_tokens"] = max_tokens + 8000
        if config.JUDGE_REASONING:
            body["reasoning_effort"] = config.JUDGE_REASONING
    else:
        body.update(temperature=temperature, max_tokens=max_tokens)
    if config.JUDGE_MODEL.startswith("openai/"):
        body["response_format"] = {"type": "json_object"}
    last_err = None
    for attempt in range(retries):
        try:
            r = _judge_http.post(f"{config.JUDGE_BASE_URL}/chat/completions", json=body,
                                 headers={"Authorization": f"Bearer {config.JUDGE_API_KEY}",
                                          "X-Title": "VANI Call Quality Audit"})
            if r.status_code == 400:  # drop whichever optional parameter this model/gateway refuses
                err = r.text.lower()
                bad = next((p for p in ("reasoning_effort", "temperature", "response_format") if p in body
                            and p in err), None) or ("response_format" if "response_format" in body else None)
                if bad:
                    log.warning("%s: judge model refused '%s'; retrying without it", name, bad)
                    body.pop(bad)
                    continue
            if r.status_code in (401, 403):
                raise RuntimeError(f"judge gateway rejected the API key ({r.status_code}): {r.text[:200]}")
            if r.status_code >= 400:
                raise RuntimeError(f"judge gateway HTTP {r.status_code}: {r.text[:300]}")
            content = ((r.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            out = _extract_json(content)
            _record_status(True, "ok")
            return out
        except ValueError as e:
            last_err = e
            log.warning("%s: judge returned bad JSON (attempt %d): %s", name, attempt + 1, e)
        except RuntimeError as e:
            last_err = e
            if "rejected the API key" in str(e):
                _judge_auth_failed = str(e)
                _record_status(False, str(e))
                log.error("%s — falling back to Sarvam for grading", e)
                return judge_json(system, user, schema, name, max_tokens=max_tokens, retries=retries,
                                  temperature=temperature)
            time.sleep(min(30, 3 * 2 ** attempt))
        except Exception as e:  # network
            last_err = e
            time.sleep(min(30, 3 * 2 ** attempt))
    raise RuntimeError(f"{name}: judge call failed: {last_err}")


def _is_reasoning_model(model: str) -> bool:
    m = model.lower().split("/")[-1]
    return m.startswith(("gpt-5", "o1", "o3", "o4"))
