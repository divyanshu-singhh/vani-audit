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
