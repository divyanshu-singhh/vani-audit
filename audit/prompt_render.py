"""Render the production prompt the way the Samvaad runtime would for one call.

The prompt file is a markdown export of the Sarvam agent config:
- markdown escapes everywhere (`bot\\_name`, `\\!=`, `\\[\\]`) — including inside Jinja tags
- Jinja `{% if %}` blocks evaluated against agent variables
- placeholders wrapped in U+FEFF: `﻿ bot_name ﻿` (value) and `﻿@ call_outcome﻿`
  (a variable the agent sets)
- four sections: Global Instructions (always), then one state section per flow
"""
import json
import re

import jinja2

SECTION_MARKERS = {
    "global": "Global Instructions",
    "enrichment": "Enrichment",
    "redial_no_name_city": "Redial - No Name City",
    "redial_ast": "Redial_AST",
}
FLOW_SECTIONS = {
    "helpdesk": ["global"],
    "unknown": ["global"],
    "enrichment": ["global", "enrichment"],
    "redial_no_name_city": ["global", "redial_no_name_city"],
    "redial_ast": ["global", "redial_ast"],
}

_MD_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!=<>|~])")
_PLACEHOLDER = re.compile(r"﻿\s*(@?)\s*([A-Za-z0-9_]+)\s*﻿")
_ENV = jinja2.Environment(undefined=jinja2.Undefined, autoescape=False, keep_trailing_newline=True)


def unescape(text: str) -> str:
    return _MD_ESCAPE.sub(r"\1", text)


def split_sections(text: str) -> dict[str, str]:
    lines = unescape(text).splitlines()
    starts = []
    for i, line in enumerate(lines):
        clean = line.strip().rstrip("-").strip()
        for key, marker in SECTION_MARKERS.items():
            if clean == marker:
                starts.append((i, key))
    if not starts:
        return {"global": "\n".join(lines)}
    out = {}
    for n, (i, key) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        out[key] = "\n".join(lines[i + 1:end])
    return out


def _value(v) -> str:
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return "" if v is None else str(v)


def render(prompt_text: str, flow: str, variables: dict) -> str:
    sections = split_sections(prompt_text)
    body = "\n\n".join(sections[k] for k in FLOW_SECTIONS.get(flow, ["global"]) if k in sections)
    ctx = {k: _value(v) for k, v in variables.items()}
    rendered = _ENV.from_string(body).render(**ctx)

    def sub(m):
        setter, name = m.groups()
        return f"`@{name}`" if setter else ctx.get(name, f"<{name}>")

    rendered = _PLACEHOLDER.sub(sub, rendered)
    return re.sub(r"\n{3,}", "\n\n", rendered)


def settable_variables(prompt_text: str) -> list[str]:
    """Variables the prompt tells the agent to update (`@ name`)."""
    return sorted({n for s, n in _PLACEHOLDER.findall(unescape(prompt_text)) if s})


def tool_names(prompt_text: str) -> list[str]:
    names = {n for n in settable_variables(prompt_text) if n.endswith("_tool")}
    return sorted(names | {"end_interaction"})
