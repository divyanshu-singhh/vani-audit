"""VANI Call Quality Audit — one tab per step, modern light UI.

    streamlit run dashboard/app.py
"""
import html
import json
from datetime import datetime
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audit import agreement, analysis, config, db, fixes, ingest, llm, refine, reset, rubric, simulate, transcribe  # noqa: E402
from audit.pipeline import LOCK, current_step, is_running  # noqa: E402

st.set_page_config(page_title="VANI · Call Quality Audit", page_icon="◉", layout="wide")
db.init()
rubric.reload()   # the audit sheet is a shared file; pick up edits made elsewhere

# ─────────────────────────────── design tokens ───────────────────────────────
INK, MUTED, LINE = "#1E2235", "#6B7085", "#E6E8F0"
PRIMARY, PRIMARY_TINT = "#5B5BD6", "#EEF0FF"
FATAL, MINOR, PASS = "#E5484D", "#E59400", "#2F9E6A"
FONT = "Plus Jakarta Sans, Inter, system-ui, sans-serif"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="st-"], .stMarkdown, button, input, textarea, select {{ font-family: {FONT} !important; }}
[data-testid="stIconMaterial"], span[class*="material-symbols"], .material-icons, [data-testid="stExpanderToggleIcon"] {{ font-family: 'Material Symbols Rounded' !important; }}
.stApp {{ background: #F6F7FB; }}
.block-container {{ padding-top: 1.3rem; padding-bottom: 3rem; max-width: 1360px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3, h4 {{ font-family: {FONT} !important; letter-spacing: -0.02em; color: {INK}; }}

/* capsule tabs */
[data-testid="stTabs"] [role="tablist"] {{ gap: .45rem; flex-wrap: wrap; border-bottom: none !important;
  box-shadow: none !important; padding: .2rem 0 .7rem; }}
[data-testid="stTabs"] [role="tablist"]::after, [data-testid="stTabs"] [role="tablist"]::before {{ display:none !important; }}
[data-testid="stTab"] {{ border-radius: 999px !important; border: 1px solid {LINE} !important; background: #FFFFFF;
  padding: .4rem 1rem !important; height: auto !important; margin: 0 !important; transition: all .15s ease; }}
[data-testid="stTab"]:hover {{ border-color: {PRIMARY} !important; }}
[data-testid="stTab"] p {{ font-weight: 600 !important; font-size: .9rem !important; color: #454A5E; }}
[data-testid="stTab"][aria-selected="true"] {{ background: {PRIMARY} !important; border-color: {PRIMARY} !important;
  box-shadow: 0 4px 14px rgba(91,91,214,.28); }}
[data-testid="stTab"][aria-selected="true"] p {{ color: #FFFFFF !important; }}
[data-testid="stTab"]::after, [data-testid="stTab"]::before {{ display:none !important; }}

/* cards, buttons, inputs */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  border-radius: 18px !important; border: 1px solid {LINE} !important; background: #FFFFFF;
  box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 6px 18px rgba(16,24,40,.04); }}
.stButton > button, .stDownloadButton > button, [data-testid="stPopover"] button {{
  border-radius: 999px !important; font-weight: 600; border: 1px solid {LINE}; }}
.stButton > button[kind="primary"] {{ background: {PRIMARY}; border-color: {PRIMARY}; }}
[data-testid="stButtonGroup"] button {{ border-radius: 999px !important; font-weight: 600; }}
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{ border-radius: 12px !important; }}
[data-testid="stExpander"] details {{ border-radius: 14px; border: 1px solid {LINE}; background: #FFF; }}
[data-testid="stFileUploaderDropzone"] {{ border-radius: 14px; background: #FAFBFF; }}

/* custom elements */
.brand {{ display:flex; align-items:center; gap:.7rem; }}
.brand .logo {{ width:40px; height:40px; border-radius:12px; background:linear-gradient(135deg,{PRIMARY},#8E7CF0);
  display:flex; align-items:center; justify-content:center; color:#fff; font-weight:800; font-size:18px; }}
.brand .t1 {{ font-size:1.4rem; font-weight:800; color:{INK}; line-height:1.1; }}
.brand .t2 {{ font-size:.82rem; color:{MUTED}; }}
.sec-title {{ font-size:1.2rem; font-weight:800; color:{INK}; margin:.2rem 0 .1rem; }}
.sec-hint {{ color:{MUTED}; font-size:.88rem; margin-bottom:.9rem; }}
.title-sm {{ font-weight:700; font-size:1rem; color:{INK}; margin-bottom:.15rem; }}
.muted {{ color:{MUTED}; font-size:.86rem; }}
.pill {{ display:inline-flex; align-items:center; gap:.35rem; padding:.18rem .62rem; border-radius:999px;
  font-size:.78rem; font-weight:600; white-space:nowrap; }}
.pill .dot {{ width:7px; height:7px; border-radius:50%; display:inline-block; }}
.p-fatal {{ background:#FFEEEE; color:{FATAL}; }} .p-fatal .dot {{ background:{FATAL}; }}
.p-minor {{ background:#FFF5E1; color:#A56A00; }} .p-minor .dot {{ background:{MINOR}; }}
.p-pass {{ background:#E7F7EE; color:{PASS}; }} .p-pass .dot {{ background:{PASS}; }}
.p-na {{ background:#F1F2F5; color:{MUTED}; }} .p-na .dot {{ background:#8B8D98; }}
.p-brand {{ background:{PRIMARY_TINT}; color:{PRIMARY}; }} .p-brand .dot {{ background:{PRIMARY}; }}
.p-review {{ background:#F3EAFE; color:#7C3AED; }} .p-review .dot {{ background:#8B5CF6; }}
.review-box {{ border-left:4px solid #8B5CF6; background:#FAF5FF; border-radius:10px; padding:.6rem .8rem; margin:.4rem 0; }}
.batch-head {{ display:flex; align-items:center; gap:.5rem; flex-wrap:wrap; margin:.9rem 0 .4rem; font-weight:700; color:#1E2235; }}
.chip {{ display:inline-block; padding:.12rem .55rem; margin:.12rem .2rem .12rem 0; border-radius:999px;
  background:#F3F4F8; color:#454A5E; font-size:.74rem; font-weight:500; }}
.kpi {{ background:#fff; border:1px solid {LINE}; border-radius:18px; padding:.9rem 1.05rem; height:100%;
  box-shadow:0 1px 2px rgba(16,24,40,.04), 0 6px 18px rgba(16,24,40,.04); }}
.kpi .lbl {{ color:{MUTED}; font-size:.76rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em; }}
.kpi .val {{ font-size:1.9rem; font-weight:800; color:{INK}; line-height:1.15; margin-top:.2rem; }}
.kpi .sub {{ font-size:.8rem; color:{MUTED}; margin-top:.1rem; }}
.kpi.a-fatal {{ border-top:4px solid {FATAL}; }} .kpi.a-minor {{ border-top:4px solid {MINOR}; }}
.kpi.a-pass {{ border-top:4px solid {PASS}; }} .kpi.a-brand {{ border-top:4px solid {PRIMARY}; }}
.clamp {{ display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; color:#454A5E;
  font-size:.85rem; min-height:2.5em; }}
.meter {{ height:8px; border-radius:999px; background:#F0F1F6; overflow:hidden; }}
.meter > span {{ display:block; height:100%; border-radius:999px; }}
.issue {{ border-left:4px solid {MINOR}; background:#FFFDF7; border-radius:10px; padding:.55rem .75rem; margin:.35rem 0; }}
.issue.fatal {{ border-left-color:{FATAL}; background:#FFF8F8; }}
.issue .q {{ color:{MUTED}; font-size:.82rem; margin-top:.2rem; }}
.bubble {{ padding:.5rem .8rem; border-radius:14px; margin:.3rem 0; max-width:88%; font-size:.9rem; line-height:1.45; }}
.bubble.bot {{ background:{PRIMARY_TINT}; color:{INK}; border-top-left-radius:4px; }}
.bubble.buyer {{ background:#F3F4F8; color:{INK}; margin-left:auto; border-top-right-radius:4px; }}
.bubble .ts {{ font-size:.7rem; color:{MUTED}; margin-right:.4rem; font-weight:700; }}
.bubble .meta {{ display:block; font-size:.72rem; color:{MUTED}; margin-top:.25rem; }}
.kv {{ display:grid; grid-template-columns: 9.5rem 1fr; gap:.35rem .8rem; font-size:.88rem; }}
.kv .k {{ color:{MUTED}; }} .kv .v {{ color:{INK}; font-weight:600; }}
</style>
""", unsafe_allow_html=True)

RESULT = {"FATAL": ("Fatal", "p-fatal"), "NON_FATAL": ("Non-fatal", "p-minor"), "PASS": ("Pass", "p-pass")}
PSTATUS = {"fatal": ("Fatal", "p-fatal"), "non_fatal": ("Non-fatal", "p-minor"), "ok": ("OK", "p-pass"),
           "na": ("N/A", "p-na")}
CALL_STATE = {"new": ("Waiting", "p-na"), "transcribing": ("Transcribing", "p-brand"),
              "transcribed": ("Transcribed", "p-brand"), "graded": ("Audited", "p-pass"), "error": ("Error", "p-fatal")}


# ─────────────────────────────── helpers ───────────────────────────────
def pill(text: str, cls: str) -> str:
    return f"<span class='pill {cls}'><span class='dot'></span>{html.escape(str(text))}</span>"


def result_pill(grade, status="") -> str:
    return pill(*RESULT[grade]) if grade in RESULT else pill(*CALL_STATE.get(status, (status or "—", "p-na")))


def chips(items) -> str:
    return "".join(f"<span class='chip'>{html.escape(str(i))}</span>" for i in items)


def mmss(sec) -> str:
    sec = float(sec or 0)
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def section(title: str, hint: str):
    st.markdown(f"<div class='sec-title'>{title}</div><div class='sec-hint'>{hint}</div>", unsafe_allow_html=True)


def kpi(col, label, value, sub="", accent="brand"):
    col.markdown(f"<div class='kpi a-{accent}'><div class='lbl'>{label}</div><div class='val'>{value}</div>"
                 f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


def launch(*args: str):
    """Run a pipeline step in the background; its console output goes to data/logs/background.log."""
    out = open(config.LOG_DIR / "background.log", "a", encoding="utf-8")
    subprocess.Popen([sys.executable, "-m", "audit.pipeline", *args], cwd=ROOT, stdout=out, stderr=out,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def busy() -> bool:
    return is_running()


def run_button(label, *args, key, help=None, primary=True, disabled=False, full=False):
    if st.button(label, key=key, type="primary" if primary else "secondary", help=help,
                 disabled=busy() or disabled, use_container_width=full):
        launch(*args)
        st.toast(f"Started: {label}")
        st.rerun()


def bubbles(turns: list[dict], meta: bool = False, height: int | None = 420, times: bool = True):
    parts = []
    for t in turns:
        bot = t["role"] == "BOT"
        m = ""
        if meta:
            bits = [f"🔧 {c.get('name')}" for c in t.get("tools") or []] + \
                   [f"{k}={v}" for k, v in (t.get("vars") or {}).items()]
            m = f"<span class='meta'>{html.escape(' · '.join(bits))}</span>" if bits else ""
        ts = f"<span class='ts'>{mmss(t['start'])}</span>" if times and "start" in t else ""
        parts.append(f"<div class='bubble {'bot' if bot else 'buyer'}'>{ts}{html.escape(t.get('text') or '…')}{m}</div>")
    box = st.container(height=height) if height else st.container()
    with box:
        st.markdown("".join(parts) or "<span class='muted'>No conversation yet.</span>", unsafe_allow_html=True)


def issue_cards(card: list[dict], turns: list[dict] | None = None):
    bad = [c for c in card if c["status"] in ("fatal", "non_fatal")]
    if not bad:
        st.markdown(pill("All parameters OK", "p-pass"), unsafe_allow_html=True)
    for c in sorted(bad, key=lambda c: c["status"] != "fatal"):
        at = ""
        if turns and isinstance(c.get("turn_index"), int) and 0 <= c["turn_index"] < len(turns):
            at = f" · at {mmss(turns[c['turn_index']]['start'])}"
        q = f"<div class='q'>“{html.escape(c['quote'])}”</div>" if c.get("quote") else ""
        st.markdown(f"<div class='issue {'fatal' if c['status'] == 'fatal' else ''}'>{pill(*PSTATUS[c['status']])} "
                    f"<b>{html.escape(rubric.PARAM_NAMES.get(c['param'], c['param']))}</b><span class='muted'>{at}</span>"
                    f"<div>{html.escape(c.get('note') or '')}</div>{q}</div>", unsafe_allow_html=True)
    ok = [rubric.PARAM_NAMES[c["param"]] for c in card if c["status"] == "ok"]
    na = [rubric.PARAM_NAMES[c["param"]] for c in card if c["status"] == "na"]
    if ok:
        st.markdown(f"<div class='muted' style='margin-top:.6rem'>OK</div>{chips(ok)}", unsafe_allow_html=True)
    if na:
        st.markdown(f"<div class='muted' style='margin-top:.4rem'>Not applicable</div>{chips(na)}",
                    unsafe_allow_html=True)


def load_calls() -> list[dict]:
    """Newest first; each call knows its upload batch and whether a human must review its grade."""
    rows = db.rows("SELECT c.call_id, c.filename, c.duration_sec, c.status, c.error, c.added_at, c.batch_id,"
                   " g.grade, g.summary, g.grade_reason, g.scorecard_json, g.manual_review, g.review_reason,"
                   " h.label AS human FROM calls c LEFT JOIN grades g USING(call_id)"
                   " LEFT JOIN human_labels h USING(call_id) ORDER BY c.added_at DESC, c.filename")
    for r in rows:
        r["card"] = db.loads(r["scorecard_json"], []) or []
        r["bad"] = [c for c in r["card"] if c["status"] in ("fatal", "non_fatal")]
        r["review"] = bool(r["manual_review"]) and r["human"] is None  # still waiting for a human decision
    return rows


def local_time(iso: str | None) -> str:
    if not iso:
        return "—"
    return datetime.fromisoformat(iso).astimezone().strftime("%d %b %Y, %H:%M")


def load_batches() -> list[dict]:
    return db.rows("SELECT b.batch_id, b.created_at, b.source, b.label, COUNT(c.call_id) AS n FROM batches b"
                   " JOIN calls c USING(batch_id) GROUP BY b.batch_id ORDER BY b.created_at DESC")


def batch_title(b: dict) -> str:
    src = {"upload": "uploaded", "folder": f"from folder {b['label']}" if b["label"] else "from folder",
           "earlier": "uploaded"}.get(b["source"], b["source"] or "")
    return f"Batch {b['batch_id']} · {local_time(b['created_at'])} · {b['n']} recording(s) {src}"


def call_label(r) -> str:
    return f"{Path(r['filename']).stem}  ·  {RESULT.get(r['grade'], CALL_STATE.get(r['status'], ('—',)))[0]}"


calls = load_calls()
batches = load_batches()


# ─────────────────────────────── quick views ───────────────────────────────
@st.dialog("Call details", width="large")
def call_view(call_id: str):
    c = next(x for x in calls if x["call_id"] == call_id)
    call = db.one("SELECT * FROM calls WHERE call_id=?", (call_id,))
    tr = db.one("SELECT * FROM transcripts WHERE call_id=?", (call_id,))
    g = db.one("SELECT * FROM grades WHERE call_id=?", (call_id,))
    turns = db.loads(tr["turns_json"], []) if tr else []
    st.markdown(f"<div class='title-sm'>{html.escape(c['filename'])}</div>{result_pill(c['grade'], c['status'])} "
                f"<span class='muted'>· {mmss(c['duration_sec'])} · {len(turns)} turns</span>", unsafe_allow_html=True)
    if g and g["summary"]:
        st.markdown(f"<p class='muted' style='margin-top:.5rem'>{html.escape(g['summary'])}</p>", unsafe_allow_html=True)
    if g and g.get("manual_review"):
        reasons = "".join(f"<li>{html.escape(x)}</li>" for x in (g["review_reason"] or "").splitlines() if x)
        status = (f"Resolved — your verdict: <b>{RESULT[c['human']][0]}</b>" if c["human"] in RESULT
                  else "Open <b>Your verdict</b> below and decide Pass / Non-fatal / Fatal.")
        st.markdown(f"<div class='review-box'>{pill('Needs human review', 'p-review')} The grader could not decide "
                    f"confidently:<ul style='margin:.3rem 0 .2rem 1rem'>{reasons}</ul><span class='muted'>{status}"
                    f"</span></div>", unsafe_allow_html=True)
    st.audio(call["audio_path"])
    t1, t2, t3, t4 = st.tabs(["Scorecard", "Transcript", "Buyer vs bot", "Your verdict"])
    with t1:
        if g:
            issue_cards(db.loads(g["scorecard_json"], []) or [], turns)
            raw = db.loads(g["raw_json"], {}) or {}
            if raw.get("additional_observation"):
                st.info(f"Additional observation: {raw['additional_observation']}")
        else:
            st.caption("Not audited yet.")
    with t2:
        bubbles(turns, height=420)
    with t3:
        req = db.loads(g["requirement_json"], {}) if g else {}
        fields = sorted(set((req.get("stated") or {}) | (req.get("captured") or {})))
        if fields:
            st.dataframe(pd.DataFrame({"Field": fields,
                                       "Buyer said": [str((req.get("stated") or {}).get(f) or "") for f in fields],
                                       "Bot captured": [str((req.get("captured") or {}).get(f) or "") for f in fields]}),
                         hide_index=True, use_container_width=True)
        else:
            st.caption("Nothing captured yet.")
    with t4:
        h = db.one("SELECT * FROM human_labels WHERE call_id=?", (call_id,))
        opts = ["PASS", "NON_FATAL", "FATAL"]
        lab = st.segmented_control("Overall", opts, format_func=lambda o: RESULT[o][0],
                                   default=h["label"] if h and h["label"] in opts else None, key=f"lab_{call_id}")
        notes = st.text_input("Notes", value=h["notes"] if h else "", key=f"notes_{call_id}")

        if st.button("Save verdict", type="primary", disabled=not lab, key=f"save_{call_id}"):
            agreement.set_label(call_id, lab, notes, "dashboard")
            st.success("Saved")

        with st.expander("Found a problem that isn't on the audit sheet? Add it as a new row"):
            with st.form(f"newrow_{call_id}", clear_on_submit=True):
                nm = st.text_input("Name", placeholder="e.g. Buyer language not honoured")
                dfn = st.text_area("What it checks", placeholder="The question an auditor asks for this row.")
                f1, f2 = st.columns(2)
                fz = f1.text_area("Fatal when", placeholder="Leave empty if it can never be fatal")
                nz = f2.text_area("Non-fatal when", placeholder="Leave empty if it can never be non-fatal")
                if st.form_submit_button("Add to the sheet", type="primary") and nm.strip() and dfn.strip():
                    rubric.add_parameter(nm, dfn, fz, nz, "dashboard")
                    st.success(f"Added “{nm.strip()}” to the audit sheet. It applies from the next audit.")


@st.dialog("Prompt change", width="large")
def fix_view(fix_id: str):
    f = db.one("SELECT f.*, r.name, r.rank FROM fixes f JOIN root_causes r USING(rc_id) WHERE fix_id=?", (fix_id,))
    st.markdown(f"<div class='title-sm'>#{f['rank']} · {html.escape(f['name'])}</div>"
                f"<p>{html.escape(f['summary'] or '')}</p><p class='muted'>{html.escape(f['rationale'] or '')}</p>",
                unsafe_allow_html=True)
    extra = db.loads(f["edits_json"], {}) or {}
    for d in extra.get("diagnosis", [])[:3]:
        st.markdown(f"<div class='muted'>Line {d.get('line_start')}: {html.escape(d.get('why', ''))}</div>",
                    unsafe_allow_html=True)
    st.code(f["diff"] or "(no applicable change)", language="diff")


@st.dialog("Test case", width="large")
def case_view(case_id: str):
    r = db.one("SELECT t.title, t.case_json, g.grade FROM test_cases t LEFT JOIN grades g ON g.call_id=t.case_id"
               " WHERE t.case_id=?", (case_id,))
    case = db.loads(r["case_json"], {})
    ctx = case.get("context", {})
    st.markdown(f"<div class='title-sm'>{html.escape(r['title'])}</div>recording was {result_pill(r['grade'])}",
                unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Bot starts with**")
        st.info(ctx.get("initial_message") or "—")
        specs = "; ".join(f"{k}: {', '.join(v)}" for k, v in (ctx.get("specification_options") or {}).items())
        st.markdown(f"<div class='kv'><div class='k'>Flow</div><div class='v'>{ctx.get('flow')}</div>"
                    f"<div class='k'>Product</div><div class='v'>{html.escape(ctx.get('product_name') or '—')}</div>"
                    f"<div class='k'>Language</div><div class='v'>{ctx.get('language_name')}</div>"
                    f"<div class='k'>Live seller</div><div class='v'>{'Yes' if ctx.get('live_seller_available') else 'No'}</div>"
                    f"<div class='k'>Quantity units</div><div class='v'>{html.escape(', '.join(ctx.get('quantity_unit_options') or []))}</div>"
                    f"<div class='k'>Specifications</div><div class='v'>{html.escape(specs or '—')}</div></div>",
                    unsafe_allow_html=True)
        st.markdown("**The bot must**")
        for chk in case.get("checks", []):
            st.markdown(f"- {chk}")
    with b:
        st.markdown("**Simulated buyer says (from the recording)**")
        for n, line in enumerate(case.get("buyer_script", []), 1):
            st.markdown(f"<div class='bubble buyer' style='max-width:100%'><span class='ts'>{n}</span>"
                        f"{html.escape(line)}</div>", unsafe_allow_html=True)
        st.markdown("**Buyer behaviour**")
        st.caption(case.get("buyer_behaviour") or "—")
    st.download_button("Download test case (.json)", json.dumps(case, ensure_ascii=False, indent=1),
                       file_name=f"{Path(r['title']).stem}_testcase.json")


# ─────────────────────────────── header ───────────────────────────────
h1, h2 = st.columns([3, 2], vertical_alignment="center")
h1.markdown("<div class='brand'><div class='logo'>V</div><div><div class='t1'>VANI Call Quality Audit</div>"
            "<div class='t2'>Upload → transcripts → issues → corrections → corrected prompt → test setup → "
            "manual & automated tests → before vs after</div></div></div>", unsafe_allow_html=True)


@st.fragment(run_every=3)
def status_bar():
    if busy():
        st.markdown(f"<div style='text-align:right'>{pill('Running · ' + (current_step() or 'working'), 'p-brand')}</div>",
                    unsafe_allow_html=True)
        st.session_state["_was_busy"] = True
    else:
        if LOCK.exists():
            LOCK.unlink(missing_ok=True)
        st.markdown(f"<div style='text-align:right'>{pill('Idle', 'p-na')}</div>", unsafe_allow_html=True)
        if st.session_state.pop("_was_busy", False):
            st.rerun()


@st.dialog("Clear history")
def clear_dialog():
    n = reset.counts()
    st.markdown(f"This removes **all recordings and everything derived from them**, so you can start fresh:"
                f"<ul style='margin:.4rem 0 .6rem 1rem'><li>{n['calls']} recording(s) in {n['batches']} batch(es), "
                f"their transcripts and audits</li><li>{n['prompt_versions']} corrected prompt version(s), test cases and "
                f"{n['test_runs']} test run(s)</li><li>{n['human_labels']} human verdict(s)</li></ul>"
                f"Kept: the original prompt, your <code>.env</code> keys and settings.", unsafe_allow_html=True)
    keep = st.checkbox("Keep a backup copy in data/archive/ (recommended)", value=True)
    typed = st.text_input("Type CLEAR to confirm")
    if busy():
        st.warning("A background step is running — wait for it to finish first.")
    if st.button("Clear history", type="primary", disabled=typed.strip() != "CLEAR" or busy(),
                 use_container_width=True):
        r = reset.clear_history(backup=keep)
        for k_ in list(st.session_state.keys()):
            del st.session_state[k_]
        st.session_state["_cleared"] = r["backup"] or "no backup"
        st.rerun()


with h2:
    status_bar()
    _, cbtn = st.columns([3, 1.3])
    if cbtn.button("🗑 Clear history", use_container_width=True, help="Remove all recordings and results"):
        clear_dialog()
if st.session_state.get("_cleared"):
    b = st.session_state.pop("_cleared")
    st.success("History cleared — start by uploading recordings in ① Upload."
               + (f" Backup saved in {b}." if b != "no backup" else ""))

if not config.SARVAM_API_KEY:
    st.error("SARVAM_API_KEY is not set. Copy `.env.example` to `.env` and add your Sarvam key, then restart.")

tabs = st.tabs(["① Upload", "② Transcripts", "③ Issues", "④ Corrections", "⑤ Corrected prompt",
                "⑥ Test setup", "⑦ Manual test", "⑧ Results", "Audit criteria"])

# ─────────────────────────────── ① UPLOAD ───────────────────────────────
with tabs[0]:
    section("Upload call recordings", "Add one file or many, then Process: each call is transcribed (a copy of the "
            "transcript is saved) and audited against the audit sheet.")
    pending = sum(c["status"] in ("new", "transcribed") for c in calls)
    errors = sum(c["status"] == "error" for c in calls)
    k = st.columns(4)
    kpi(k[0], "Recordings", len(calls), "added so far")
    kpi(k[1], "Waiting", pending, "to process", "minor" if pending else "brand")
    kpi(k[2], "Audited", sum(c["status"] == "graded" for c in calls), "done", "pass")
    kpi(k[3], "Errors", errors, "can be retried", "fatal" if errors else "brand")
    st.write("")
    c1, c2 = st.columns(2, gap="medium")
    with c1.container(border=True):
        st.markdown("<div class='title-sm'>Upload files</div>", unsafe_allow_html=True)
        files = st.file_uploader("Audio files", type=sorted(e[1:] for e in ingest.AUDIO_EXTS),
                                 accept_multiple_files=True, label_visibility="collapsed")
        if st.button("Add files", disabled=not files, type="primary"):
            batch = ingest.new_batch("upload", f"{len(files)} file(s)")
            new = sum(ingest.save_upload(f.name, f.getvalue(), batch)[1] for f in files)
            ingest.drop_if_empty(batch)
            st.success(f"Added {new} new recording(s); {len(files) - new} were already added.")
            if new and not busy():
                launch("process")  # start speech-to-text + audit right away
            st.rerun()
    with c2.container(border=True):
        st.markdown("<div class='title-sm'>Add from a folder</div>", unsafe_allow_html=True)
        folder = st.text_input("Folder", placeholder=r"C:\Users\IndiaMart\Downloads\call_recordings\call_recordings",
                               label_visibility="collapsed")
        fc = st.columns([1, 1], vertical_alignment="bottom")
        limit = fc[0].number_input("How many (0 = all)", min_value=0, value=0, step=10)
        if fc[1].button("Add folder", disabled=not folder, use_container_width=True):
            p = Path(folder.strip().strip('"'))
            if not p.is_dir():
                st.error("Folder not found.")
            else:
                audio = sorted(x for x in p.rglob("*") if x.suffix.lower() in ingest.AUDIO_EXTS)
                audio = audio[:int(limit)] if limit else audio
                bar = st.progress(0.0, text="Adding…")
                new = 0
                batch = ingest.new_batch("folder", p.name)
                for n, a in enumerate(audio, 1):
                    new += ingest.register(a, batch)[1]
                    bar.progress(n / len(audio), text=f"Adding {n}/{len(audio)}")
                ingest.drop_if_empty(batch)
                st.success(f"Added {new} new recording(s) from {len(audio)} file(s).")
                if new and not busy():
                    launch("process")  # start speech-to-text + audit right away
                st.rerun()

    st.write("")
    bc = st.columns([1.4, 1.2, 4])
    with bc[0]:
        run_button(f"▶ Process {pending} recording(s)", "process", key="process", disabled=not pending, full=True,
                   help="Speech-to-text (Sarvam saaras:v4) + issue detection")
    with bc[1]:
        if errors:
            run_button(f"↻ Retry {errors} failed", "retry", key="retry", primary=False, full=True)
    if calls:
        flt = st.pills("Show", ["All", "Waiting", "Audited", "Needs review", "Error"], default="All", key="up_filter")
        want = {"Waiting": ("new", "transcribing", "transcribed"), "Audited": ("graded",), "Error": ("error",)}
        st.markdown("<div class='muted'>Recordings grouped by upload batch, newest first.</div>", unsafe_allow_html=True)
        for n_b, b in enumerate(batches):
            rows = [c for c in calls if c["batch_id"] == b["batch_id"] and (
                flt in (None, "All") or (flt == "Needs review" and c["review"]) or c["status"] in want.get(flt, ()))]
            if not rows:
                continue
            in_batch = [c for c in calls if c["batch_id"] == b["batch_id"]]
            done = sum(c["status"] == "graded" for c in in_batch)
            waiting = sum(c["status"] in ("new", "transcribing", "transcribed") for c in in_batch)
            summary = (f"{done}/{len(in_batch)} audited" + (f" · {waiting} waiting" if waiting else "")
                       + (f" · {sum(c['review'] for c in in_batch)} need review" if any(c["review"] for c in in_batch) else ""))
            with st.expander(f"{batch_title(b)}  —  {summary}", expanded=n_b == 0):
                st.dataframe(pd.DataFrame({
                    "File": [r["filename"] for r in rows], "Added": [local_time(r["added_at"]) for r in rows],
                    "Length": [mmss(r["duration_sec"]) for r in rows],
                    "Status": [CALL_STATE.get(r["status"], (r["status"],))[0] for r in rows],
                    "Result": [RESULT.get(r["grade"], ("",))[0] + (" · needs review" if r["review"] else "")
                               for r in rows],
                    "Note": [r["error"] or "" for r in rows]}), hide_index=True, use_container_width=True)
    else:
        st.info("No recordings yet.")

# ─────────────────────────────── ② TRANSCRIPTS ───────────────────────────────
with tabs[1]:
    section("Transcripts", "Speech-to-text of each call with bot and buyer separated. A copy is saved in "
            "data/transcripts_text/ and reused for test setup and simulation.")
    tr_calls = [c for c in calls if c["status"] in ("transcribed", "graded")]
    if not tr_calls:
        st.info("No transcripts yet — process recordings in ① Upload.")
    else:
        pick = st.selectbox("Call", tr_calls, format_func=call_label, key="tr_pick")
        cid = pick["call_id"]
        call = db.one("SELECT * FROM calls WHERE call_id=?", (cid,))
        tr = db.one("SELECT * FROM transcripts WHERE call_id=?", (cid,))
        turns = db.loads(tr["turns_json"], [])
        left, right = st.columns([2.2, 1], gap="medium")
        with left.container(border=True):
            st.audio(call["audio_path"])
            bubbles(turns, height=560)
        with right.container(border=True):
            st.markdown(f"<div class='title-sm'>{html.escape(Path(call['filename']).stem)}</div>"
                        f"{result_pill(pick['grade'], pick['status'])}", unsafe_allow_html=True)
            st.markdown(f"<div class='kv' style='margin:.8rem 0'><div class='k'>Length</div><div class='v'>"
                        f"{mmss(call['duration_sec'])}</div><div class='k'>Turns</div><div class='v'>{len(turns)} "
                        f"({sum(t['role'] == 'BOT' for t in turns)} bot)</div><div class='k'>Language</div>"
                        f"<div class='v'>{tr['language'] or '—'}</div></div>", unsafe_allow_html=True)
            txt = transcribe.export_copy(cid)
            if txt and txt.exists():
                st.download_button("Transcript (.txt)", txt.read_bytes(), file_name=txt.name, use_container_width=True)
                js = txt.with_suffix(".json")
                st.download_button("Transcript (.json)", js.read_bytes(), file_name=js.name, use_container_width=True)
            st.markdown("<div class='muted' style='margin-top:.8rem'>Bot and buyer the wrong way round?</div>",
                        unsafe_allow_html=True)
            if st.button("Swap and re-audit", use_container_width=True):
                transcribe.swap_speakers(cid)
                launch("process")
                st.rerun()

# ─────────────────────────────── ③ ISSUES ───────────────────────────────
with tabs[2]:
    section("Issues found", "Each call is scored on every audit-sheet parameter. Overall: Fatal if any parameter is "
            "fatal, Non-fatal if any is non-fatal, otherwise Pass. Click Quick view for the full scorecard.")
    graded = [c for c in calls if c["grade"]]
    if not graded:
        st.info("No audited calls yet.")
    else:
        n_f = sum(c["grade"] == "FATAL" for c in graded)
        n_m = sum(c["grade"] == "NON_FATAL" for c in graded)
        n_p = sum(c["grade"] == "PASS" for c in graded)
        n_r = sum(c["review"] for c in graded)
        k = st.columns(5)
        kpi(k[0], "Audited calls", len(graded), "scored on the audit sheet")
        kpi(k[1], "Fatal", n_f, f"{n_f / len(graded):.0%} of audited", "fatal")
        kpi(k[2], "Non-fatal", n_m, f"{n_m / len(graded):.0%} of audited", "minor")
        kpi(k[3], "Pass", n_p, f"{n_p / len(graded):.0%} of audited", "pass")
        k[4].markdown(f"<div class='kpi' style='border-top:4px solid #8B5CF6'><div class='lbl'>Needs review</div>"
                      f"<div class='val'>{n_r}</div><div class='sub'>grader unsure — your call</div></div>",
                      unsafe_allow_html=True)
        st.write("")

        counts = {key: {"fatal": 0, "non_fatal": 0} for key in rubric.PARAM_NAMES}
        for c in graded:
            for b in c["bad"]:
                counts.setdefault(b["param"], {"fatal": 0, "non_fatal": 0})[b["status"]] += 1
        freq = [(k_, v) for k_, v in counts.items() if v["fatal"] or v["non_fatal"]]
        with st.container(border=True):
            st.markdown("<div class='title-sm'>Which parameters fail most</div>", unsafe_allow_html=True)
            if not freq:
                st.markdown(pill("No parameter failed in any call", "p-pass"), unsafe_allow_html=True)
            else:
                freq.sort(key=lambda x: x[1]["fatal"] + x[1]["non_fatal"])
                names = [rubric.PARAM_NAMES[k_] for k_, _ in freq]
                fig = go.Figure()
                fig.add_bar(y=names, x=[v["fatal"] for _, v in freq], orientation="h", name="Fatal", marker_color=FATAL,
                            hovertemplate="%{y}: %{x} fatal<extra></extra>")
                fig.add_bar(y=names, x=[v["non_fatal"] for _, v in freq], orientation="h", name="Non-fatal",
                            marker_color=MINOR, hovertemplate="%{y}: %{x} non-fatal<extra></extra>")
                fig.update_layout(barmode="stack", height=max(200, 36 * len(freq) + 70), bargap=0.4,
                                  margin=dict(l=0, r=8, t=8, b=0), plot_bgcolor="#fff", paper_bgcolor="#fff",
                                  font=dict(family=FONT, color=INK, size=13),
                                  xaxis=dict(gridcolor="#F0F1F6", title="Calls", dtick=1),
                                  legend=dict(orientation="h", y=1.12, x=0))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        graded_batches = [b for b in batches if any(c["batch_id"] == b["batch_id"] for c in graded)]
        bsel = st.pills("Batch", ["All batches"] + [b["batch_id"] for b in graded_batches], default="All batches",
                        key="iss_batch", format_func=lambda x: x if x == "All batches" else next(
                            f"Batch {b['batch_id']} · {local_time(b['created_at'])}" for b in graded_batches
                            if b["batch_id"] == x))
        f1, f2 = st.columns([1.7, 3], vertical_alignment="bottom")
        res = f1.pills("Result", ["All", "Fatal", "Non-fatal", "Pass", "Needs review"], default="All", key="iss_res")
        order = list(rubric.PARAM_NAMES)
        seen = sorted({b["param"] for c in graded for b in c["bad"]},
                      key=lambda p: order.index(p) if p in order else len(order))
        sel = f2.pills("Error", seen, selection_mode="multi", format_func=rubric.PARAM_NAMES.get, key="iss_err")
        want = {"Fatal": "FATAL", "Non-fatal": "NON_FATAL", "Pass": "PASS"}.get(res or "All")
        shown = [c for c in graded if (bsel in (None, "All batches") or c["batch_id"] == bsel)
                 and (res != "Needs review" or c["review"]) and (want is None or c["grade"] == want)
                 and (not sel or any(b["param"] in sel for b in c["bad"]))]
        st.markdown(f"<div class='muted' style='margin:.2rem 0 .2rem'>{len(shown)} call(s), newest first</div>",
                    unsafe_allow_html=True)
        for b in batches:
            group = [c for c in shown if c["batch_id"] == b["batch_id"]]
            if not group:
                continue
            st.markdown(f"<div class='batch-head'>{pill('Batch ' + str(b['batch_id']), 'p-brand')} "
                        f"<span class='muted'>{local_time(b['created_at'])} · {len(group)} call(s)</span></div>",
                        unsafe_allow_html=True)
            for i in range(0, len(group), 3):
                cols = st.columns(3, gap="small")
                for col, c in zip(cols, group[i:i + 3]):
                    with col.container(border=True):
                        review = pill("Needs review", "p-review") if c["review"] else ""
                        st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center;gap:.5rem'>"
                                    f"<div class='title-sm' style='overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>"
                                    f"{html.escape(Path(c['filename']).stem)}</div><div>{review} {result_pill(c['grade'])}</div></div>"
                                    f"<div class='muted'>{mmss(c['duration_sec'])}"
                                f"{' · your verdict: ' + RESULT[c['human']][0] if c['human'] in RESULT else ''}</div>"
                                f"<div class='clamp' style='margin:.45rem 0'>{html.escape(c['summary'] or '')}</div>"
                                f"<div style='min-height:1.8rem'>"
                                f"{chips([rubric.PARAM_NAMES[b['param']] for b in c['bad']][:3])}</div>",
                                unsafe_allow_html=True)
                        if st.button("Quick view", key=f"qv_{c['call_id']}", use_container_width=True):
                            call_view(c["call_id"])

# ─────────────────────────────── ④ CORRECTIONS ───────────────────────────────
with tabs[3]:
    section("Prompt corrections", "Failing calls are grouped into root causes; for each top cause a concrete prompt "
            "change is suggested. Switch on the ones to include in the corrected prompt.")
    n_issue_calls = sum(c["grade"] in ("FATAL", "NON_FATAL") for c in calls)
    run_button("✨ Suggest corrections", "suggest", key="suggest", disabled=not n_issue_calls,
               help="Group issues into root causes, then propose prompt changes for the top 5")
    run_id = analysis.latest_run_id()
    if run_id is None:
        st.info("No suggestions yet. Audit some calls first, then click Suggest corrections.")
    else:
        rcs = db.rows("SELECT * FROM root_causes WHERE run_id=? ORDER BY rank", (run_id,))
        fx = db.rows("SELECT f.*, r.name, r.rank FROM fixes f JOIN root_causes r USING(rc_id) WHERE r.run_id=?"
                     " ORDER BY r.rank", (run_id,))
        left, right = st.columns([2, 3], gap="medium")
        with left.container(border=True):
            st.markdown("<div class='title-sm'>Root causes</div><div class='muted'>Most impactful first</div>",
                        unsafe_allow_html=True)
            mx = max([r["priority"] for r in rcs] or [1]) or 1
            for r in rcs:
                st.markdown(f"<div style='margin:.6rem 0'>{pill('#' + str(r['rank']), 'p-brand')} "
                            f"<b>{html.escape(r['name'])}</b><div class='muted'>{r['n_calls']} call(s) · "
                            f"{r['n_fatal']} fatal · fix in: {r['layer']}</div><div class='meter' style='margin-top:.3rem'>"
                            f"<span style='width:{r['priority'] / mx:.0%};background:{FATAL if r['n_fatal'] else MINOR}'>"
                            f"</span></div></div>", unsafe_allow_html=True)
            with st.expander("How priority is computed"):
                st.code(analysis.PRIORITY_FORMULA, language=None)
        with right:
            st.markdown(f"<div class='title-sm'>Suggested corrections ({len(fx)})</div>", unsafe_allow_html=True)
            for f in fx:
                with st.container(border=True):
                    a, b = st.columns([5, 1.6], vertical_alignment="center")
                    a.markdown(f"{pill('#' + str(f['rank']), 'p-na')} <b>{html.escape(f['name'])}</b>"
                               f"<div style='margin-top:.3rem'>{html.escape(f['summary'] or '')}</div>",
                               unsafe_allow_html=True)
                    current = bool(f["selected"] if f["selected"] is not None else 1) and f["status"] != "rejected"
                    chosen = b.toggle("Include", value=current, key=f"sel{f['fix_id']}")
                    if chosen != current:
                        db.execute("UPDATE fixes SET selected=?, status=? WHERE fix_id=?",
                                   (int(chosen), "proposed" if chosen else "rejected", f["fix_id"]))
                    if b.button("View change", key=f"fv{f['fix_id']}", use_container_width=True):
                        fix_view(f["fix_id"])
            no_fix = [r for r in rcs if r["rc_id"] not in {f["rc_id"] for f in fx}]
            if no_fix:
                with st.container(border=True):
                    pick = st.selectbox("Suggest a correction for another root cause", no_fix,
                                        format_func=lambda r: f"#{r['rank']} · {r['name']}", key="rc_more")
                    run_button("Suggest for this root cause", "fixes", "--rc", pick["rc_id"], key="one_fix",
                               primary=False)

# ─────────────────────────────── ⑤ CORRECTED PROMPT ───────────────────────────────
with tabs[4]:
    section("Corrected prompt", "A copy of the production prompt with the selected corrections applied. The original "
            "is never changed.")
    sel_fixes = db.rows("SELECT f.fix_id, r.name FROM fixes f JOIN root_causes r USING(rc_id)"
                        " WHERE f.status != 'rejected' AND COALESCE(f.selected,1)=1"
                        " AND r.run_id = (SELECT MAX(run_id) FROM analysis_runs) ORDER BY r.rank")
    with st.container(border=True):
        a, b = st.columns([4, 1.4], vertical_alignment="center")
        a.markdown(f"<div class='title-sm'>Selected corrections · {len(sel_fixes)}</div>"
                   f"{chips([x['name'] for x in sel_fixes]) or '<span class=muted>None selected in ④</span>'}",
                   unsafe_allow_html=True)
        if b.button("Create corrected prompt", type="primary", disabled=not sel_fixes or busy(), use_container_width=True):
            vid = fixes.build_version()
            st.toast(f"Created {vid}")
            st.rerun()
    versions = db.rows("SELECT * FROM prompt_versions ORDER BY created_at DESC")
    if not versions:
        st.info("No corrected prompt yet.")
    else:
        vids = [v["version_id"] for v in versions]
        vid = st.pills("Version", vids, default=vids[0], key="ver_pick") or vids[0]
        v = next(x for x in versions if x["version_id"] == vid)
        p = Path(v["path"])
        left, right = st.columns([1, 3], gap="medium")
        with left.container(border=True):
            st.markdown(f"<div class='title-sm'>{v['version_id']}</div><div class='muted'>"
                        f"{v['created_at'][:16].replace('T', ' ')}</div>", unsafe_allow_html=True)
            kpi(st, "Corrections", len(db.loads(v["fix_ids"], [])), "applied")
            st.write("")
            if p.exists():
                st.download_button("Corrected prompt (.md)", p.read_bytes(), file_name=p.name, use_container_width=True)
            st.download_button("Original prompt (.md)", config.BOT_PROMPT_PATH.read_bytes(),
                               file_name=config.BOT_PROMPT_PATH.name, use_container_width=True)
            if v["notes"]:
                st.caption(v["notes"])
        with right.container(border=True):
            st.markdown("<div class='title-sm'>What changed</div>", unsafe_allow_html=True)
            st.code(v["diff"] or "(no changes)", language="diff")

# ─────────────────────────────── ⑥ TEST SETUP ───────────────────────────────
with tabs[5]:
    section("Test setup", "Each recording becomes a replayable test: what the bot knew at the start, the buyer's own "
            "lines from the recording (the simulated buyer says these), and what the bot must get right.")
    n_missing = (db.one("SELECT COUNT(*) n FROM grades WHERE call_id NOT IN (SELECT case_id FROM test_cases)")
                 or {"n": 0})["n"]
    run_button(f"🧪 Generate test setup for {n_missing} call(s)", "testcases", key="tc", disabled=not n_missing)
    cases = db.rows("SELECT t.case_id, t.title, t.case_json, g.grade FROM test_cases t"
                    " LEFT JOIN grades g ON g.call_id=t.case_id ORDER BY t.created_at")
    if not cases:
        st.info("No test cases yet.")
    else:
        st.write("")
        for i in range(0, len(cases), 3):
            cols = st.columns(3, gap="small")
            for col, c in zip(cols, cases[i:i + 3]):
                case = db.loads(c["case_json"], {})
                ctx = case.get("context", {})
                with col.container(border=True):
                    st.markdown(f"<div style='display:flex;justify-content:space-between;gap:.5rem'>"
                                f"<div class='title-sm' style='overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>"
                                f"{html.escape(Path(c['title']).stem)}</div>{result_pill(c['grade'])}</div>"
                                f"<div class='muted'>{html.escape(ctx.get('product_name') or '—')} · "
                                f"{ctx.get('language_name')} · live seller "
                                f"{'yes' if ctx.get('live_seller_available') else 'no'}</div>"
                                f"<div style='margin:.4rem 0'>{chips([str(len(case.get('buyer_script', []))) + ' buyer lines', str(len(case.get('checks', []))) + ' checks'])}</div>",
                                unsafe_allow_html=True)
                    if st.button("Quick view", key=f"cv_{c['case_id']}", use_container_width=True):
                        case_view(c["case_id"])

        st.write("")
        with st.container(border=True):
            st.markdown("<div class='title-sm'>Run the automated test</div><div class='muted'>Each chosen case is replayed "
                        "with the corrected prompt and compared with its recording; results appear in ⑧ Results.</div>",
                        unsafe_allow_html=True)
            versions = db.rows("SELECT version_id FROM prompt_versions ORDER BY created_at DESC")
            if not versions:
                st.warning("Create a corrected prompt in ⑤ first.")
            else:
                c1, c2, c3 = st.columns([1, 3, 1.1], vertical_alignment="bottom")
                ver = c1.selectbox("Corrected prompt", [x["version_id"] for x in versions], key="test_ver")
                failed = [r["case_id"] for r in cases if r["grade"] in ("FATAL", "NON_FATAL")]
                chosen = c2.multiselect("Cases", [r["case_id"] for r in cases], default=failed or None,
                                        format_func=lambda cid: Path(next(r["title"] for r in cases
                                                                          if r["case_id"] == cid)).stem,
                                        key="test_cases")
                r1, r2 = st.columns([2.2, 1.6], vertical_alignment="center")
                auto = r1.toggle("Auto-improve: if the corrected prompt is not better than the recordings, revise it "
                                 "and test again", value=True, key="test_refine")
                rounds = r2.pills("Max rounds", [2, 3, 4, 5], default=config.REFINE_MAX_ROUNDS
                                  if config.REFINE_MAX_ROUNDS in (2, 3, 4, 5) else 3, key="test_rounds",
                                  disabled=not auto)
                extra = ["--rounds", str(rounds or 3)] if auto else ["--no-refine"]
                with c3:
                    run_button("▶ Run test", "test", "--version", ver, "--cases", ",".join(chosen) or "none", *extra,
                               key="run_test", disabled=not chosen, full=True)
                st.caption(f"About {max(1, len(chosen) // 2)}–{max(1, len(chosen))} min per round · bot engine "
                           f"{config.SIM_BACKEND} ({config.SIM_BOT_MODEL}) · up to {config.SIM_MAX_TURNS} bot turns · "
                           f"each revision is saved as a new prompt version")

# ─────────────────────────────── ⑦ MANUAL TEST ───────────────────────────────
with tabs[6]:
    section("Manual test", "You play the buyer. The bot follows the prompt as in production (tools are simulated). "
            "Compare the original and the corrected prompt side by side.")
    versions = [x["version_id"] for x in db.rows("SELECT version_id FROM prompt_versions ORDER BY created_at DESC")]
    cases = db.rows("SELECT case_id, title, case_json FROM test_cases ORDER BY created_at")
    with st.container(border=True):
        s1, s2, s3 = st.columns([2, 2, 1], vertical_alignment="bottom")
        src = s1.selectbox("Start from", ["custom"] + [c["case_id"] for c in cases], key="mt_src",
                           format_func=lambda cid: "Custom product" if cid == "custom" else
                           "Test case: " + Path(next(c["title"] for c in cases if c["case_id"] == cid)).stem)
        modes = ["Original prompt"] + ([f"Corrected ({v})" for v in versions] +
                                       [f"Compare original vs {v}" for v in versions] if versions else [])
        mode = s2.selectbox("Bot", modes, index=len(modes) - 1 if versions else 0, key="mt_mode")
        restart = s3.button("↺ New conversation", use_container_width=True)
        if src == "custom":
            cc = st.columns([2, 1, 1], vertical_alignment="bottom")
            product = cc[0].text_input("Product", value="Solar Panel", key="mt_prod")
            lang = cc[1].selectbox("Language", ["hindi", "english", "telugu"], key="mt_lang")
            live = cc[2].toggle("Live seller available", key="mt_live")
            ctx = {"flow": "helpdesk", "product_name": product, "language_name": lang, "live_seller_available": live,
                   "quantity_unit_options": ["Piece"], "specification_options": {},
                   "initial_message": f"Namaste, main Vani bol rahi hoon IndiaMART se. Kya aapko {product} ki requirement hai?"}
        else:
            ctx = db.loads(next(c["case_json"] for c in cases if c["case_id"] == src), {}).get("context", {})
    if mode.startswith("Compare"):
        ver, variants = mode.split(" vs ")[1], ["original", "corrected"]
    elif mode.startswith("Corrected"):
        ver, variants = mode[len("Corrected ("):-1], ["corrected"]
    else:
        ver, variants = None, ["original"]

    key = (src, mode, json.dumps(ctx, sort_keys=True))
    mt = st.session_state.get("mt")
    if restart or not mt or mt["key"] != key:
        bots = {v: simulate.new_manual_bot(v, ver, ctx) for v in variants}
        mt = {"key": key, "bots": bots, "score": {},
              "turns": {v: [{"idx": 0, "role": "BOT", "text": b.opener()["text"]}] for v, b in bots.items()}}
        st.session_state["mt"] = mt

    cols = st.columns(len(variants), gap="medium")
    for col, v in zip(cols, variants):
        with col.container(border=True):
            label = "Original prompt" if v == "original" else f"Corrected prompt · {ver}"
            st.markdown(f"<div class='title-sm'>{label}</div>"
                        f"{pill('call ended', 'p-na') if mt['bots'][v].ended else ''}", unsafe_allow_html=True)
            bubbles(mt["turns"][v], meta=True, height=420, times=False)
            if v in mt["score"]:
                sc = mt["score"][v]
                st.markdown(result_pill(sc["grade"]) + " " + " ".join(
                    pill(rubric.PARAM_NAMES[c["param"]], PSTATUS[c["status"]][1])
                    for c in sc["scorecard"] if c["status"] in ("fatal", "non_fatal")), unsafe_allow_html=True)

    msg = st.chat_input("Type what the buyer says…", key="mt_input")
    if msg:
        live_variants = [v for v in variants if not mt["bots"][v].ended]
        for v in live_variants:
            mt["turns"][v].append({"idx": len(mt["turns"][v]), "role": "BUYER", "text": msg})

        def reply(v):
            try:
                r = mt["bots"][v].reply(mt["turns"][v])
                return v, {"idx": len(mt["turns"][v]), "role": "BOT", "text": r["text"], "tools": r["tools"],
                           "vars": r["vars"]}
            except Exception as e:
                return v, {"idx": len(mt["turns"][v]), "role": "BOT", "text": f"⚠ error: {e}"}

        with st.spinner("Bot is replying…"):
            with ThreadPoolExecutor(max_workers=2) as pool:
                for v, t in pool.map(reply, live_variants):
                    mt["turns"][v].append(t)
        st.rerun()
    if st.button("Score these conversations with the audit sheet",
                 disabled=all(len(mt["turns"][v]) < 3 for v in variants)):
        with st.spinner("Scoring…"):
            for v in variants:
                mt["score"][v] = simulate.grade_sim(mt["turns"][v])
        st.rerun()

# ─────────────────────────────── ⑧ RESULTS ───────────────────────────────
with tabs[7]:
    section("Recording vs corrected prompt", "Each test case is replayed with the corrected prompt and scored with the "
            "same audit sheet, then compared with the audit of its real recording.")
    runs = db.rows("SELECT * FROM test_runs ORDER BY run_id DESC")
    if not runs:
        st.info("No test runs yet — run one from ⑥ Test setup.")
    else:
        VERDICT = {"better": ("Better than recordings", "p-pass"), "same": ("No improvement", "p-minor"),
                   "worse": ("Worse than recordings", "p-fatal"), "error": ("No result", "p-na")}
        run = st.selectbox("Test run", runs, key="run_pick",
                           format_func=lambda r: f"Run {r['run_id']} · prompt {r['version_id']} · round {r['round'] or 1}"
                                                 f" · {VERDICT.get(r['verdict'], (r['status'],))[0]} · "
                                                 f"{local_time(r['started_at'])}")

        @st.fragment(run_every=5 if run["status"] == "running" else None)
        def results_view():
            r = db.one("SELECT * FROM test_runs WHERE run_id=?", (run["run_id"],))
            sims = db.rows("SELECT * FROM sim_results WHERE run_id=? AND variant='corrected' ORDER BY case_id",
                           (r["run_id"],))
            if r["status"] == "running":
                st.progress(len(sims) / max(1, r["n_cases"] or 1),
                            text=f"Running… {len(sims)} of {r['n_cases'] or 0} test case(s) done")
            if r["status"] == "error":
                st.error((db.loads(r["summary_json"], {}) or {}).get("error", "failed"))
            steps = refine.chain(r["run_id"])
            if len(steps) > 1 or r.get("verdict"):
                flow = " <span class='muted'>→</span> ".join(
                    f"{pill(x['version_id'] + ' · ' + VERDICT.get(x['verdict'], ('running…', 'p-brand'))[0], VERDICT.get(x['verdict'], ('', 'p-brand'))[1])}"
                    + (" ◀" if x["run_id"] == r["run_id"] else "") for x in steps)
                last = steps[-1]
                cmp = db.loads(r.get("compare_json"), {}) or {}
                outcome = ("Kept: this prompt version does better than the recordings."
                           if last.get("verdict") == "better" else
                           "Still running…" if last["status"] == "running" else
                           "Not better after all rounds — review the last version manually or adjust corrections in ④.")
                with st.container(border=True):
                    st.markdown(f"<div class='title-sm'>Improvement rounds</div><div style='display:flex;gap:.4rem;"
                                f"flex-wrap:wrap;align-items:center;margin:.4rem 0'>{flow}</div>"
                                f"<div class='muted'>Score (lower is better: Fatal 10 · Non-fatal 3 · +1 per failing "
                                f"parameter) — this round: recordings <b>{cmp.get('recording_score', '—')}</b> vs corrected "
                                f"prompt <b>{cmp.get('corrected_score', '—')}</b>. {outcome}</div>", unsafe_allow_html=True)
                    v = db.one("SELECT notes FROM prompt_versions WHERE version_id=?", (r["version_id"],)) or {}
                    if (v.get("notes") or "").startswith("Refined"):
                        st.caption(v["notes"])
            case_ids = sorted({s["case_id"] for s in sims})
            real = {c: db.one("SELECT grade, scorecard_json FROM grades WHERE call_id=?", (c,)) or {} for c in case_ids}
            sim = {s["case_id"]: s for s in sims}

            def corrected(c):
                s = sim.get(c)
                return s if s and not s["error"] else None

            k = st.columns(2, gap="medium")
            for col, label, sub, get in (
                    (k[0], "Recording", "real call, as audited", lambda c: real[c].get("grade")),
                    (k[1], "Corrected prompt", "simulated replay", lambda c: (corrected(c) or {}).get("grade"))):
                gs = [get(c) for c in case_ids]
                f_, m_, p_ = gs.count("FATAL"), gs.count("NON_FATAL"), gs.count("PASS")
                col.markdown(f"<div class='kpi a-{'fatal' if f_ else 'minor' if m_ else 'pass'}'><div class='lbl'>{label}"
                             f"</div><div class='sub'>{sub}</div><div style='display:flex;gap:.5rem;flex-wrap:wrap;"
                             f"margin-top:.6rem'>{pill(f'{f_} Fatal', 'p-fatal')}{pill(f'{m_} Non-fatal', 'p-minor')}"
                             f"{pill(f'{p_} Pass', 'p-pass')}</div></div>", unsafe_allow_html=True)
            st.write("")

            def fails(card_json, key):
                return any(c["param"] == key and c["status"] in ("fatal", "non_fatal")
                           for c in db.loads(card_json, []) or [])

            rows = []
            for key in rubric.PARAM_KEYS:
                if key in ("DEAD_AIR", "WER"):  # audio-only parameters: not measurable in a text replay
                    continue
                rec = sum(fails(real[c].get("scorecard_json"), key) for c in case_ids)
                cor = sum(fails((corrected(c) or {}).get("scorecard_json"), key) for c in case_ids)
                if rec or cor:
                    rows.append({"Parameter": rubric.PARAM_NAMES[key], "Recording": rec, "Corrected prompt": cor,
                                 "Change": "✅ better" if cor < rec else "❌ worse" if cor > rec else "= same"})
            with st.container(border=True):
                st.markdown("<div class='title-sm'>Failing calls per parameter</div><div class='muted'>fatal + "
                            "non-fatal · dead air and WER are audio-only, so not compared</div>", unsafe_allow_html=True)
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                else:
                    st.markdown(pill("No parameter failed", "p-pass"), unsafe_allow_html=True)

            st.markdown("<div class='title-sm' style='margin-top:.8rem'>Each case</div>", unsafe_allow_html=True)
            for c in case_ids:
                s = sim.get(c) or {}
                title = (db.one("SELECT title FROM test_cases WHERE case_id=?", (c,)) or {}).get("title", c)
                with st.container(border=True):
                    st.markdown(f"<b>{html.escape(Path(title).stem)}</b>&nbsp;&nbsp; recording "
                                f"{result_pill(real[c].get('grade'))} → corrected prompt "
                                f"{result_pill(s.get('grade') if not s.get('error') else None, 'error' if s.get('error') else '')}",
                                unsafe_allow_html=True)
                    with st.expander("Compare"):
                        rc = db.loads(real[c].get("scorecard_json"), []) or []
                        cc = db.loads(s.get("scorecard_json"), []) or []
                        diff = []
                        for key in rubric.PARAM_KEYS:
                            if key in ("DEAD_AIR", "WER"):
                                continue
                            sr = next((x["status"] for x in rc if x["param"] == key), "na")
                            sc = next((x["status"] for x in cc if x["param"] == key), "na")
                            if sr in ("fatal", "non_fatal") or sc in ("fatal", "non_fatal"):
                                diff.append({"Parameter": rubric.PARAM_NAMES[key], "Recording": PSTATUS[sr][0],
                                             "Corrected prompt": PSTATUS[sc][0]})
                        if diff:
                            st.dataframe(pd.DataFrame(diff), hide_index=True, use_container_width=True)
                        a, b = st.columns(2, gap="medium")
                        with a:
                            st.markdown("**Recording**")
                            tr = db.one("SELECT turns_json FROM transcripts WHERE call_id=?", (c,))
                            bubbles(db.loads(tr["turns_json"], []) if tr else [], height=380)
                        with b:
                            st.markdown("**Corrected prompt (simulated)**")
                            if s.get("error"):
                                st.error(s["error"])
                            bubbles(db.loads(s.get("turns_json"), []) or [], meta=True, height=380, times=False)

        results_view()

# ─────────────────────────────── AUDIT CRITERIA ───────────────────────────────
with tabs[8]:
    section("Audit criteria", "IndiaMART's call-audit sheet (plus two VANI prompt rules). Overall: Fatal if any "
            f"parameter is fatal · Non-fatal if any is non-fatal · otherwise Pass. Looping = more than "
            f"{rubric.LOOP_LIMIT} asks of one field · Dead air = more than {rubric.DEAD_AIR_SEC:.0f} s of bot silence.")
    st.markdown(f"{pill('Errors decided by · ' + llm.judge_label(), 'p-brand')} "
                f"{pill('Speech-to-text · Sarvam ' + config.STT_MODEL, 'p-na')} "
                f"{pill('Simulation · Sarvam ' + config.SIM_BOT_MODEL, 'p-na')}", unsafe_allow_html=True)
    st.write("")
    a, b = st.columns([3, 2], gap="medium")
    with a.container(border=True):
        st.markdown("<div class='title-sm'>Audit sheet</div>"
                    "<div class='muted'>A living knowledge base: edit any row or add a new one. Saved to "
                    f"<code>{rubric.SHEET_FILE.relative_to(config.ROOT).as_posix()}</code> — commit that file to "
                    "share it. Changes apply to the next audit (use “Re-audit” to apply them to existing calls)."
                    "</div>", unsafe_allow_html=True)
        for r in rubric.load_sheet():
            key, name = r["key"], r["name"]
            tag = ("" if r.get("builtin") else "  ·  added by you") + ("" if r.get("active", True) else "  ·  OFF")
            with st.expander(name + tag):
                nd = st.text_area("What it checks", r["definition"], key=f"sd_{key}")
                c1, c2 = st.columns(2)
                nf_ = c1.text_area("Fatal when  (— = not applicable)", r["fatal"], key=f"sf_{key}")
                nn = c2.text_area("Non-fatal when  (— = not applicable)", r["non_fatal"], key=f"sn_{key}")
                b1, b2, b3 = st.columns([1.3, 1.3, 1])
                if b1.button("Save changes", key=f"ss_{key}", type="primary",
                             disabled=(nd, nf_, nn) == (r["definition"], r["fatal"], r["non_fatal"])):
                    rubric.update_parameter(key, "dashboard", definition=nd, fatal=nf_, non_fatal=nn)
                    st.rerun()
                if key not in rubric.PROTECTED:   # on_change: acts only on a real click, never on page re-runs
                    b2.toggle("Use in audits", bool(r.get("active", True)), key=f"sa_{key}",
                              on_change=lambda k=key: rubric.set_active(k, st.session_state[f"sa_{k}"], "dashboard"))
                if not r.get("builtin") and b3.button("Delete row", key=f"sx_{key}"):
                    rubric.delete_parameter(key)
                    st.rerun()
                if r.get("history"):
                    h = r["history"][-1]
                    st.caption(f"Last change: {h['change']} · {h['by'] or 'unknown'} · {h['at'][:16].replace('T', ' ')}")
        with st.expander("➕ Add a new row to the sheet"):
            with st.form("sheet_add", clear_on_submit=True):
                nm = st.text_input("Name", placeholder="e.g. Buyer language not honoured")
                df_ = st.text_area("What it checks", placeholder="The question an auditor asks for this row.")
                c1, c2 = st.columns(2)
                nf_ = c1.text_area("Fatal when", placeholder="Leave empty if it can never be fatal")
                nn_ = c2.text_area("Non-fatal when", placeholder="Leave empty if it can never be non-fatal")
                if st.form_submit_button("Add row", type="primary") and nm.strip() and df_.strip():
                    rubric.add_parameter(nm, df_, nf_, nn_, "dashboard")
                    st.rerun()
        run_button("Re-audit all calls with the updated sheet", "regrade", key="sheet_regrade",
                   help="Grades every call again so sheet changes apply to existing recordings (uses Sarvam credits).")
    with b.container(border=True):
        st.markdown("<div class='title-sm'>Agreement with human auditors</div>", unsafe_allow_html=True)
        m = agreement.metrics()
        if not m:
            st.caption("No labelled calls yet — open Quick view on a call in ③ Issues → Your verdict, or import a CSV.")
        else:
            x = st.columns(2)
            kpi(x[0], "Labelled calls", m["n"])
            kpi(x[1], "Cohen's κ", f"{m['kappa']:.2f}")
            x = st.columns(2)
            kpi(x[0], "Fatal vs not", f"{m['accuracy']:.0%}", "agreement")
            kpi(x[1], "Exact match", f"{m['exact_match']:.0%}", "3 levels")
            if m["disagreements"]:
                st.write("")
                st.dataframe(pd.DataFrame([{
                    "Call": next((Path(r["filename"]).stem for r in calls if r["call_id"] == d["call_id"]), d["call_id"]),
                    "Human": RESULT.get(d["human"], (d["human"],))[0], "Auto": RESULT.get(d["auto"], (d["auto"],))[0]}
                    for d in m["disagreements"]]), hide_index=True, use_container_width=True)
        st.write("")
        up = st.file_uploader("Import labels CSV (filename or call_id, label = Pass / Non-fatal / Fatal)",
                              type=["csv"], key="lab_csv")
        if up and st.button("Import labels", type="primary"):
            r = agreement.import_csv(up.getvalue())
            st.success(f"Imported {r['imported']} label(s)")
            if r["unmatched"]:
                st.warning("Not found: " + ", ".join(r["unmatched"][:10]))
