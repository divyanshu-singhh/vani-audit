"""VANI Call Quality — modern light dashboard.

    streamlit run dashboard/app.py
"""
import html
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audit import agreement, config, db, improve, ingest, rubric, simulate, transcribe  # noqa: E402
from audit.pipeline import LOCK, current_step, is_running  # noqa: E402

st.set_page_config(page_title="VANI · Call Quality", page_icon="◉", layout="wide")
db.init()
improve.init()

# ─────────────────────────────── design tokens ───────────────────────────────
INK, MUTED, LINE = "#1E2235", "#6B7085", "#E6E8F0"
PRIMARY, PRIMARY_TINT = "#5B5BD6", "#EEF0FF"
FATAL, FATAL_TINT = "#E5484D", "#FFEEEE"
MINOR, MINOR_TINT = "#E59400", "#FFF5E1"
PASS, PASS_TINT = "#2F9E6A", "#E7F7EE"
NA, NA_TINT = "#8B8D98", "#F1F2F5"
FONT = "Plus Jakarta Sans, Inter, system-ui, sans-serif"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="st-"], .stMarkdown, button, input, textarea, select {{ font-family: {FONT} !important; }}
.stApp {{ background: #F6F7FB; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1360px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3, h4 {{ font-family: {FONT} !important; letter-spacing: -0.02em; color: {INK}; }}

/* cards */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div.card-marker),
div[data-testid="stVerticalBlockBorderWrapper"] {{
  border-radius: 18px !important; border: 1px solid {LINE} !important; background: #FFFFFF;
  box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 6px 18px rgba(16,24,40,.04);
}}
/* buttons & capsules */
.stButton > button, .stDownloadButton > button, [data-testid="stPopover"] button {{
  border-radius: 999px !important; font-weight: 600; padding: .35rem 1rem; border: 1px solid {LINE};
}}
.stButton > button[kind="primary"] {{ background: {PRIMARY}; border-color: {PRIMARY}; }}
[data-testid="stButtonGroup"] button {{ border-radius: 999px !important; font-weight: 600; }}
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{ border-radius: 12px !important; }}
[data-testid="stExpander"] details {{ border-radius: 14px; border: 1px solid {LINE}; background: #FFF; }}

/* custom bits */
.brand {{ display:flex; align-items:center; gap:.7rem; }}
.brand .logo {{ width:38px; height:38px; border-radius:12px; background:linear-gradient(135deg,{PRIMARY},#8E7CF0);
  display:flex; align-items:center; justify-content:center; color:#fff; font-weight:800; font-size:18px; }}
.brand .t1 {{ font-size:1.35rem; font-weight:800; color:{INK}; line-height:1.1; }}
.brand .t2 {{ font-size:.82rem; color:{MUTED}; }}
.pill {{ display:inline-flex; align-items:center; gap:.35rem; padding:.18rem .62rem; border-radius:999px;
  font-size:.78rem; font-weight:600; white-space:nowrap; }}
.pill .dot {{ width:7px; height:7px; border-radius:50%; display:inline-block; }}
.p-fatal {{ background:{FATAL_TINT}; color:{FATAL}; }} .p-fatal .dot {{ background:{FATAL}; }}
.p-minor {{ background:{MINOR_TINT}; color:#A56A00; }} .p-minor .dot {{ background:{MINOR}; }}
.p-pass {{ background:{PASS_TINT}; color:{PASS}; }} .p-pass .dot {{ background:{PASS}; }}
.p-na {{ background:{NA_TINT}; color:{MUTED}; }} .p-na .dot {{ background:{NA}; }}
.p-brand {{ background:{PRIMARY_TINT}; color:{PRIMARY}; }} .p-brand .dot {{ background:{PRIMARY}; }}
.chip {{ display:inline-block; padding:.12rem .55rem; margin:.12rem .2rem .12rem 0; border-radius:999px;
  background:#F3F4F8; color:#454A5E; font-size:.74rem; font-weight:500; }}
.kpi {{ background:#fff; border:1px solid {LINE}; border-radius:18px; padding:1rem 1.1rem;
  box-shadow:0 1px 2px rgba(16,24,40,.04), 0 6px 18px rgba(16,24,40,.04); height:100%; }}
.kpi .lbl {{ color:{MUTED}; font-size:.8rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }}
.kpi .val {{ font-size:2rem; font-weight:800; color:{INK}; line-height:1.15; margin-top:.25rem; }}
.kpi .sub {{ font-size:.8rem; color:{MUTED}; margin-top:.15rem; }}
.kpi.accent-fatal {{ border-top:4px solid {FATAL}; }} .kpi.accent-minor {{ border-top:4px solid {MINOR}; }}
.kpi.accent-pass {{ border-top:4px solid {PASS}; }} .kpi.accent-brand {{ border-top:4px solid {PRIMARY}; }}
.muted {{ color:{MUTED}; font-size:.86rem; }}
.title-sm {{ font-weight:700; font-size:1.02rem; color:{INK}; margin-bottom:.2rem; }}
.clamp {{ display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; color:#454A5E;
  font-size:.86rem; min-height:2.5em; }}
.meter {{ height:8px; border-radius:999px; background:#F0F1F6; overflow:hidden; }}
.meter > span {{ display:block; height:100%; border-radius:999px; }}
.issue {{ border-left:4px solid {MINOR}; background:#FFFDF7; border-radius:10px; padding:.55rem .75rem; margin:.35rem 0; }}
.issue.fatal {{ border-left-color:{FATAL}; background:#FFF8F8; }}
.issue .q {{ color:{MUTED}; font-size:.82rem; margin-top:.2rem; }}
.bubble {{ padding:.5rem .8rem; border-radius:14px; margin:.3rem 0; max-width:88%; font-size:.9rem; line-height:1.45; }}
.bubble.bot {{ background:{PRIMARY_TINT}; color:{INK}; border-top-left-radius:4px; }}
.bubble.buyer {{ background:#F3F4F8; color:{INK}; margin-left:auto; border-top-right-radius:4px; }}
.bubble .ts {{ font-size:.7rem; color:{MUTED}; margin-right:.4rem; font-weight:600; }}
.bubble .meta {{ display:block; font-size:.72rem; color:{MUTED}; margin-top:.25rem; }}
.flow {{ display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; }}
.flow .arrow {{ color:{MUTED}; }}
</style>
""", unsafe_allow_html=True)

RESULT = {"FATAL": ("Fatal", "p-fatal"), "NON_FATAL": ("Non-fatal", "p-minor"), "PASS": ("Pass", "p-pass")}
PSTATUS = {"fatal": ("Fatal", "p-fatal"), "non_fatal": ("Non-fatal", "p-minor"), "ok": ("OK", "p-pass"),
           "na": ("N/A", "p-na")}
STEP_STATUS = {"kept": ("Kept as base", "p-pass"), "approved": ("Approved → base", "p-pass"),
               "manual_review": ("Manual review", "p-minor"), "rejected": ("Rejected", "p-na"),
               "error": ("Error", "p-fatal"), "correcting": ("Correcting…", "p-brand"),
               "testing": ("Testing…", "p-brand")}
CALL_STATE = {"new": ("Waiting", "p-na"), "transcribing": ("Transcribing", "p-brand"),
              "transcribed": ("Transcribed", "p-brand"), "error": ("Error", "p-fatal")}


def pill(text: str, cls: str) -> str:
    return f"<span class='pill {cls}'><span class='dot'></span>{html.escape(text)}</span>"


def result_pill(grade: str | None, status: str = "") -> str:
    if grade in RESULT:
        return pill(*RESULT[grade])
    return pill(*CALL_STATE.get(status, (status or "—", "p-na")))


def chips(items: list[str]) -> str:
    return "".join(f"<span class='chip'>{html.escape(i)}</span>" for i in items)


def mmss(sec) -> str:
    sec = float(sec or 0)
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def kpi(col, label, value, sub="", accent="brand"):
    col.markdown(f"<div class='kpi accent-{accent}'><div class='lbl'>{label}</div><div class='val'>{value}</div>"
                 f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


def launch(*args: str):
    subprocess.Popen([sys.executable, "-m", "audit.pipeline", *args], cwd=ROOT,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def bubbles(turns: list[dict], meta: bool = False, height: int | None = 420):
    parts = []
    for t in turns:
        bot = t["role"] == "BOT"
        m = ""
        if meta:
            bits = [f"🔧 {c.get('name')}" for c in t.get("tools") or []] + \
                   [f"{k}={v}" for k, v in (t.get("vars") or {}).items()]
            m = f"<span class='meta'>{html.escape(' · '.join(bits))}</span>" if bits else ""
        ts = f"<span class='ts'>{mmss(t['start'])}</span>" if "start" in t and not meta else ""
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
        st.markdown(f"<div class='issue {'fatal' if c['status'] == 'fatal' else ''}'>"
                    f"{pill(*PSTATUS[c['status']])} <b>{html.escape(rubric.PARAM_NAMES.get(c['param'], c['param']))}</b>"
                    f"<span class='muted'>{at}</span><div>{html.escape(c.get('note') or '')}</div>{q}</div>",
                    unsafe_allow_html=True)
    ok = [rubric.PARAM_NAMES[c["param"]] for c in card if c["status"] == "ok"]
    na = [rubric.PARAM_NAMES[c["param"]] for c in card if c["status"] == "na"]
    if ok:
        st.markdown(f"<div class='muted' style='margin-top:.6rem'>OK</div>{chips(ok)}", unsafe_allow_html=True)
    if na:
        st.markdown(f"<div class='muted' style='margin-top:.4rem'>Not applicable</div>{chips(na)}",
                    unsafe_allow_html=True)


# ─────────────────────────────── data ───────────────────────────────
def load_calls() -> list[dict]:
    rows = db.rows("SELECT c.call_id, c.filename, c.duration_sec, c.status, c.error, g.grade, g.summary,"
                   " g.scorecard_json, h.label AS human FROM calls c LEFT JOIN grades g USING(call_id)"
                   " LEFT JOIN human_labels h USING(call_id) ORDER BY c.added_at DESC")
    for r in rows:
        r["card"] = db.loads(r["scorecard_json"], []) or []
        r["bad"] = [c for c in r["card"] if c["status"] in ("fatal", "non_fatal")]
    return rows


calls = load_calls()
graded = [c for c in calls if c["grade"]]
busy = is_running()
base = improve.current_base()

# ─────────────────────────────── dialogs ───────────────────────────────
@st.dialog("Call details", width="large")
def quick_view(call_id: str):
    c = next(x for x in calls if x["call_id"] == call_id)
    call = db.one("SELECT * FROM calls WHERE call_id=?", (call_id,))
    tr = db.one("SELECT * FROM transcripts WHERE call_id=?", (call_id,))
    g = db.one("SELECT * FROM grades WHERE call_id=?", (call_id,))
    turns = db.loads(tr["turns_json"], []) if tr else []
    st.markdown(f"<div class='title-sm'>{html.escape(c['filename'])}</div>"
                f"{result_pill(c['grade'], c['status'])} <span class='muted'>· {mmss(c['duration_sec'])} · "
                f"{len(turns)} turns</span>", unsafe_allow_html=True)
    if g and g["summary"]:
        st.markdown(f"<p class='muted' style='margin-top:.5rem'>{html.escape(g['summary'])}</p>", unsafe_allow_html=True)
    st.audio(call["audio_path"])
    t1, t2, t3, t4 = st.tabs(["Scorecard", "Transcript", "Buyer vs bot", "Your verdict"])
    with t1:
        if g:
            issue_cards(db.loads(g["scorecard_json"], []) or [], turns)
        else:
            st.info("Not audited yet.")
    with t2:
        bubbles(turns, height=420)
        txt = transcribe.export_copy(call_id) if tr else None
        if txt and txt.exists():
            st.download_button("Download transcript", txt.read_bytes(), file_name=txt.name)
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
        lab = st.segmented_control("Your verdict", opts, format_func=lambda o: RESULT[o][0],
                                   default=h["label"] if h and h["label"] in opts else None, key=f"v_{call_id}")
        notes = st.text_input("Notes", value=h["notes"] if h else "", key=f"n_{call_id}")
        if st.button("Save verdict", type="primary", disabled=not lab):
            agreement.set_label(call_id, lab, notes, "dashboard")
            st.success("Saved")
        if tr and st.button("Bot and buyer swapped? Swap and re-audit"):
            transcribe.swap_speakers(call_id)
            launch("process")
            st.rerun()


@st.dialog("Improvement step", width="large")
def step_view(step_id: int):
    s = db.one("SELECT * FROM improve_steps WHERE step_id=?", (step_id,))
    m = db.loads(s["metrics_json"], {}) or {}
    name = rubric.PARAM_NAMES.get(s["param"], s["param"])
    st.markdown(f"<div class='title-sm'>#{s['rank']} · {html.escape(name)}</div>"
                f"{pill(*STEP_STATUS.get(s['status'], (s['status'], 'p-na')))} "
                f"<span class='muted'>· base {s['base_id']}{' → ' + s['new_base_id'] if s['new_base_id'] else ''}</span>",
                unsafe_allow_html=True)
    st.markdown(f"<p style='margin-top:.6rem'><b>{html.escape(s['summary'] or '')}</b><br>"
                f"<span class='muted'>{html.escape(s['reason'] or '')}</span></p>", unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["Audit output", "Prompt change", "Replayed calls"])
    with t1:
        if not m:
            st.caption("No test results.")
        else:
            rows = [{"Measure": f"'{name}' in replays", "Current base": m["target"]["before"],
                     "With correction": m["target"]["after"]},
                    {"Measure": "Fatal calls", "Current base": m["fatal"]["before"], "With correction": m["fatal"]["after"]}]
            for k, d in m["params"].items():
                if k != s["param"] and (d["before"] or d["after"]):
                    rows.append({"Measure": rubric.PARAM_NAMES[k], "Current base": d["before"], "With correction": d["after"]})
            df = pd.DataFrame(rows)
            df["Change"] = df.apply(lambda r: "better" if r["With correction"] < r["Current base"] else
                                    "worse" if r["With correction"] > r["Current base"] else "same", axis=1)
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.caption(f"{m['usable']} of {m['cases']} calls replayed successfully with both prompts.")
    with t2:
        st.caption(s["rationale"] or "")
        st.code(s["diff"] or "(no change)", language="diff")
        if s["candidate_path"] and Path(s["candidate_path"]).exists():
            st.download_button("Download corrected prompt", Path(s["candidate_path"]).read_bytes(),
                               file_name=Path(s["candidate_path"]).name)
    with t3:
        sims = db.rows("SELECT * FROM sim_results WHERE run_id=? ORDER BY case_id, variant", (s["run_id"],)) if s["run_id"] else []
        by = {}
        for r in sims:
            by.setdefault(r["case_id"], {})[r["variant"]] = r
        if not by:
            st.caption("No replays.")
        for cid, p in by.items():
            title = (db.one("SELECT filename FROM calls WHERE call_id=?", (cid,)) or {}).get("filename", cid)
            o, n = p.get("original"), p.get("corrected")
            with st.expander(f"{title} — base {RESULT.get((o or {}).get('grade'), ('…',))[0]} → "
                             f"corrected {RESULT.get((n or {}).get('grade'), ('…',))[0]}"):
                a, b = st.columns(2)
                for col, r, label in ((a, o, "Current base"), (b, n, "With correction")):
                    with col:
                        st.markdown(f"**{label}**")
                        if not r:
                            continue
                        if r["error"]:
                            st.error(r["error"])
                            continue
                        bad = [c for c in db.loads(r["scorecard_json"], []) if c["status"] in ("fatal", "non_fatal")]
                        st.markdown(" ".join(pill(f"{rubric.PARAM_NAMES[c['param']]}", PSTATUS[c['status']][1])
                                             for c in bad) or pill("All OK", "p-pass"), unsafe_allow_html=True)
                        bubbles(db.loads(r["turns_json"], []), meta=True, height=320)
    if s["status"] == "manual_review":
        st.divider()
        c1, c2, _ = st.columns([1.3, 1, 3])
        if c1.button("Approve → make base", type="primary", disabled=not s["candidate_path"]):
            bid = improve.approve(step_id)
            st.success(f"New base prompt: {bid}")
            st.rerun()
        if c2.button("Reject"):
            improve.reject(step_id)
            st.rerun()


# ─────────────────────────────── header + nav ───────────────────────────────
h1, h2 = st.columns([3, 2], vertical_alignment="center")
with h1:
    st.markdown(f"<div class='brand'><div class='logo'>V</div><div><div class='t1'>VANI Call Quality</div>"
                f"<div class='t2'>Audit · auto-improve · test — IndiaMART buyer voice bot</div></div></div>",
                unsafe_allow_html=True)


@st.fragment(run_every=3)
def live_status():
    running = is_running()
    if running:
        st.markdown(f"<div style='text-align:right'>{pill('Running · ' + (current_step() or 'working'), 'p-brand')}</div>",
                    unsafe_allow_html=True)
        st.session_state["_busy"] = True
    else:
        if LOCK.exists():
            LOCK.unlink(missing_ok=True)
        st.markdown(f"<div style='text-align:right'>{pill('Base prompt · ' + base['base_id'], 'p-brand')} "
                    f"{pill('Idle', 'p-na')}</div>", unsafe_allow_html=True)
        if st.session_state.pop("_busy", False):
            st.rerun()


with h2:
    live_status()

if not config.SARVAM_API_KEY:
    st.error("SARVAM_API_KEY is not set. Add it to `.env` and restart.")

PAGES = ["Overview", "Calls", "Auto-improve", "Test lab", "Criteria"]
if "_goto" in st.session_state:
    st.session_state["nav"] = st.session_state.pop("_goto")
st.session_state.setdefault("nav", "Overview")
page = st.segmented_control("Navigation", PAGES, key="nav", label_visibility="collapsed") or "Overview"
st.write("")

# ─────────────────────────────── OVERVIEW ───────────────────────────────
if page == "Overview":
    n_f = sum(c["grade"] == "FATAL" for c in graded)
    n_m = sum(c["grade"] == "NON_FATAL" for c in graded)
    n_p = sum(c["grade"] == "PASS" for c in graded)
    pending = sum(c["status"] in ("new", "transcribed") for c in calls)
    bases = db.rows("SELECT * FROM base_prompts ORDER BY CAST(SUBSTR(base_id, 7) AS INTEGER)")
    k = st.columns(5)
    kpi(k[0], "Recordings", len(calls), f"{pending} waiting to process" if pending else "all processed", "brand")
    kpi(k[1], "Fatal", n_f, f"{n_f / len(graded):.0%} of audited" if graded else "—", "fatal")
    kpi(k[2], "Non-fatal", n_m, f"{n_m / len(graded):.0%} of audited" if graded else "—", "minor")
    kpi(k[3], "Pass", n_p, f"{n_p / len(graded):.0%} of audited" if graded else "—", "pass")
    kpi(k[4], "Base prompt", base["base_id"].replace("base_", ""), f"{len(bases) - 1} correction(s) kept", "brand")
    st.write("")

    left, right = st.columns([3, 2], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("<div class='title-sm'>Most frequent errors</div><div class='muted'>Click a bar to see "
                        "those calls</div>", unsafe_allow_html=True)
            freq = improve.error_frequency()
            if not freq:
                st.caption("No errors found yet.")
            else:
                f = list(reversed(freq[:10]))
                fat = [e["n_fatal"] for e in f]
                minor = [e["n_failing"] - e["n_fatal"] for e in f]
                names = [e["name"] for e in f]
                fig = go.Figure()
                fig.add_bar(y=names, x=fat, orientation="h", name="Fatal", marker_color=FATAL,
                            hovertemplate="%{y}: %{x} fatal<extra></extra>", customdata=[e["param"] for e in f])
                fig.add_bar(y=names, x=minor, orientation="h", name="Non-fatal", marker_color=MINOR,
                            hovertemplate="%{y}: %{x} non-fatal<extra></extra>", customdata=[e["param"] for e in f])
                fig.update_layout(barmode="stack", height=max(240, 36 * len(f) + 70), bargap=0.4,
                                  margin=dict(l=0, r=8, t=8, b=0), plot_bgcolor="#fff", paper_bgcolor="#fff",
                                  font=dict(family=FONT, color=INK, size=13), xaxis=dict(gridcolor="#F0F1F6",
                                  title="Calls", dtick=1), legend=dict(orientation="h", y=1.1, x=0))
                ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points",
                                     key="freq_chart", config={"displayModeBar": False})
                pts = (ev or {}).get("selection", {}).get("points", []) if ev else []
                if pts:
                    st.session_state["f_param"] = [pts[0].get("customdata")]
                    st.session_state["_goto"] = "Calls"
                    st.rerun()
    with right:
        with st.container(border=True):
            st.markdown("<div class='title-sm'>Next step</div>", unsafe_allow_html=True)
            if pending:
                st.markdown(f"<p class='muted'>{pending} recording(s) are waiting.</p>", unsafe_allow_html=True)
                if st.button(f"Process {pending} recording(s)", type="primary", disabled=busy):
                    launch("process"); st.rerun()
            elif graded and any(e["fixable"] for e in improve.error_frequency()):
                st.markdown("<p class='muted'>Fix the most frequent errors one by one; each correction is kept only "
                            "if the audit improves.</p>", unsafe_allow_html=True)
                if st.button("Run auto-improve (top 3)", type="primary", disabled=busy):
                    launch("improve", "--top", "3"); st.session_state["_goto"] = "Auto-improve"; st.rerun()
            else:
                st.markdown("<p class='muted'>Add recordings in Calls.</p>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<div class='title-sm'>Prompt history</div>", unsafe_allow_html=True)
            flow = " <span class='arrow'>→</span> ".join(
                pill(b["base_id"].replace("base_", ""), "p-brand" if b["base_id"] == base["base_id"] else "p-na")
                for b in bases)
            st.markdown(f"<div class='flow'>{flow}</div>", unsafe_allow_html=True)
            for b in bases[::-1][:4]:
                st.markdown(f"<div class='muted' style='margin-top:.35rem'><b>{b['base_id']}</b> · "
                            f"{html.escape(b['notes'] or '')}</div>", unsafe_allow_html=True)

# ─────────────────────────────── CALLS ───────────────────────────────
elif page == "Calls":
    top = st.columns([3, 1.2, 1.2], vertical_alignment="center")
    top[0].markdown("<div class='title-sm' style='font-size:1.25rem'>Calls</div>"
                    f"<div class='muted'>{len(calls)} recordings · {len(graded)} audited</div>", unsafe_allow_html=True)
    with top[1].popover("＋ Add recordings", use_container_width=True):
        files = st.file_uploader("Audio files", type=sorted(e[1:] for e in ingest.AUDIO_EXTS),
                                 accept_multiple_files=True)
        if st.button("Add files", disabled=not files, type="primary"):
            new = sum(ingest.save_upload(f.name, f.getvalue())[1] for f in files)
            st.success(f"Added {new} new"); st.rerun()
        st.divider()
        folder = st.text_input("Or a folder", placeholder=r"C:\...\call_recordings")
        limit = st.number_input("How many (0 = all)", min_value=0, value=20, step=10)
        if st.button("Add from folder", disabled=not folder):
            p = Path(folder.strip().strip('"'))
            if not p.is_dir():
                st.error("Folder not found")
            else:
                audio = sorted(x for x in p.rglob("*") if x.suffix.lower() in ingest.AUDIO_EXTS)
                audio = audio[:int(limit)] if limit else audio
                new = sum(ingest.register(a)[1] for a in audio)
                st.success(f"Added {new} new"); st.rerun()
    pending = sum(c["status"] in ("new", "transcribed") for c in calls)
    errors = sum(c["status"] == "error" for c in calls)
    with top[2]:
        if st.button(f"▶ Process {pending}" if pending else "Nothing to process", type="primary",
                     disabled=busy or not pending, use_container_width=True):
            launch("process"); st.rerun()
        if errors and st.button(f"↻ Retry {errors} failed", disabled=busy, use_container_width=True):
            launch("retry"); st.rerun()

    f1, f2 = st.columns([1.4, 3], vertical_alignment="bottom")
    with f1:
        res = st.pills("Result", ["All", "Fatal", "Non-fatal", "Pass", "Pending"], default="All", key="f_result")
    with f2:
        params_seen = sorted({c["param"] for x in graded for c in x["bad"]}, key=lambda k: rubric.PARAM_KEYS.index(k))
        sel_params = st.pills("Error", params_seen, selection_mode="multi", format_func=rubric.PARAM_NAMES.get,
                              key="f_param")
    q = st.text_input("Search", placeholder="Search file name or summary…", label_visibility="collapsed")

    want = {"Fatal": "FATAL", "Non-fatal": "NON_FATAL", "Pass": "PASS"}.get(res or "All")
    shown = [c for c in calls
             if (res in (None, "All") or (res == "Pending" and not c["grade"]) or c["grade"] == want)
             and (not sel_params or any(b["param"] in sel_params for b in c["bad"]))
             and (not q or q.lower() in (c["filename"] + " " + (c["summary"] or "")).lower())]
    st.markdown(f"<div class='muted' style='margin:.3rem 0 .6rem'>{len(shown)} call(s)</div>", unsafe_allow_html=True)
    per_page = 12
    page_n = st.session_state.get("calls_page", 1)
    for i in range(0, min(len(shown), per_page * page_n), 3):
        cols = st.columns(3, gap="small")
        for col, c in zip(cols, shown[i:i + 3]):
            with col.container(border=True):
                st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                            f"<div class='title-sm' style='overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
                            f"max-width:65%'>{html.escape(Path(c['filename']).stem)}</div>"
                            f"{result_pill(c['grade'], c['status'])}</div>"
                            f"<div class='muted'>{mmss(c['duration_sec'])}"
                            f"{' · you: ' + RESULT[c['human']][0] if c['human'] in RESULT else ''}</div>"
                            f"<div class='clamp' style='margin:.45rem 0'>{html.escape(c['summary'] or c['error'] or '')}</div>"
                            f"<div style='min-height:1.8rem'>{chips([rubric.PARAM_NAMES[b['param']] for b in c['bad']][:3])}</div>",
                            unsafe_allow_html=True)
                if st.button("Quick view", key=f"qv_{c['call_id']}", use_container_width=True):
                    quick_view(c["call_id"])
    if len(shown) > per_page * page_n and st.button("Show more"):
        st.session_state["calls_page"] = page_n + 1
        st.rerun()

# ─────────────────────────────── AUTO-IMPROVE ───────────────────────────────
elif page == "Auto-improve":
    st.markdown("<div class='title-sm' style='font-size:1.25rem'>Auto-improve</div><div class='muted'>Most frequent "
                "error first. Each correction is tested against the current base prompt and kept only if that error "
                "drops, nothing else gets worse and fatal calls don't rise — otherwise it waits for your review.</div>",
                unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([2, 3], gap="medium")
    freq = improve.error_frequency()
    with left:
        with st.container(border=True):
            st.markdown("<div class='title-sm'>Errors by frequency</div>", unsafe_allow_html=True)
            mx = max([e["n_failing"] for e in freq] or [1])
            rank = 0
            for e in freq:
                rank += e["fixable"]
                tag = pill(f"#{rank}", "p-brand") if e["fixable"] else pill("not prompt-fixable: " + e["why_not"], "p-na")
                share = e["n_failing"] / mx
                st.markdown(f"<div style='margin:.55rem 0'>{tag} <b>{html.escape(e['name'])}</b>"
                            f"<span class='muted'> · {e['n_failing']} call(s), {e['n_fatal']} fatal</span>"
                            f"<div class='meter' style='margin-top:.3rem'><span style='width:{share:.0%};background:"
                            f"{FATAL if e['n_fatal'] else MINOR}'></span></div></div>", unsafe_allow_html=True)
            if not freq:
                st.caption("No errors yet — audit some calls first.")
            st.write("")
            n_fix = sum(e["fixable"] for e in freq)
            top_n = st.pills("Errors to work through", [1, 2, 3, 5], default=min(3, max(1, n_fix)), key="imp_n")
            if st.button(f"Run auto-improve on top {top_n or 1}", type="primary", disabled=busy or not n_fix,
                         use_container_width=True):
                launch("improve", "--top", str(top_n or 1)); st.rerun()
            st.caption(f"Each error replays up to {config.SIM_CASES_PER_FIX} failing calls + "
                       f"{improve.REGRESSION_CALLS} others, with both prompts (~1–2 min per error).")
        with st.container(border=True):
            st.markdown(f"<div class='title-sm'>Current base prompt · {base['base_id']}</div>"
                        f"<div class='muted'>{html.escape(base['notes'] or '')}</div>", unsafe_allow_html=True)
            p = Path(base["path"])
            if p.exists():
                st.download_button("Download base prompt", p.read_bytes(), file_name=p.name, use_container_width=True)
    with right:
        sessions = db.rows("SELECT * FROM improve_sessions ORDER BY session_id DESC")
        if not sessions:
            with st.container(border=True):
                st.caption("No auto-improve run yet.")
        for sess in sessions[:3]:
            steps = db.rows("SELECT * FROM improve_steps WHERE session_id=? ORDER BY rank", (sess["session_id"],))
            st.markdown(f"<div class='muted' style='margin:.2rem 0 .4rem'>Run {sess['session_id']} · "
                        f"{sess['started_at'][:16].replace('T', ' ')} · {sess['start_base']} → "
                        f"{sess['end_base'] or '…'}</div>", unsafe_allow_html=True)
            for s in steps:
                m = db.loads(s["metrics_json"], {}) or {}
                with st.container(border=True):
                    a, b = st.columns([4, 1.2], vertical_alignment="center")
                    with a:
                        st.markdown(f"{pill('#' + str(s['rank']), 'p-na')} "
                                    f"<b>{html.escape(rubric.PARAM_NAMES.get(s['param'], s['param']))}</b> "
                                    f"<span class='muted'>· {s['n_failing']} call(s)</span> "
                                    f"{pill(*STEP_STATUS.get(s['status'], (s['status'], 'p-na')))}",
                                    unsafe_allow_html=True)
                        if m:
                            t, fa = m["target"], m["fatal"]
                            reg = ", ".join(rubric.PARAM_NAMES[k] for k in m.get("regressions", []))
                            st.markdown(f"<div class='muted' style='margin-top:.35rem'>error in replays "
                                        f"<b>{t['before']} → {t['after']}</b> · fatal <b>{fa['before']} → {fa['after']}</b>"
                                        f"{' · worse: ' + html.escape(reg) if reg else ''}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='muted'>{html.escape(s['reason'] or s['summary'] or '')}</div>",
                                    unsafe_allow_html=True)
                    with b:
                        if st.button("Details", key=f"sv_{s['step_id']}", use_container_width=True):
                            step_view(s["step_id"])

# ─────────────────────────────── TEST LAB ───────────────────────────────
elif page == "Test lab":
    mode = st.segmented_control("Mode", ["Chat as buyer", "Test cases"], default="Chat as buyer", key="tl_mode",
                                label_visibility="collapsed")
    bases = db.rows("SELECT * FROM base_prompts ORDER BY CAST(SUBSTR(base_id, 7) AS INTEGER)")
    cands = db.rows("SELECT step_id, rank, param, candidate_path FROM improve_steps WHERE candidate_path IS NOT NULL"
                    " ORDER BY step_id DESC")
    prompts = {f"{b['base_id']}{' (original)' if b['base_id'] == 'base_v0' else ''}": b["path"] for b in bases}
    prompts.update({f"candidate · step {c['step_id']} · {rubric.PARAM_NAMES.get(c['param'], c['param'])}":
                    c["candidate_path"] for c in cands})
    cases = db.rows("SELECT case_id, title, case_json FROM test_cases ORDER BY created_at DESC")
    if mode == "Test cases":
        if st.button("Generate test cases for audited calls", disabled=busy):
            launch("testcases"); st.rerun()
        if not cases:
            st.info("No test cases yet.")
        for i in range(0, len(cases), 2):
            cols = st.columns(2)
            for col, c in zip(cols, cases[i:i + 2]):
                case = db.loads(c["case_json"], {})
                ctx = case.get("context", {})
                with col.container(border=True):
                    st.markdown(f"<div class='title-sm'>{html.escape(Path(c['title']).stem)}</div>"
                                f"<div class='muted'>{html.escape(ctx.get('product_name') or '—')} · "
                                f"{ctx.get('language_name')} · live seller {'yes' if ctx.get('live_seller_available') else 'no'}"
                                f" · {len(case.get('buyer_script', []))} buyer lines</div>", unsafe_allow_html=True)
                    with st.expander("Buyer script & checks"):
                        st.markdown(f"**Bot opens with:** {ctx.get('initial_message', '')}")
                        for n, line in enumerate(case.get("buyer_script", []), 1):
                            st.markdown(f"{n}. {line}")
                        st.markdown("**The bot must:**\n" + "\n".join(f"- {x}" for x in case.get("checks", [])))
    else:
        s1, s2, s3, s4 = st.columns([2, 2, 2, 1], vertical_alignment="bottom")
        src = s1.selectbox("Context", ["custom"] + [c["case_id"] for c in cases],
                           format_func=lambda cid: "Custom product" if cid == "custom" else
                           next(Path(c["title"]).stem for c in cases if c["case_id"] == cid))
        names = list(prompts)
        pa = s2.selectbox("Prompt A", names, index=0)
        pb = s3.selectbox("Prompt B (optional)", ["—"] + names, index=len(names) if len(names) > 1 else 0)
        if src == "custom":
            cc = st.columns([2, 1, 1])
            product = cc[0].text_input("Product", value="Solar Panel")
            lang = cc[1].selectbox("Language", ["hindi", "english", "telugu"])
            live = cc[2].toggle("Live seller available")
            ctx = {"flow": "helpdesk", "product_name": product, "language_name": lang,
                   "live_seller_available": live, "quantity_unit_options": ["Piece"], "specification_options": {},
                   "initial_message": f"Namaste, main Vani bol rahi hoon IndiaMART se. Kya aapko {product} ki requirement hai?"}
        else:
            ctx = db.loads(next(c["case_json"] for c in cases if c["case_id"] == src), {}).get("context", {})
        chosen = [x for x in (pa, pb) if x and x != "—"]
        key = json.dumps([src, chosen, ctx], sort_keys=True)
        mt = st.session_state.get("mt")
        if s4.button("Restart", use_container_width=True) or not mt or mt["key"] != key:
            bots = {n: simulate.LocalBot(Path(prompts[n]).read_text(encoding="utf-8"), dict(ctx)) for n in chosen}
            mt = {"key": key, "bots": bots, "score": {},
                  "turns": {n: [{"idx": 0, "role": "BOT", "text": b.opener()["text"]}] for n, b in bots.items()}}
            st.session_state["mt"] = mt
        cols = st.columns(len(chosen), gap="medium")
        for col, n in zip(cols, chosen):
            with col.container(border=True):
                st.markdown(f"<div class='title-sm'>{html.escape(n)}</div>"
                            f"{pill('call ended', 'p-na') if mt['bots'][n].ended else ''}", unsafe_allow_html=True)
                bubbles(mt["turns"][n], meta=True, height=400)
                if n in mt["score"]:
                    sc = mt["score"][n]
                    st.markdown(result_pill(sc["grade"]) + " " + " ".join(
                        pill(rubric.PARAM_NAMES[c["param"]], PSTATUS[c["status"]][1])
                        for c in sc["scorecard"] if c["status"] in ("fatal", "non_fatal")), unsafe_allow_html=True)
        msg = st.chat_input("Type what the buyer says…")
        if msg:
            live_ = [n for n in chosen if not mt["bots"][n].ended]
            for n in live_:
                mt["turns"][n].append({"idx": len(mt["turns"][n]), "role": "BUYER", "text": msg})

            def reply(n):
                try:
                    r = mt["bots"][n].reply(mt["turns"][n])
                    return n, {"idx": len(mt["turns"][n]), "role": "BOT", "text": r["text"], "tools": r["tools"],
                               "vars": r["vars"]}
                except Exception as e:
                    return n, {"idx": len(mt["turns"][n]), "role": "BOT", "text": f"⚠ {e}"}

            with st.spinner("Bot is replying…"):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    for n, t in pool.map(reply, live_):
                        mt["turns"][n].append(t)
            st.rerun()
        if st.button("Score with the audit sheet", disabled=all(len(mt["turns"][n]) < 3 for n in chosen)):
            with st.spinner("Scoring…"):
                for n in chosen:
                    mt["score"][n] = simulate.grade_sim(mt["turns"][n])
            st.rerun()

# ─────────────────────────────── CRITERIA ───────────────────────────────
elif page == "Criteria":
    a, b = st.columns([3, 2], gap="medium")
    with a:
        with st.container(border=True):
            st.markdown("<div class='title-sm'>Audit sheet</div><div class='muted'>Overall: Fatal if any parameter "
                        "is fatal · Non-fatal if any is non-fatal · otherwise Pass.</div>", unsafe_allow_html=True)
            for key, name, d, f, nf in rubric.PARAMETERS:
                with st.expander(name):
                    st.markdown(f"{d}\n\n{pill('Fatal', 'p-fatal')} {html.escape(f)}<br><br>"
                                f"{pill('Non-fatal', 'p-minor')} {html.escape(nf)}", unsafe_allow_html=True)
    with b:
        with st.container(border=True):
            st.markdown("<div class='title-sm'>Agreement with human auditors</div>", unsafe_allow_html=True)
            m = agreement.metrics()
            if m:
                k = st.columns(3)
                kpi(k[0], "Labelled", m["n"])
                kpi(k[1], "Fatal agree", f"{m['accuracy']:.0%}")
                kpi(k[2], "Cohen's κ", f"{m['kappa']:.2f}")
            else:
                st.caption("No verdicts yet — open a call's Quick view → Your verdict, or import a CSV.")
            up = st.file_uploader("Import labels (CSV: filename, label)", type=["csv"])
            if up and st.button("Import", type="primary"):
                r = agreement.import_csv(up.getvalue())
                st.success(f"Imported {r['imported']}")
