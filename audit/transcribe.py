"""Speech-to-text via Sarvam's batch API (the only Sarvam STT path with diarization).

Recordings are mono 8 kHz telephony, so bot vs buyer comes from diarization
(num_speakers=2) followed by a lexical bot-detection pass over each speaker.
"""
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sarvamai import SarvamAI

from . import config, db

log = logging.getLogger(__name__)

# Phrases VANI says far more than any buyer does (the prompt's own scripted lines).
BOT_LEXICON = [
    "indiamart", "india mart", "इंडियामार्ट", "इंडिया मार्ट", "seller", "सेलर", "सेलर्स",
    "requirement", "रिक्वायरमेंट", "रिक्वायरमेंट्स", "transfer", "ट्रांसफर", "whatsapp",
    "व्हाट्सएप", "व्हाट्सऐप", "dhanyavaad", "धन्यवाद", "connect", "कनेक्ट", "sahi hai",
    "सही है", "aapka naam", "आपका नाम", "kis city", "किस सिटी", "state", "स्टेट",
    "best seller", "बेस्ट सेलर", "help desk", "हेल्प डेस्क", "thank you for calling",
    "busy", "बिज़ी", "बिजी", "details", "डिटेल्स",
]


def _client() -> SarvamAI:
    if not config.SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set (copy .env.example to .env)")
    return SarvamAI(api_subscription_key=config.SARVAM_API_KEY)


def _entries(data: dict) -> list[dict]:
    """Normalise Sarvam output into [{speaker_id, start, end, text}]."""
    dia = (data.get("diarized_transcript") or {}).get("entries") or []
    if dia:
        return [{"speaker_id": str(e.get("speaker_id", "0")),
                 "start": float(e.get("start_time_seconds") or 0),
                 "end": float(e.get("end_time_seconds") or 0),
                 "text": (e.get("transcript") or "").strip()} for e in dia]
    ts = data.get("timestamps") or {}
    chunks = ts.get("chunks") or ts.get("words") or []
    starts, ends = ts.get("start_time_seconds") or [], ts.get("end_time_seconds") or []
    if chunks and len(chunks) == len(starts) == len(ends):
        return [{"speaker_id": "0", "start": float(s), "end": float(e), "text": c.strip()}
                for c, s, e in zip(chunks, starts, ends)]
    text = (data.get("transcript") or "").strip()
    return [{"speaker_id": "0", "start": 0.0, "end": 0.0, "text": text}] if text else []


def _merge(entries: list[dict], max_gap: float = 1.0) -> list[dict]:
    """Merge consecutive same-speaker fragments into turns."""
    turns: list[dict] = []
    for e in entries:
        if not e["text"]:
            continue
        if turns and turns[-1]["speaker_id"] == e["speaker_id"] and e["start"] - turns[-1]["end"] <= max_gap:
            turns[-1]["text"] += " " + e["text"]
            turns[-1]["end"] = max(turns[-1]["end"], e["end"])
        else:
            turns.append(dict(e))
    return turns


def bot_score(texts: list[str], spoke_first: bool) -> float:
    blob = " ".join(texts).lower()
    hits = sum(len(re.findall(re.escape(w), blob)) for w in BOT_LEXICON)
    return hits + (2.0 if spoke_first else 0.0) + 0.002 * len(blob)


def assign_roles(turns: list[dict], bot_speaker: str | None = None) -> tuple[list[dict], str | None]:
    """Mark each turn BOT or BUYER. Pass bot_speaker to force a mapping (manual swap)."""
    speakers = list(dict.fromkeys(t["speaker_id"] for t in turns))
    if bot_speaker is None and speakers:
        scores = {s: bot_score([t["text"] for t in turns if t["speaker_id"] == s], s == speakers[0])
                  for s in speakers}
        bot_speaker = max(scores, key=scores.get)
    for i, t in enumerate(turns):
        t["idx"] = i
        t["role"] = "BOT" if t["speaker_id"] == bot_speaker else "BUYER"
    return turns, bot_speaker


def store_transcript(call_id: str, data: dict, raw_path: str | None):
    turns, bot = assign_roles(_merge(_entries(data)))
    with db.connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO transcripts (call_id, language, bot_speaker, turns_json, raw_path)"
            " VALUES (?,?,?,?,?)",
            (call_id, data.get("language_code"), bot, json.dumps(turns, ensure_ascii=False), raw_path),
        )
    db.set_status(call_id, "transcribed", transcribed_at=db.now())
    export_copy(call_id)


def export_copy(call_id: str) -> Path | None:
    """Save a readable copy of the transcript (txt + json) for reuse in testing/simulation."""
    c = db.one("SELECT filename FROM calls WHERE call_id=?", (call_id,))
    t = db.one("SELECT language, turns_json FROM transcripts WHERE call_id=?", (call_id,))
    if not c or not t:
        return None
    turns = db.loads(t["turns_json"], [])
    stem = Path(c["filename"]).stem
    lines = [f"# {c['filename']}  (call_id {call_id}, language {t['language']})", ""]
    lines += [f"[{int(x['start'] // 60):02d}:{int(x['start'] % 60):02d}] {x['role']}: {x['text']}" for x in turns]
    txt = config.TRANSCRIPT_COPY_DIR / f"{stem}.txt"
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (config.TRANSCRIPT_COPY_DIR / f"{stem}.json").write_text(
        json.dumps({"call_id": call_id, "filename": c["filename"], "language": t["language"], "turns": turns},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return txt


def _run_job(calls: list[dict]) -> None:
    ids = [c["call_id"] for c in calls]
    for cid in ids:
        db.set_status(cid, "transcribing")
    try:
        params = dict(model=config.STT_MODEL, mode=config.STT_MODE,
                      language_code=config.STT_LANGUAGE, with_diarization=True, num_speakers=2)
        if config.STT_MODEL.startswith("saaras:v4"):
            params["keyterms"] = config.STT_KEYTERMS
        job = _client().speech_to_text_job.create_job(**params)
        job.upload_files(file_paths=[c["audio_path"] for c in calls], timeout=300)
        job.start()
        job.wait_until_complete(poll_interval=5, timeout=3600)
        out_dir = config.TRANSCRIPT_DIR
        job.download_outputs(output_dir=str(out_dir))
        failed = {f["file_name"]: f.get("error_message") for f in job.get_file_results()["failed"]}
    except Exception as e:
        log.exception("STT job failed")
        for cid in ids:
            db.set_status(cid, "error", f"stt: {e}")
        return

    for c in calls:
        name = Path(c["audio_path"]).name
        out = config.TRANSCRIPT_DIR / f"{name}.json"
        if name in failed or not out.exists():
            db.set_status(c["call_id"], "error", f"stt: {failed.get(name) or 'no output file'}")
            continue
        try:
            store_transcript(c["call_id"], json.loads(out.read_text(encoding="utf-8")), str(out))
        except Exception as e:
            log.exception("parse failed for %s", c["call_id"])
            db.set_status(c["call_id"], "error", f"stt-parse: {e}")


def transcribe_pending() -> int:
    calls = db.rows("SELECT call_id, audio_path FROM calls WHERE status='new' ORDER BY added_at")
    if not calls:
        return 0
    n = config.STT_FILES_PER_JOB
    batches = [calls[i:i + n] for i in range(0, len(calls), n)]
    log.info("transcribing %d calls in %d batch job(s)", len(calls), len(batches))
    with ThreadPoolExecutor(max_workers=config.STT_PARALLEL_JOBS) as pool:
        list(pool.map(_run_job, batches))
    ids = tuple(c["call_id"] for c in calls)
    return db.one(f"SELECT COUNT(*) AS n FROM calls WHERE status='transcribed' AND call_id IN"
                  f" ({','.join('?' * len(ids))})", ids)["n"]


def swap_speakers(call_id: str):
    """Manual correction from the dashboard when diarization picked the wrong bot speaker."""
    t = db.one("SELECT bot_speaker, turns_json FROM transcripts WHERE call_id=?", (call_id,))
    turns = db.loads(t["turns_json"], [])
    others = [s for s in dict.fromkeys(x["speaker_id"] for x in turns) if s != t["bot_speaker"]]
    if not others:
        return
    turns, bot = assign_roles(turns, bot_speaker=others[0])
    with db.connect() as c:
        c.execute("UPDATE transcripts SET bot_speaker=?, turns_json=? WHERE call_id=?",
                  (bot, json.dumps(turns, ensure_ascii=False), call_id))
        c.execute("DELETE FROM grades WHERE call_id=?", (call_id,))
        c.execute("DELETE FROM signals WHERE call_id=?", (call_id,))
    db.set_status(call_id, "transcribed")
    export_copy(call_id)
