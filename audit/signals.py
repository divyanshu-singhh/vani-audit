"""Deterministic checks over a transcript. Cheap, explainable, and fed to the LLM grader
as hints. Each finding carries the turn index + timestamp it came from."""
import json
import re

from rapidfuzz import fuzz

from . import db

# Buyer asks for a person / objects to the bot (TALKING_TO_AI / refuses-AI path).
HUMAN_REQUEST = [
    r"\binsaan\b", r"\bhuman\b", r"real person", r"\bmanager\b", r"kisi aadmi", r"aadmi se",
    r"\brobot\b", r"\bbot\b", r"\bmachine\b", r"recorded", r"इंसान", r"रोबोट", r"मशीन",
    r"आदमी से", r"मैनेजर", r"असली",
]
# Buyer irritation / repeating themselves.
FRUSTRATION = [
    r"kitni baar", r"bola na", r"bataya na", r"already bata", r"pehle hi bata", r"bakwas",
    r"faltu", r"time waste", r"pareshan", r"dimag", r"pagal", r"samajh nahi aata",
    r"कितनी बार", r"बोला ना", r"बताया ना", r"बता दिया", r"बकवास", r"फालतू", r"परेशान",
    r"दिमाग", r"पागल", r"टाइम वेस्ट",
]
# Audio trouble (CONNECTIVITY_ISSUE).
CONNECTIVITY = [
    r"awaaz nahi", r"aawaz nahi", r"sunai nahi", r"hello\W+hello", r"can you hear",
    r"आवाज़ नहीं", r"आवाज नहीं", r"सुनाई नहीं", r"हेलो\W+हेलो", r"हैलो\W+हैलो",
]
# Bot asking the buyer to repeat → low-ASR-confidence proxy (no confidence scores from batch STT).
BOT_REPROMPT = [
    r"samajh nahi aaya", r"phir se", r"dobara bol", r"repeat", r"clear nahi",
    r"समझ नहीं आया", r"फिर से", r"दोबारा", r"साफ़ नहीं", r"साफ नहीं",
]
BUYER_HUH = [r"^\W*(kya|haan\?|huh|what|sorry|क्या|हां\?|जी\?)\W*$"]
# Prompt's TTS rules forbid speaking these.
LEAK = [
    r"buyer_disposition", r"call_outcome", r"turn_intent", r"end_interaction", r"current_phase",
    r"LOOP_DETECTED", r"identity_revealed", r"_tool\b", r"\{%", r"\{\{", r"@\s?\w+_\w+",
    r"\*\*", r"BL Approved", r"BL Enriched", r"BL Deleted",
]
CLAIMS_HUMAN = [r"main (ek )?insaan", r"i am (a )?human", r"मैं (एक )?इंसान", r"main seller hoon",
                r"मैं सेलर", r"i am the seller"]
CLOSING = [r"dhanyavaad", r"धन्यवाद", r"thank you", r"शुक्रिया", r"shukriya", r"great day",
           r"nice day", r"शुभ", r"दिन शुभ", r"थैंक"]
INFORMAL_FILLERS = [r"\bdekho\b", r"\bsuno\b", r"\barre\b", r"\byaar\b", r"\bruko\b",
                    r"देखो", r"सुनो", r"अरे", r"यार", r"रुको"]


def _hits(patterns, text):
    return [p for p in patterns if re.search(p, text, re.IGNORECASE)]


def _ev(t, why):
    return {"turn": t["idx"], "t": round(t["start"], 1), "quote": t["text"][:200], "why": why}


def compute(turns: list[dict], duration: float | None) -> dict:
    bot = [t for t in turns if t["role"] == "BOT"]
    buyer = [t for t in turns if t["role"] == "BUYER"]
    s: dict = {
        "n_turns": len(turns), "n_bot_turns": len(bot), "n_buyer_turns": len(buyer),
        "duration_sec": duration,
        "bot_talk_share": round(sum(len(t["text"]) for t in bot) / max(1, sum(len(t["text"]) for t in turns)), 2),
    }
    ev: dict[str, list] = {k: [] for k in (
        "bot_repeats", "buyer_repeats", "human_request", "frustration", "connectivity",
        "bot_reprompts", "buyer_confused", "prompt_leak", "claims_human_or_seller",
        "informal_filler", "long_silence")}

    # Loops: near-duplicate bot utterances (the same question asked again).
    for i, a in enumerate(bot):
        for b in bot[i + 1:]:
            if len(a["text"]) >= 20 and fuzz.token_set_ratio(a["text"], b["text"]) >= 85:
                ev["bot_repeats"].append(_ev(b, f"repeats turn {a['idx']}"))
                break
    for i, a in enumerate(buyer):
        for b in buyer[i + 1:]:
            if len(a["text"]) >= 12 and fuzz.token_set_ratio(a["text"], b["text"]) >= 85:
                ev["buyer_repeats"].append(_ev(b, f"buyer repeats turn {a['idx']}"))
                break

    for t in buyer:
        if _hits(HUMAN_REQUEST, t["text"]):
            ev["human_request"].append(_ev(t, "asks for a human / objects to bot"))
        if _hits(FRUSTRATION, t["text"]):
            ev["frustration"].append(_ev(t, "frustration phrase"))
        if _hits(CONNECTIVITY, t["text"]):
            ev["connectivity"].append(_ev(t, "audio/network complaint"))
        if _hits(BUYER_HUH, t["text"]):
            ev["buyer_confused"].append(_ev(t, "buyer did not follow"))
    for t in bot:
        if _hits(BOT_REPROMPT, t["text"]):
            ev["bot_reprompts"].append(_ev(t, "bot asks buyer to repeat"))
        if h := _hits(LEAK, t["text"]):
            ev["prompt_leak"].append(_ev(t, f"internal token spoken: {h[0]}"))
        if _hits(CLAIMS_HUMAN, t["text"]):
            ev["claims_human_or_seller"].append(_ev(t, "claims to be human/seller"))
        if _hits(INFORMAL_FILLERS, t["text"]):
            ev["informal_filler"].append(_ev(t, "informal filler"))

    for prev, cur in zip(turns, turns[1:]):
        gap = cur["start"] - prev["end"]
        if gap >= 6:
            ev["long_silence"].append(_ev(cur, f"{gap:.0f}s silence before this turn"))

    # How did the call end?
    end = "empty"
    if turns:
        last = turns[-1]
        if last["role"] == "BOT":
            end = "bot_closed" if _hits(CLOSING, last["text"]) else "bot_cut_mid_flow"
        else:
            end = "buyer_last"  # buyer hung up / dropped, or bot never answered
        if last["role"] == "BOT" and last["text"].rstrip().endswith("?"):
            end = "bot_cut_mid_flow"  # bot's last words were a question → call died mid-flow
    s["end_state"] = end

    s["evidence"] = ev
    s["counts"] = {k: len(v) for k, v in ev.items()}
    s["risk_hints"] = {
        "loop": len(ev["bot_repeats"]) >= 2,
        "frustration": bool(ev["frustration"]) or len(ev["buyer_repeats"]) >= 2,
        "human_request": bool(ev["human_request"]),
        "low_confidence": len(ev["bot_reprompts"]) + len(ev["buyer_confused"]) >= 2,
    }
    return s


def compute_for(call_id: str) -> dict:
    t = db.one("SELECT turns_json FROM transcripts WHERE call_id=?", (call_id,))
    c = db.one("SELECT duration_sec FROM calls WHERE call_id=?", (call_id,))
    s = compute(db.loads(t["turns_json"], []), c["duration_sec"])
    db.execute("INSERT OR REPLACE INTO signals (call_id, signals_json) VALUES (?,?)",
               (call_id, json.dumps(s, ensure_ascii=False)))
    return s
