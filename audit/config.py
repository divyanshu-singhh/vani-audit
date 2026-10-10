"""Central configuration, read from environment / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else ROOT / p


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
if SARVAM_API_KEY.startswith("sk_xxxx"):  # untouched .env.example placeholder
    SARVAM_API_KEY = ""

STT_MODEL = os.getenv("STT_MODEL", "saaras:v4")
STT_MODE = os.getenv("STT_MODE", "transcribe")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "unknown")
STT_FILES_PER_JOB = min(_int("STT_FILES_PER_JOB", 20), 20)
STT_PARALLEL_JOBS = _int("STT_PARALLEL_JOBS", 3)
# Biases the recogniser towards domain words. saaras:v4 only, max 50 terms.
STT_KEYTERMS = [
    "IndiaMART", "IndiaMart", "seller", "buyer", "requirement", "WhatsApp",
    "quotation", "transfer", "Help Desk", "five times nine six",
]

LLM_MODEL = os.getenv("LLM_MODEL", "sarvam-105b")
GRADER_REASONING = os.getenv("GRADER_REASONING", "off")
ANALYSIS_REASONING = os.getenv("ANALYSIS_REASONING", "off")
# Judge = the model that decides errors / fatal / non-fatal (grading + second review).
# Uses the IM LLM gateway (OpenAI-compatible) when JUDGE_API_KEY is set; otherwise falls back to Sarvam.
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", "").strip()
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-5.4")
JUDGE_REASONING = os.getenv("JUDGE_REASONING", "low")  # GPT-5: minimal | low | medium | high
GRADER_WORKERS = _int("GRADER_WORKERS", 4)
GRADER_SAMPLES = _int("GRADER_SAMPLES", 1 if JUDGE_API_KEY else 3)  # votes per call (GPT/Sonnet: 1 is enough)
GRADER_INCLUDE_FULL_PROMPT = os.getenv("GRADER_INCLUDE_FULL_PROMPT", "0") == "1"
REVIEW_CONFIDENCE = float(os.getenv("REVIEW_CONFIDENCE", "0.6"))  # below this, a decision goes to human review
REFINE_MAX_ROUNDS = _int("REFINE_MAX_ROUNDS", 3)  # corrected prompt worse? revise + re-test up to N times

# Fix testing: replay failed calls against baseline vs fixed prompt.
# local   = we run the prompt ourselves on Sarvam LLM with stubbed tools (fully automatic)
# samvaad = drive your real Sarvam agent over the Conv AI SDK (CHAT mode); needs a committed
#           app version per prompt (baseline below; fixed version entered per fix on the dashboard)
SIM_BACKEND = os.getenv("SIM_BACKEND", "local")
SIM_BOT_MODEL = os.getenv("SIM_BOT_MODEL", "sarvam-105b")
SIM_BOT_REASONING = os.getenv("SIM_BOT_REASONING", "off")
SIM_BUYER_MODEL = os.getenv("SIM_BUYER_MODEL", "sarvam-105b")
SIM_CASES_PER_FIX = _int("SIM_CASES_PER_FIX", 4)       # failed real calls replayed per fix
SIM_SCENARIOS_PER_FIX = _int("SIM_SCENARIOS_PER_FIX", 2)  # extra scripted scenarios per fix
SIM_MAX_TURNS = _int("SIM_MAX_TURNS", 14)               # bot turns per simulated call
SIM_WORKERS = _int("SIM_WORKERS", 4)
SAMVAAD_ORG_ID = os.getenv("SAMVAAD_ORG_ID", "")
SAMVAAD_WORKSPACE_ID = os.getenv("SAMVAAD_WORKSPACE_ID", "")
SAMVAAD_APP_ID = os.getenv("SAMVAAD_APP_ID", "")
SAMVAAD_BASELINE_VERSION = os.getenv("SAMVAAD_BASELINE_VERSION", "")
SAMVAAD_CORRECTED_VERSION = os.getenv("SAMVAAD_CORRECTED_VERSION", "")  # app version with the corrected prompt
SAMVAAD_BASE_URL = os.getenv("SAMVAAD_BASE_URL", "https://apps.sarvam.ai/api/app-runtime/")

DATA_DIR = ROOT / "data"
AUDIT_DB = _path("AUDIT_DB", "data/audit.db")
INBOX_DIR = _path("INBOX_DIR", "data/inbox")
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
FIX_DIR = DATA_DIR / "fixes"
LOG_DIR = DATA_DIR / "logs"
BOT_PROMPT_PATH = _path("BOT_PROMPT_PATH", "prompts/vani_buyer_prompt.md")

TRANSCRIPT_COPY_DIR = DATA_DIR / "transcripts_text"   # readable STT copy per call (step 2)
PROMPT_VERSION_DIR = DATA_DIR / "prompts"             # corrected prompt copies (step 5)
TESTCASE_DIR = DATA_DIR / "testcases"                 # replayable test setups (step 6)

for d in (DATA_DIR, INBOX_DIR, AUDIO_DIR, TRANSCRIPT_DIR, FIX_DIR, LOG_DIR, TRANSCRIPT_COPY_DIR,
          PROMPT_VERSION_DIR, TESTCASE_DIR):
    d.mkdir(parents=True, exist_ok=True)
