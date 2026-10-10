# VANI Call Quality Audit & Fix Loop

Hackathon PS-01. Audits every recording of VANI (IndiaMART's buyer-side voice bot) against IndiaMART's
call-audit sheet, suggests prompt corrections, and proves them with simulated before/after tests.

## The loop (one dashboard tab per step)

| Tab | What happens |
|---|---|
| ① Upload | Add one recording, many, or a whole folder. **Process** = speech-to-text + issue detection. |
| ② Transcripts | Bot/buyer-separated transcript with audio. A copy is saved to `data/transcripts_text/<file>.txt/.json`. |
| ③ Issues | Every call scored on each audit-sheet parameter (OK / Non-fatal / Fatal / N/A) with evidence → overall **Pass / Non-fatal / Fatal**. Record your own verdict here. |
| ④ Corrections | Failing calls grouped into root causes; a concrete prompt change per cause. Toggle which to include. |
| ⑤ Corrected prompt | One copy of the prompt with the selected corrections merged (`data/prompts/vani_prompt_vN.md`) + diff. The original is never changed. |
| ⑥ Test setup | Per recording: what the bot knew at the start, the buyer's own lines from the recording, how the buyer behaves, what the bot must get right (`data/testcases/<file>.json`). Run the automated test from here. |
| ⑦ Manual test | You type as the buyer; the bot answers from the original and/or corrected prompt, side by side. Score the chats with the audit sheet. |
| ⑧ Results | Before vs after: recording vs original prompt vs corrected prompt — calls by result, failing calls per parameter, and each case side by side. |
| Audit criteria | The audit sheet in use, and agreement with human verdicts (import CSV or label in ③). |

## Run

```bash
.venv/Scripts/streamlit run dashboard/app.py
```

The same steps from the command line:

```bash
.venv/Scripts/python -m audit.pipeline process --folder D:/calls     # ①②③
.venv/Scripts/python -m audit.pipeline suggest                       # ④
.venv/Scripts/python -m audit.pipeline version                       # ⑤
.venv/Scripts/python -m audit.pipeline testcases                     # ⑥
.venv/Scripts/python -m audit.pipeline test --cases failed           # ⑧ (original vs latest corrected prompt)
.venv/Scripts/python -m audit.pipeline all --folder D:/calls         # everything
.venv/Scripts/python scripts/smoke_test.py                           # offline check, no API credits
```

## How a call is graded

1. **Speech-to-text:** Sarvam `saaras:v4` batch API with 2-speaker diarization. VANI is identified by the phrases its prompt scripts.
2. **Audit sheet** (`audit/rubric.py`): Outcome accuracy, quantity value/unit/probing/missed, spec 1/2 accuracy, spec missed, looping, WER, additional details, OnCall AstBuy, product, ISQ, MCAT, dead air, plus two VANI prompt rules (compliance, wrong info).
3. **Three independent scorecards** from `sarvam-105b` run in parallel; each parameter takes the majority status.
4. **Measured rules:** dead air = more than 5 s of silence before a bot turn (from timestamps); looping = more than 1+2 asks of one field; spoken prompt variables = compliance.
5. **Second reviewer:** re-checks every fatal finding against the transcript and a numbered list of "not an issue" rules, and drops or downgrades unsupported findings.

`sarvam-105b` runs with reasoning **off**: with reasoning on, it spent the whole token budget thinking and never answered. JSON mode is `json_object`, because strict `json_schema` mode degenerated into endless whitespace.

## Testing a correction

The simulated buyer replays the recording's buyer lines in order, adapting only so they fit what the bot says. The bot runs the real production prompt, rendered with the call's variables (Jinja `{% if %}` blocks and `bot_name`-style placeholders). Its `@variable` updates are applied and its tools are stubbed (`SIM_BACKEND=local`). Set `SIM_BACKEND=samvaad` with `SAMVAAD_*` in `.env` to drive the real Sarvam agent instead. Both prompts are scored with the same audit sheet.

**Limits:** it is a text simulation, so there are no STT or TTS errors (WER and dead air are N/A). Tool results are stubs.
