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
| ⑧ Results | Recording vs corrected prompt — calls by result, failing calls per parameter, and each case side by side. |
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

## Which model does what

| Step | Model |
|---|---|
| Speech-to-text | Sarvam `saaras:v4` |
| Deciding errors / fatal / non-fatal (scorecard + second review) | **Judge**: GPT or Sonnet via the IM LLM gateway (`JUDGE_API_KEY`, `JUDGE_MODEL` in `.env`); falls back to Sarvam if no key |
| Test setup, simulation (bot + buyer), root causes, corrections | Sarvam `sarvam-105b` |

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

## Batches, auto-improve rounds, human review

- **Upload batches** — every "Add files" / "Add folder" is one batch with its time. Upload and Issues
  list calls grouped by batch, newest first; Issues can be filtered by batch.
- **Auto-improve rounds** (`audit/refine.py`) — after a test, the corrected prompt is scored against the
  recordings (Fatal 10 · Non-fatal 3 · +1 per failing parameter; lower is better). If it is worse — or no
  better while the recordings had issues — it is revised from what went wrong in the replays, saved as the
  next version and re-tested, up to `REFINE_MAX_ROUNDS` (default 3). ⑧ Results shows the chain, e.g.
  `v3 · worse → v4 · no improvement → v5 · better`. `python -m audit.pipeline test --no-refine` tests once.
- **Needs human review** — only when the bot cannot stand behind its verdict: (1) it found a MAJOR problem
  that no audit parameter covers (shown as "Outside audit sheet"), or (2) the call's result rests entirely on
  findings below `REVIEW_CONFIDENCE` (default 0.5). Everything else, including findings the second reviewer
  overturned, is decided automatically. Your verdict in Quick view resolves a flagged call.
