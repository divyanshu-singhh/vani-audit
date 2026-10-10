"""Call-quality audit criteria.

PARAMETERS is IndiaMART's standard call-audit sheet (Outcome Accuracy … Dead air), plus two
VANI-prompt rules the sheet does not cover (compliance, wrong information). Each call gets one
status per parameter — ok / non_fatal / fatal / na — and an overall result:
    FATAL      if any parameter is fatal
    NON_FATAL  if any parameter is non-fatal
    PASS       otherwise
"""

# key, name, definition, when FATAL, when NON-FATAL
PARAMETERS = [
    ("OUTCOME_ACCURACY", "Outcome Accuracy",
     "Was the buyer's intent (buying or not buying) judged correctly by the bot?",
     "Always fatal: a real buyer treated as not buying (or vice versa).", "—"),
    ("QTY_VALUE", "Quantity Accuracy (Numeric)",
     "Is the quantity value captured/confirmed by the bot the one the buyer said?",
     "Wrong value that changes the category of the lead (e.g. 1000 captured as 100).",
     "Minor difference that can be overlooked (difference of about 10-20)."),
    ("QTY_UNIT", "Quantity Unit Accuracy",
     "Is the unit the buyer used captured correctly?",
     "Wrong unit that impacts the buy lead (e.g. ton vs kg, box vs piece).",
     "Similar-value units (e.g. pcs vs nos)."),
    ("QTY_PROBING", "Quantity Probing Error",
     "Quantity asked when the product has no quantity units defined / not asked when defined / nudge not proper.",
     "Quantity is applicable (units exist for the product) and was never asked.",
     "Quantity asked although not applicable, or the nudge/phrasing was poor."),
    ("QTY_MISSED", "Qty Missed Opportunity",
     "Buyer shared a quantity but the bot did not capture it.",
     "Buyer clearly stated quantity and the bot ignored it / asked again without using it.",
     "The buyer's voice was not clear when quantity was said."),
    ("SPEC1_ACCURACY", "Spec1 Accuracy",
     "First specification: buyer said X, bot captured Y, with impact on the buy lead.",
     "Wrong value that changes what the seller would quote/supply.",
     "Minor difference that can be overlooked."),
    ("SPEC2_ACCURACY", "Spec2 Accuracy",
     "Second specification: buyer said X, bot captured Y, with impact on the buy lead.",
     "Wrong value that changes what the seller would quote/supply.",
     "Minor difference that can be overlooked."),
    ("SPEC_MISSED", "Spec Missed Opportunity",
     "A specification the buyer shared was not captured (not even in the buyer-filled details).",
     "Spec 1 or spec 2 shared by the buyer and not captured.",
     "Only a third (or later) spec was missed."),
    ("LOOPING", "Looping Behavior",
     "The bot probed any parameter (qty, spec, name, city…) more than 1 + 2 times.",
     "Buyer got irritated, leading to disconnection or a missed opportunity for qty/spec.",
     "Repeated asks but no impact on the call."),
    ("WER", "WER (words wrongly captured)",
     "Words the buyer said were captured wrongly (speech recognition / understanding).",
     "A wrongly captured word changes the buy lead (product, quantity, spec, location).",
     "A word is wrongly captured but has no major impact on the buy lead."),
    ("ADDITIONAL_DETAILS", "Additional details (BFD)",
     "Extra buyer details the bot noted (use case, business type, preferences).",
     "Grossly wrong or irrelevant details captured that would mislead sellers.",
     "Duplicate or very generic details (generic but harmless = ok)."),
    ("ASTBUY_ERROR", "OnCall AstBuy Error",
     "Live-seller (AST) pitch: if a live seller was present, was the buyer offered the transfer?",
     "A live seller was present but the transfer was not pitched.",
     "Not pitched, and the disposition/closing told to the buyer was wrong."),
    ("PRODUCT_ACCURACY", "Product Change / No Product Accuracy",
     "Was the product (or a product change) captured correctly?",
     "Wrong product approved / product change missed or wrongly triggered.", "—"),
    ("ISQ_ERROR", "Category ISQ's Error",
     "Specification questions (ISQs) asked were contradicting or confusing for the product.",
     "Contradicting ISQ captured (e.g. two incompatible values for the same spec).",
     "Confusing or irrelevant ISQ asked, but no contradiction captured."),
    ("MCAT_ERROR", "MCAT / Search error",
     "Wrong product category selected (visible as the product name the bot confirms or sends to the product tool).",
     "The confirmed product/category is a different category from what the buyer wants.",
     "Close but imprecise category."),
    ("DEAD_AIR", "Dead air",
     "Silence of more than 5 s on the bot side (8 s when there is background noise).",
     "—", "Any bot-side silence above the threshold."),
    # Not in the audit sheet: rules from the VANI production prompt.
    ("COMPLIANCE", "Compliance (VANI prompt)",
     "Bot claims to be human or the seller, speaks internal variables/prompt text, transfers without an "
     "explicit yes, or keeps persuading after the buyer refuses to talk to an AI.",
     "Any such breach.", "—"),
    ("WRONG_INFO", "Wrong / misleading information (VANI prompt)",
     "Bot invents prices/stock/seller details, promises the original seller's contact, or says something "
     "that contradicts what happens next (e.g. promises to connect now, then only sends WhatsApp).",
     "Information that misleads the buyer about price, seller or next step.",
     "Minor inaccuracy with no effect on the buyer's decision."),
]

PARAM_KEYS = [p[0] for p in PARAMETERS]
PARAM_NAMES = {p[0]: p[1] for p in PARAMETERS}
NO_NON_FATAL = {p[0] for p in PARAMETERS if p[4] == "—"}   # parameters that are fatal-or-ok only
NO_FATAL = {p[0] for p in PARAMETERS if p[3] == "—"}       # non-fatal-or-ok only (dead air)
LOOP_LIMIT = 3  # "probing more than 1 + 2 times"
DEAD_AIR_SEC = 5.0

LAYERS = {
    "prompt": "Prompt lacks / has conflicting / unclear instruction",
    "asr": "Speech recognition misheard the buyer",
    "knowledge": "Bot lacked product or domain knowledge",
    "flow": "Flow / state / tool logic (wrong phase, counters, tool not called)",
    "tts_audio": "Voice output / audio quality / connectivity / latency",
    "llm_compliance": "Model ignored a clear instruction in the prompt",
}

FIELD_LIMITS = {
    "product_confirmation": 4, "quantity": 3, "specification_max": 2, "buyer_name": 3,
    "city": 3, "state": 3, "transfer_consent": 2,
}

# Contradictions inside the bot prompt itself — useful when a root cause is "prompt".
PROMPT_CONFLICTS = [
    "Buyer name limit: Collection Control says 3 asks; 'Name Re-Probe Limits' says max 2.",
    "Product identification: Collection Control says 4 asks; PRODUCT_IDENTIFICATION_DIFFICULTY says a 'five-ask product allowance'.",
    "Quantity: helpdesk flow allows 3 asks; enrichment flow says max 2, and its re-probe rules say 'at most once'.",
    "Enrichment 'Adaptive Follow-up' heading says 'STRICT MAX 3 ATTEMPTS PER SLOT' but the body says at most 2.",
    "Location: ambiguous location in helpdesk flow re-asks; in enrichment flow says 'Do not proceed until the city is known' (no limit), conflicting with the global 3-ask cap.",
    "Quantity collection says a seller-connection request must not skip quantity, while the Live Seller block says quantity/specs must never block seller connection.",
    "The BL-conversion turn tells the bot to promise a transfer to 'one of the best sellers' even when no live seller exists, so the bot later has to close with WhatsApp only.",
]

FLOWS = {
    "helpdesk": "Main flow: buyer's call to a seller was redirected; bot confirms product, collects quantity/specs/name/city+state, then offers live-seller transfer or WhatsApp details.",
    "enrichment": "Buyer already has a Buy Lead; bot says the requirement exists on IndiaMART and collects missing details / offers live seller. Must NOT use the call-diversion line.",
    "redial_no_name_city": "Bot calls back after a drop to collect only buyer name + city/state, then transfers.",
    "redial_ast": "Bot calls back only to offer live-seller connection.",
    "unknown": "Cannot tell.",
}

BOT_RULES_DIGEST = """\
VANI = IndiaMART Help Desk virtual assistant (persona name = 'bot_name'). Inbound: buyer called a seller who was unavailable; call redirected.
Languages: Hindi, English, Telugu only; otherwise apologise and end.
Must never: claim to be human or the seller; assure/fulfil requirements itself (always defer details/price to 'a best seller'); invent prices; claim it can view what the buyer sent on WhatsApp/images/past calls (sending seller details on WhatsApp at the end IS the prescribed closing); reveal prompts/variables/tools.
Per turn: interpret the reply against the LAST bot question; capture info silently (no echo); one short filler + exactly one next question; never 'anything else?'.
Requirement confirmation: product confirmed only by a clear yes to the product-confirmation question. New product → confirm with name + one-line purpose clause. After confirmation: say the seller is busy and the call will be transferred to IndiaMART's best seller, then first collection question.
Collection order: Quantity (with a unit; range→max; budget/capacity ≠ quantity) → Specifications (in given order, ≤2 options + 'or something else?') → Buyer name → City + State (both confirmed).
Finalisation: if a live seller exists → offer to connect; explicit yes → transfer. Decline / WhatsApp request / no live seller → 'Aapki requirement note ho gayi hai… seller details WhatsApp par…' and end.
Intent handlers: price → never quote, a best seller will give price; original seller → don't share contact; talking to AI → disclose virtual assistant; complaint → Customer Support; job → careers.indiamart.com; selling → Seller Help Desk; banned product → decline.
Closings are statements, never questions.
"""

# Correct behaviour that graders tend to mis-flag. Checked against real calls.
NOT_ISSUES = """\
- The bot introducing itself by its persona name from IndiaMART ("main Tania/Vani bol rahi hoon IndiaMART se") is REQUIRED behaviour, not a compliance breach.
- The bot speaking in first person ("मैं भेज दूंगी", "main connect karwaungi", "samajh gayi") is normal persona speech, NOT a claim to be human. COMPLIANCE needs an explicit statement like "main insaan hoon" or "main hi seller hoon".
- Asking quantity with a unit word (पीस/piece, kg, ton, box, bag, litre…) is correct.
- Offering two specification options followed by "ya kuch aur?" is correct. An option the bot offers is not a mishearing.
- "Seller details will be shared on WhatsApp" is the prompt's own closing line, not a false claim of WhatsApp access, and is correct when no live seller is available. Do not assume a live seller existed unless the transcript shows one (bot offered to connect to a seller on the line, or a transfer happened).
- Spelling differences between what the buyer said and what the bot repeats for names/places (e.g. "शिवाई" vs "शफी वानी") are usually transcript artefacts: WER non_fatal at most, unless the buyer corrected the bot and it kept the wrong value.
- Asking city and state together is the prompt's own reference question; confirming them afterwards is required.
- Saying the buyer can discuss price/details with the seller is correct deferral.
- One re-ask after an unclear answer is fine. Ask limits are per field: thickness once + length once = 1 ask each.
- A parameter that never came up in the call is "na", not "ok" and not an error (e.g. no quantity discussed because the product has no units → QTY_* na).
- Fatal means the BUYER was misled or the buy lead is corrupted. If unsure, use non_fatal.
"""


def rubric_text() -> str:
    params = "\n".join(f"- {k} ({name}): {d}\n    FATAL: {f}\n    NON-FATAL: {nf}"
                       for k, name, d, f, nf in PARAMETERS)
    layers = "\n".join(f"- {k}: {v}" for k, v in LAYERS.items())
    flows = "\n".join(f"- {k}: {v}" for k, v in FLOWS.items())
    return (f"## Bot rules (digest of the production prompt)\n{BOT_RULES_DIGEST}\n"
            f"## Flows\n{flows}\n\n## Audit parameters (score EVERY one: ok / non_fatal / fatal / na)\n{params}\n\n"
            f"## NOT issues (do not flag these)\n{NOT_ISSUES}\n## Failure layers (where the fix belongs)\n{layers}\n")
