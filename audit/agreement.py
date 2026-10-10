"""Human labels and agreement of auto-grades with them (25% of the judging score)."""
import csv
import io

from . import db


def normalise_label(raw: str) -> str | None:
    s = (raw or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if s in ("fatal", "f", "1", "fail", "failed", "yes"):
        return "FATAL"
    if s in ("nonfatal", "nf", "minor"):
        return "NON_FATAL"
    if s in ("pass", "ok", "clean", "0", "no", "p"):
        return "PASS"
    return None


def set_label(call_id: str, label: str, notes: str = "", labeler: str = ""):
    db.execute("INSERT OR REPLACE INTO human_labels (call_id, label, notes, labeler, created_at)"
               " VALUES (?,?,?,?,?)", (call_id, label, notes, labeler, db.now()))


def import_csv(data: bytes, labeler: str = "csv") -> dict:
    """CSV with a `call_id` or `filename` column and a `label` column (Fatal / Non-Fatal)."""
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    by_name = {r["filename"]: r["call_id"] for r in db.rows("SELECT call_id, filename FROM calls")}
    by_stem = {k.rsplit(".", 1)[0]: v for k, v in by_name.items()}
    ok, missing, bad = 0, [], []
    for row in reader:
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        label = normalise_label(row.get("label") or row.get("grade") or row.get("human_label") or "")
        key = row.get("call_id") or row.get("filename") or row.get("file") or ""
        cid = key if db.one("SELECT 1 FROM calls WHERE call_id=?", (key,)) else \
            by_name.get(key) or by_stem.get(key.rsplit(".", 1)[0])
        if not label:
            bad.append(key)
        elif not cid:
            missing.append(key)
        else:
            set_label(cid, label, row.get("notes", ""), labeler)
            ok += 1
    return {"imported": ok, "unmatched": missing, "bad_label": bad}


def metrics() -> dict | None:
    pairs = db.rows("SELECT h.call_id, h.label AS human, g.grade AS auto FROM human_labels h"
                    " JOIN grades g USING(call_id)")
    n = len(pairs)
    if not n:
        return None
    hf = [p["human"] == "FATAL" for p in pairs]   # agreement is measured on Fatal vs not-Fatal
    af = [p["auto"] == "FATAL" for p in pairs]
    tp = sum(h and a for h, a in zip(hf, af))
    tn = sum(not h and not a for h, a in zip(hf, af))
    fp = sum(not h and a for h, a in zip(hf, af))
    fn = sum(h and not a for h, a in zip(hf, af))
    acc = (tp + tn) / n
    p_h, p_a = (tp + fn) / n, (tp + fp) / n
    pe = p_h * p_a + (1 - p_h) * (1 - p_a)
    kappa = (acc - pe) / (1 - pe) if pe < 1 else 1.0
    return {
        "n": n, "accuracy": acc, "kappa": kappa,
        "precision_fatal": tp / (tp + fp) if tp + fp else None,
        "recall_fatal": tp / (tp + fn) if tp + fn else None,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "exact_match": sum(p["human"] == p["auto"] for p in pairs) / n,
        "disagreements": [p for p in pairs if p["human"] != p["auto"]],
    }
