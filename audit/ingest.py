"""Register recordings: from a folder scan or from dashboard uploads. Idempotent — the
call_id is a content hash, so the same file is never processed twice."""
import hashlib
import shutil
from pathlib import Path

from . import audio, config, db

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def new_batch(source: str, label: str = "") -> int:
    """One upload action = one batch (files uploaded together, or one folder scan)."""
    with db.connect() as c:
        return c.execute("INSERT INTO batches (created_at, source, label) VALUES (?,?,?)",
                         (db.now(), source, label)).lastrowid


def drop_if_empty(batch_id: int):
    db.execute("DELETE FROM batches WHERE batch_id=? AND NOT EXISTS (SELECT 1 FROM calls WHERE batch_id=?)",
               (batch_id, batch_id))


def register(path: Path, batch_id: int | None = None) -> tuple[str, bool]:
    """Copy one file into data/audio and add a `calls` row. Returns (call_id, is_new)."""
    path = Path(path)
    sha = _sha1(path)
    call_id = sha[:12]
    if db.one("SELECT 1 FROM calls WHERE call_id=?", (call_id,)):
        return call_id, False
    dest = config.AUDIO_DIR / f"{call_id}{path.suffix.lower()}"
    if not dest.exists():
        shutil.copy2(path, dest)
    info = audio.probe(dest)
    db.execute(
        "INSERT INTO calls (call_id, filename, src_path, audio_path, duration_sec, channels,"
        " sample_rate, status, added_at, batch_id) VALUES (?,?,?,?,?,?,?, 'new', ?, ?)",
        (call_id, path.name, str(path), str(dest), info["duration_sec"], info["channels"],
         info["sample_rate"], db.now(), batch_id),
    )
    return call_id, True


def scan_folder(folder: Path | str | None = None, limit: int = 0) -> dict:
    folder = Path(folder or config.INBOX_DIR)
    added, skipped, failed = [], 0, []
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTS and p.is_file())
    batch = new_batch("folder", folder.name)
    for p in files:
        if limit and len(added) >= limit:
            break
        try:
            call_id, new = register(p, batch)
            if new:
                added.append(call_id)
            else:
                skipped += 1
        except Exception as e:  # keep scanning; report per-file failures
            failed.append(f"{p.name}: {e}")
    drop_if_empty(batch)
    return {"added": added, "skipped": skipped, "failed": failed, "batch_id": batch if added else None}


def save_upload(name: str, data: bytes, batch_id: int | None = None) -> tuple[str, bool]:
    """Persist a dashboard upload into the inbox, then register it."""
    dest = config.INBOX_DIR / Path(name).name
    if dest.exists() and dest.read_bytes() != data:
        dest = dest.with_stem(f"{dest.stem}_{hashlib.sha1(data).hexdigest()[:6]}")
    dest.write_bytes(data)
    return register(dest, batch_id)
