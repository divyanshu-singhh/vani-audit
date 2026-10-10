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


def register(path: Path) -> tuple[str, bool]:
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
        " sample_rate, status, added_at) VALUES (?,?,?,?,?,?,?, 'new', ?)",
        (call_id, path.name, str(path), str(dest), info["duration_sec"], info["channels"],
         info["sample_rate"], db.now()),
    )
    return call_id, True


def scan_folder(folder: Path | str | None = None) -> dict:
    folder = Path(folder or config.INBOX_DIR)
    added, skipped, failed = [], 0, []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXTS or not p.is_file():
            continue
        try:
            call_id, new = register(p)
            if new:
                added.append(call_id)
            else:
                skipped += 1
        except Exception as e:  # keep scanning; report per-file failures
            failed.append(f"{p.name}: {e}")
    return {"added": added, "skipped": skipped, "failed": failed}


def save_upload(name: str, data: bytes) -> tuple[str, bool]:
    """Persist a dashboard upload into the inbox, then register it."""
    dest = config.INBOX_DIR / Path(name).name
    if dest.exists() and dest.read_bytes() != data:
        dest = dest.with_stem(f"{dest.stem}_{hashlib.sha1(data).hexdigest()[:6]}")
    dest.write_bytes(data)
    return register(dest)
