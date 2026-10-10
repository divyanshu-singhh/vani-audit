"""Pipeline entry point — one command per step of the loop.

    python -m audit.pipeline process  [--folder D:/calls]   # 1-3: ingest → speech-to-text (+ saved copy) → audit
    python -m audit.pipeline suggest  [--top 5]             # 4: root causes → prompt corrections
    python -m audit.pipeline version                         # 5: corrected copy of the prompt (selected fixes)
    python -m audit.pipeline testcases                       # 6: test setup from each recording
    python -m audit.pipeline test     [--version v1] [--cases fatal|all|<id,id>]   # 8: original vs corrected
    python -m audit.pipeline all      [--folder ...]         # everything above
    python -m audit.pipeline retry | regrade
"""
import argparse
import logging
import os
import sys
import time

from . import analysis, config, db, fixes, grader, ingest, simulate, transcribe

LOCK = config.DATA_DIR / "pipeline.lock"
log = logging.getLogger("pipeline")


def _setup_logging():
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(config.LOG_DIR / "pipeline.log", encoding="utf-8")])
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _alive(pid: int) -> bool:
    if os.name == "nt":  # os.kill(pid, 0) would send CTRL_C_EVENT on Windows
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def is_running() -> bool:
    try:
        return LOCK.exists() and _alive(int(LOCK.read_text()))
    except ValueError:
        return False


def current_step() -> str:
    p = config.DATA_DIR / "pipeline.step"
    return p.read_text(encoding="utf-8") if p.exists() and is_running() else ""


def _step(text: str):
    (config.DATA_DIR / "pipeline.step").write_text(text, encoding="utf-8")
    log.info("── %s", text)


def retry_errors() -> int:
    with db.connect() as c:
        n1 = c.execute("UPDATE calls SET status='transcribed', error=NULL WHERE status='error'"
                       " AND call_id IN (SELECT call_id FROM transcripts)").rowcount
        n2 = c.execute("UPDATE calls SET status='new', error=NULL WHERE status='error'").rowcount
    return n1 + n2


def regrade_all() -> int:
    with db.connect() as c:
        return c.execute("UPDATE calls SET status='transcribed' WHERE call_id IN"
                         " (SELECT call_id FROM transcripts)").rowcount


def pick_cases(which: str) -> list[str]:
    if which == "all":
        return [r["case_id"] for r in db.rows("SELECT case_id FROM test_cases")]
    if which in ("fatal", "failed"):
        grades = ("FATAL",) if which == "fatal" else ("FATAL", "NON_FATAL")
        return [r["case_id"] for r in db.rows(
            f"SELECT t.case_id FROM test_cases t JOIN grades g ON g.call_id=t.case_id"
            f" WHERE g.grade IN ({','.join('?' * len(grades))})", grades)]
    return [x.strip() for x in which.split(",") if x.strip()]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["all", "process", "suggest", "version", "testcases", "test", "retry",
                                   "regrade", "fixes"])
    p.add_argument("--folder", default=None, help="folder to scan (default: data/inbox)")
    p.add_argument("--top", type=int, default=5, help="corrections for the top-N root causes")
    p.add_argument("--rc", default=None, help="suggest a correction for this one root cause id")
    p.add_argument("--version", default=None, help="prompt version to test (default: latest)")
    p.add_argument("--cases", default="failed", help="failed | fatal | all | comma-separated call ids")
    a = p.parse_args(argv)

    _setup_logging()
    db.init()
    if is_running():
        log.error("another pipeline run is in progress (%s)", LOCK)
        return 1
    LOCK.write_text(str(os.getpid()))
    t0 = time.time()
    try:
        if a.cmd in ("all", "process", "retry", "regrade"):
            if a.cmd in ("all", "process"):
                _step("Adding recordings")
                r = ingest.scan_folder(a.folder)
                log.info("ingest: %d new, %d already known, %d failed", len(r["added"]), r["skipped"], len(r["failed"]))
            if a.cmd == "retry":
                log.info("re-queued %d calls", retry_errors())
            if a.cmd == "regrade":
                log.info("re-queued %d calls for grading", regrade_all())
            _step("Speech-to-text")
            log.info("transcribed %d calls", transcribe.transcribe_pending())
            _step("Detecting issues")
            log.info("audited %d calls", grader.grade_pending())
        if a.cmd in ("all", "suggest", "fixes"):
            if a.rc:
                _step("Suggesting a correction")
                log.info("fix: %s", fixes.generate(a.rc))
            else:
                _step("Finding root causes")
                log.info("analysis run: %s", analysis.run())
                _step("Suggesting prompt corrections")
                log.info("fixes: %s", fixes.generate_top(a.top))
        if a.cmd in ("all", "version"):
            _step("Building corrected prompt")
            log.info("prompt version: %s", fixes.build_version())
        if a.cmd in ("all", "testcases"):
            _step("Generating test setup")
            log.info("test cases: %d", simulate.build_test_cases())
        if a.cmd in ("all", "test"):
            v = a.version or (fixes.latest_version() or {}).get("version_id")
            if not v:
                log.error("no corrected prompt yet — run 'version' first")
            else:
                cases = pick_cases(a.cases)
                _step(f"Testing {len(cases)} case(s): original vs {v}")
                log.info("test run: %s", simulate.run_test(v, cases))
    finally:
        LOCK.unlink(missing_ok=True)
        (config.DATA_DIR / "pipeline.step").unlink(missing_ok=True)
        log.info("done in %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
