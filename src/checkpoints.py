"""Append-only attempt journal for single-operator development runs."""
import json
import os
from pathlib import Path


class JournalError(ValueError):
    """The journal needs inspection before it can be used."""


def append_event(path: Path, event: dict, mode: str = "a") -> None:
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _journal_object(line: bytes, number: int) -> dict:
    try:
        record = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise JournalError(f"Invalid journal JSON at line {number}.") from None
    if not isinstance(record, dict):
        raise JournalError(f"Expected a journal object at line {number}.")
    return record


def _can_retry(attempts: list[dict]) -> bool:
    return (len(attempts) == 1 and attempts[0]["status"] == "error"
            and attempts[0].get("retryable") is True)


def parse_journal(payload: bytes) -> tuple[dict, dict]:
    """Read a complete UTF-8 JSONL snapshot, preserving unfinished attempts."""
    if not payload:
        raise JournalError("Journal is empty.")
    if not payload.endswith(b"\n"):
        raise JournalError("Journal ends with an unterminated record.")
    lines = iter(payload[:-1].split(b"\n"))
    header = _journal_object(next(lines), 1)
    history = {}
    for number, line in enumerate(lines, 2):
        event = _journal_object(line, number)
        question_id, kind = event.get("id"), event.get("event")
        if not isinstance(question_id, str) or not question_id:
            raise JournalError(f"Invalid question ID at journal line {number}.")
        attempts = history.setdefault(question_id, [])
        if kind == "start":
            if attempts and not _can_retry(attempts):
                raise JournalError(f"Unexpected repeated attempt at journal line {number}.")
            attempts.append({"status": "interrupted", "retryable": False})
        elif kind == "finish" and attempts and attempts[-1]["status"] == "interrupted":
            result = event.get("result")
            if not isinstance(result, dict) or result.get("status") not in (
                    "answered", "insufficient", "conflicting", "error"):
                raise JournalError(f"Invalid attempt result at journal line {number}.")
            attempts[-1] = result
        else:
            raise JournalError(f"Invalid attempt event at journal line {number}.")
    return header, history


def read_journal(path: Path) -> tuple[dict, dict]:
    return parse_journal(path.read_bytes())


def run_attempts(path: Path, header: dict, ids: list[str], execute, *, retry_failed: bool = False,
                 resume: bool = False, before_start=None) -> None:
    """Write attempts for ``ids`` in order, stopping after a result that sets ``stop_run``.

    A new run creates the journal. ``retry_failed`` repeats each recorded eligible failure once.
    ``resume`` runs only questions with no recorded start. Both continue an intact journal whose header
    equals ``header``. A start with no finish may have reached the provider, so it is never resent.
    ``before_start``, such as a provider's pacing wait, runs under the lock before each start is written.
    An interruption there leaves that question unattempted and eligible for resume.
    """
    if retry_failed and resume:
        raise ValueError("Choose either retry_failed or resume.")
    # Exclusive lock prevents concurrent writers from authorizing the same provider call.
    lock = path.with_suffix(".lock")
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        raise JournalError("Journal is locked by a collection run or a stale lock.") from None
    try:
        if retry_failed or resume:
            previous, history = read_journal(path)
            if previous != header or not history.keys() <= set(ids):
                raise JournalError("Run identity changed since the original attempts.")
            if retry_failed:
                ids = [qid for qid in ids if _can_retry(history.get(qid, []))]
            else:
                ids = [qid for qid in ids if qid not in history]
        else:
            append_event(path, header, "x")
        for qid in ids:
            if before_start is not None:
                before_start()
            append_event(path, {"event": "start", "id": qid})
            result = execute(qid)
            append_event(path, {"event": "finish", "id": qid, "result": result})
            if result.get("stop_run"):
                break
    finally:
        lock.unlink()
