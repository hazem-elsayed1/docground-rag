import pytest
from src.checkpoints import JournalError, read_journal, run_attempts


def test_failed_items_have_one_explicit_retry_and_keep_original_outcomes(tmp_path):
    journal = tmp_path / "attempts.jsonl"
    manifest = {"model": "example"}
    cases = {"answer": {"status": "answered"}, "refusal": {"status": "insufficient"},
             "quota": {"status": "error", "retryable": True},
             "schema": {"status": "error", "retryable": False},
             "truncated": {"status": "error", "error_code": "response_truncated", "retryable": False}}
    run_attempts(journal, manifest, list(cases), cases.__getitem__)
    assert read_journal(journal)[1] == {qid: [row] for qid, row in cases.items()}
    with pytest.raises(ValueError, match="identity"):
        run_attempts(journal, {"model": "other"}, list(cases), pytest.fail, retry_failed=True)
    run_attempts(journal, manifest, list(cases), {"quota": {"status": "error", "retryable": True}}.__getitem__,
                 retry_failed=True)
    assert read_journal(journal)[1] == {qid: [row, row] if qid == "quota" else [row] for qid, row in cases.items()}
    run_attempts(journal, manifest, list(cases), pytest.fail, retry_failed=True)


def test_interruption_preserves_prior_results_without_retrying_unknown_outcome(tmp_path):
    journal = tmp_path / "attempts.jsonl"
    manifest = {"model": "example"}

    def execute(qid):
        if qid == "interrupted":
            raise KeyboardInterrupt()
        return {"status": "answered"}

    with pytest.raises(KeyboardInterrupt):
        run_attempts(journal, manifest, ["finished", "interrupted", "unattempted"], execute)
    _, history = read_journal(journal)
    assert history == {"finished": [{"status": "answered"}],
                       "interrupted": [{"status": "interrupted", "retryable": False}]}
    run_attempts(journal, manifest, ["finished", "interrupted", "unattempted"], pytest.fail, retry_failed=True)
    assert read_journal(journal)[1] == history

def test_existing_lock_blocks_retry_without_changing_journal(tmp_path):
    journal = tmp_path / "attempts.jsonl"
    header = {"model": "example"}
    run_attempts(journal, header, ["known"], lambda _: {"status": "error", "retryable": True})
    original = journal.read_bytes()
    lock = journal.with_suffix(".lock")
    lock.touch()
    with pytest.raises(JournalError, match="Journal is locked"):
        run_attempts(journal, header, ["known"], pytest.fail, retry_failed=True)
    assert journal.read_bytes() == original and lock.is_file()


@pytest.mark.parametrize(("tail", "diagnostic"), [
    (b'{"event":"start","id":"unknown"}', "unterminated"),
    (b'{"event":"finish","id":"unknown","result":{"text":"\xe2', "unterminated"),
    (b'{"event":\n', "Invalid journal JSON at line 4"),
])
def test_damaged_tail_blocks_even_otherwise_eligible_retries(tmp_path, tail, diagnostic):
    journal = tmp_path / "attempts.jsonl"
    header = {"model": "example"}
    run_attempts(journal, header, ["known"], lambda _: {"status": "error", "retryable": True})
    damaged = journal.read_bytes() + tail
    journal.write_bytes(damaged)
    with pytest.raises(JournalError, match=diagnostic):
        run_attempts(journal, header, ["known", "unknown"], pytest.fail, retry_failed=True)
    assert journal.read_bytes() == damaged and not journal.with_suffix(".lock").exists()


def test_partial_run_retries_only_the_recorded_failure(tmp_path):
    journal = tmp_path / "attempts.jsonl"
    header = {"model": "example"}
    failed = {"status": "error", "retryable": True, "stop_run": True}
    recovered = {"status": "answered"}
    ids = ["known", "unattempted"]
    run_attempts(journal, header, ids, {"known": failed}.__getitem__)
    run_attempts(journal, header, ids, {"known": recovered}.__getitem__, retry_failed=True)
    assert read_journal(journal) == (header, {"known": [failed, recovered]})


def test_second_attempt_after_completed_answer_is_rejected(tmp_path):
    journal = tmp_path / "attempts.jsonl"
    run_attempts(journal, {}, ["known"], lambda _: {"status": "answered", "retryable": True})
    with journal.open("ab") as stream:
        stream.write(b'{"event":"start","id":"known"}\n')
    with pytest.raises(JournalError, match="Unexpected repeated attempt"):
        read_journal(journal)
