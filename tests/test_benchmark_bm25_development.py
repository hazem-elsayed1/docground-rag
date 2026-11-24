from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import benchmark_bm25_development as benchmark


@pytest.mark.parametrize("relative", list(benchmark.INPUT_SHA256))
def test_pinned_input_rejects_changed_bytes(tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{}")
    with pytest.raises(ValueError, match=f"SHA-256 mismatch for {relative.as_posix()}"):
        benchmark.pinned_input(tmp_path.resolve(), relative)


def test_cli_rejects_an_alternate_annotation_path(tmp_path):
    synthetic_path = tmp_path / "frozen-test" / "labels.json"
    outcome = subprocess.run(
        [sys.executable, "-m", "scripts.benchmark_bm25_development", str(synthetic_path)],
        capture_output=True, text=True,
    )
    assert outcome.returncode == 2
    assert "unrecognized arguments" in outcome.stderr
    assert not outcome.stdout


def test_input_reader_rejects_an_alternate_path_before_opening(tmp_path):
    alternate = Path("annotations/frozen-test/synthetic.json")
    with pytest.raises(ValueError, match="Only the fixed benchmark input paths are permitted"):
        benchmark.pinned_input(tmp_path.resolve(), alternate)


def test_fixed_path_with_frozen_split_is_rejected(tmp_path, monkeypatch):
    for relative in benchmark.INPUT_SHA256:
        payload = b'{"split": "frozen-test"}' if relative == benchmark.ANNOTATIONS else b"{}"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        monkeypatch.setitem(benchmark.INPUT_SHA256, relative, sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="requires the development split"):
        benchmark.load_inputs(tmp_path.resolve())
