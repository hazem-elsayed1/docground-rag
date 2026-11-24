from hashlib import sha256
from io import BytesIO
from urllib.request import Request

import pytest

from scripts import fetch_common_pitfalls as fetch


@pytest.fixture
def snapshot(monkeypatch):
    payload = b"<html>pinned fixture</html>"
    monkeypatch.setattr(fetch, "SOURCE_SHA256", sha256(payload).hexdigest())
    return payload


@pytest.mark.parametrize("matches", [True, False])
def test_existing_file_is_never_downloaded_or_replaced(tmp_path, monkeypatch, snapshot, matches):
    path = tmp_path / "common_pitfalls.html"
    original = snapshot if matches else b"existing different snapshot"
    path.write_bytes(original)

    def unexpected_download(*args, **kwargs):
        pytest.fail("Attempted to download over an existing snapshot")

    monkeypatch.setattr(fetch._opener, "open", unexpected_download)
    if matches:
        assert fetch.fetch_snapshot(path) == "Verified existing snapshot"
    else:
        with pytest.raises(ValueError, match="expected SHA-256"):
            fetch.fetch_snapshot(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize(("kind", "error"), [
    ("matching", None),
    ("mismatch", "expected SHA-256"),
    ("oversize", "exceeds the size limit"),
])
def test_download_is_saved_only_after_verification(tmp_path, monkeypatch, snapshot, kind, error):
    path = tmp_path / "raw" / "common_pitfalls.html"
    payload = {
        "matching": snapshot, "mismatch": b"changed upstream",
        "oversize": b"x" * (fetch.MAX_BYTES + 1),
    }[kind]
    monkeypatch.setattr(fetch._opener, "open", lambda url, timeout: BytesIO(payload))
    if error is None:
        assert fetch.fetch_snapshot(path) == "Saved verified snapshot"
        assert path.read_bytes() == snapshot
    else:
        with pytest.raises(ValueError, match=error):
            fetch.fetch_snapshot(path)
        assert not path.exists()


def test_file_created_during_download_is_preserved(tmp_path, monkeypatch, snapshot):
    path = tmp_path / "common_pitfalls.html"

    def concurrent_download(url, timeout):
        path.write_bytes(b"other writer")
        return BytesIO(snapshot)

    monkeypatch.setattr(fetch._opener, "open", concurrent_download)
    with pytest.raises(FileExistsError):
        fetch.fetch_snapshot(path)
    assert path.read_bytes() == b"other writer"


def test_official_http_redirect_is_upgraded_to_https():
    redirect = fetch._SnapshotRedirect()
    request = Request(fetch.SOURCE_URL)
    target = redirect.redirect_request(
        request, None, 301, "Moved", {}, "http://scikit-learn.org/1.9/common_pitfalls.html"
    )
    assert target.full_url == "https://scikit-learn.org/1.9/common_pitfalls.html"


def test_unexpected_redirect_is_rejected():
    redirect = fetch._SnapshotRedirect()
    with pytest.raises(ValueError, match="Unexpected snapshot redirect"):
        redirect.redirect_request(Request(fetch.SOURCE_URL), None, 301, "Moved", {}, "https://example.org/")
