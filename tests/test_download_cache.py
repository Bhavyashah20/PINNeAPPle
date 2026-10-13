"""pinneapple_data.datasets.download.fetch_cached: local HTTP server, file:// URLs and a failing opener."""
import http.server
import threading
from functools import partial

import pytest

from pinneapple_data.datasets import download


@pytest.fixture
def http_dir(tmp_path):
    root = tmp_path / "www"
    root.mkdir()
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield root, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_http_download_then_cache_hit(http_dir, tmp_path, monkeypatch):
    root, base = http_dir
    (root / "a.csv").write_text("x,y\n1,2\n")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    cache = tmp_path / "cache"
    p = download.fetch_cached(f"{base}/a.csv", cache)
    assert p.read_text() == "x,y\n1,2\n" and p.suffix == ".csv"
    assert p == download.cache_path(f"{base}/a.csv", cache)
    (root / "a.csv").unlink()                      # served from the cache now
    assert download.fetch_cached(f"{base}/a.csv", cache).read_text() == "x,y\n1,2\n"
    assert [f.name for f in cache.iterdir()] == [p.name]


def test_file_url(tmp_path):
    src = tmp_path / "data.bin"
    src.write_bytes(b"\x00\x01payload")
    p = download.fetch_cached(src.as_uri(), tmp_path / "c")
    assert p.read_bytes() == b"\x00\x01payload"


def test_retries_with_exponential_backoff(tmp_path, monkeypatch):
    src = tmp_path / "ok.txt"
    src.write_text("done")
    real_open, calls, waits = download._urlopen, [], []

    def flaky(url, timeout):
        calls.append(timeout)
        if len(calls) < 3:
            raise OSError("connection reset")
        return real_open(url, timeout=timeout)

    monkeypatch.setattr(download, "_urlopen", flaky)
    monkeypatch.setattr(download, "_sleep", waits.append)
    p = download.fetch_cached(src.as_uri(), tmp_path / "c", tries=4, timeout=7, backoff=3.0)
    assert p.read_text() == "done"
    assert calls == [7, 7, 7] and waits == [1.0, 3.0]


def test_gives_up_and_leaves_no_partial_file(tmp_path, monkeypatch):
    def broken(url, timeout):
        raise OSError("down")

    waits = []
    monkeypatch.setattr(download, "_urlopen", broken)
    monkeypatch.setattr(download, "_sleep", waits.append)
    with pytest.raises(RuntimeError, match="3 attempts"):
        download.fetch_cached("https://example.invalid/x.nc", tmp_path / "c", tries=3)
    assert waits == [1.0, 2.0]
    assert list((tmp_path / "c").iterdir()) == []
