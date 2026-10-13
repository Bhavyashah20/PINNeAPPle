"""Download a URL once into a local cache, with retries (standard library only).

    >>> from pinneapple_data.datasets.download import fetch_cached
    >>> path = fetch_cached("https://example.org/catalog.csv", "data/.cache")
    >>> text = path.read_text()

The cache file is named after the MD5 hash of the URL (plus the URL's file extension), so the same URL is fetched
once and different URLs never collide. The body is written to a temporary file and renamed when complete: an
interrupted download never leaves a truncated file that a later call would mistake for a cached one.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

__all__ = ["fetch_cached", "cache_path"]

# Indirections so tests can replace the network and the clock.
_urlopen = urllib.request.urlopen
_sleep = time.sleep


def cache_path(url: str, cache_dir: str | Path) -> Path:
    """The file :func:`fetch_cached` uses for ``url``: ``<cache_dir>/<md5(url)><ext>``."""
    ext = Path(urlparse(url).path).suffix
    if not re.fullmatch(r"\.[A-Za-z0-9]{1,8}", ext):
        ext = ""
    return Path(cache_dir) / (hashlib.md5(url.encode("utf-8"), usedforsecurity=False).hexdigest() + ext)


def fetch_cached(url: str, cache_dir: str | Path, *, tries: int = 4, timeout: float = 60,
                 backoff: float = 2.0) -> Path:
    """Return a local copy of ``url``, downloading it only if it is not cached yet.

    Parameters
    ----------
    url : any URL ``urllib.request.urlopen`` accepts (``https://``, ``http://``, ``file://``).
    cache_dir : directory of the cache (created if missing).
    tries : number of attempts before giving up.
    timeout : socket timeout per attempt, in seconds.
    backoff : the wait before attempt ``k + 1`` (``k = 1, 2, ...``) is ``backoff ** (k - 1)`` seconds
        (1, 2, 4, ... for the default), i.e. exponential backoff.

    Returns
    -------
    Path of the cached file.

    Raises
    ------
    RuntimeError if every attempt fails (the last error is chained).
    """
    if tries < 1:
        raise ValueError(f"tries must be >= 1, got {tries}")
    out = cache_path(url, cache_dir)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.part{os.getpid()}")
    last: Exception | None = None
    for k in range(tries):
        if k:
            _sleep(backoff ** (k - 1))
        try:
            with _urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, out)
            return out
        except Exception as e:  # noqa: BLE001  network errors come in many types; retry them all
            last = e
            print(f"fetch_cached: attempt {k + 1}/{tries} failed for {url[:120]}: {e}", file=sys.stderr)
        finally:
            if tmp.exists():
                tmp.unlink()
    raise RuntimeError(f"fetch_cached: {tries} attempts failed for {url}") from last
