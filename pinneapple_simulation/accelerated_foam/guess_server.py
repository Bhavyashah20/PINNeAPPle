"""Serve initial guesses from Python to a running OpenFOAM solver (``pinneWarmStart`` with ``guess external``).

Two named pipes in ``pipe_dir`` carry the exchange. Before each solve the wrapper writes to ``request``
a message ``[1, count, nCells, nFaces, len(name)] (int64) + name + lowerAddr, upperAddr (int32) + diag,
upper, source, psi (float64)``, then reads from ``reply`` ``[status] (int64) + guess (float64)``
(status 1 = use the guess). After the solve it writes ``[2, count, nCells, 0, len(name)] + name + psi``
(no reply), which is how the guesser sees converged solutions. Any object with ``guess(system, x_prev)`` and
``update(x)`` can serve (:class:`ProjectionWarmStart`, :class:`PODWarmStart`, :class:`LearnedWarmStart`).

    with GuessServer(pipe_dir, {"p": PODWarmStart().fit(snapshots)}) as srv:
        run_case(case_dir, "icoFoam")          # fvSolution: guess external; pipeDir "<pipe_dir>";
    srv.stats                                   # per-field number of guesses served, Python time spent
"""

from __future__ import annotations

import os
import struct
import threading
import time
from pathlib import Path

import numpy as np

from .ldu import LduSystem


def _read_exact(f, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return buf


class GuessServer:
    """Answers guess requests on ``pipe_dir/request`` with ``guessers[field]`` in a background thread."""

    def __init__(self, pipe_dir, guessers: dict):
        self.pipe_dir = Path(pipe_dir)
        self.guessers = dict(guessers)
        self.stats: dict[str, dict] = {}
        self.error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._stop = False

    @property
    def request(self) -> Path:
        return self.pipe_dir / "request"

    @property
    def reply(self) -> Path:
        return self.pipe_dir / "reply"

    def __enter__(self) -> GuessServer:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        self.pipe_dir.mkdir(parents=True, exist_ok=True)
        for p in (self.request, self.reply):
            if p.exists():
                p.unlink()
            os.mkfifo(p)
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread and self._thread.is_alive():
            # unblock the open() of the request pipe with an empty "quit" message
            with open(self.request, "wb") as f:
                f.write(struct.pack("<q", 0))
            self._thread.join(timeout=10)
        for p in (self.request, self.reply):
            if p.exists():
                p.unlink()
        if self.error is not None:
            raise RuntimeError("guess server failed") from self.error

    def _loop(self) -> None:
        try:
            while not self._stop:
                with open(self.request, "rb") as f:
                    head = f.read(8)
                    if len(head) < 8:
                        continue
                    (kind,) = struct.unpack("<q", head)
                    if kind == 0:
                        return
                    count, nc, nf, nlen = struct.unpack("<4q", _read_exact(f, 32))
                    name = _read_exact(f, nlen).decode()
                    if kind == 1:
                        lo = np.frombuffer(_read_exact(f, 4 * nf), "<i4").astype(np.int64)
                        up = np.frombuffer(_read_exact(f, 4 * nf), "<i4").astype(np.int64)
                        diag, upper, src, psi = (
                            np.frombuffer(_read_exact(f, 8 * k), "<f8").copy()
                            for k in (nc, nf, nc, nc)
                        )
                    else:
                        psi = np.frombuffer(_read_exact(f, 8 * nc), "<f8").copy()
                st = self.stats.setdefault(name, {"guesses": 0, "solutions": 0, "python_time": 0.0})
                g = self.guessers.get(name)
                if kind == 2:
                    st["solutions"] += 1
                    if g is not None:
                        g.update(psi)
                    continue
                t0 = time.perf_counter()
                x = None
                if g is not None:
                    system = LduSystem(
                        lower_addr=lo,
                        upper_addr=up,
                        diag=diag,
                        upper=upper,
                        source=src,
                        psi0=psi,
                        meta={"count": count},
                    )
                    x = np.ascontiguousarray(g.guess(system, psi), dtype="<f8")
                st["python_time"] += time.perf_counter() - t0
                with open(self.reply, "wb") as r:
                    if x is None or x.shape != (nc,) or not np.all(np.isfinite(x)):
                        r.write(struct.pack("<q", 0))
                    else:
                        r.write(struct.pack("<q", 1))
                        r.write(x.tobytes())
                        st["guesses"] += 1
        except BaseException as e:  # surfaced by stop()
            self.error = e
