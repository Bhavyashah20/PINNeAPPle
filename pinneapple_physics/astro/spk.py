"""A small reader for JPL SPK kernels (DAF files, segment types 2 and 3: Chebyshev position / position+velocity).

Implemented from the NAIF "DAF Required Reading" and "SPK Required Reading" documents; no SPICE dependency.
JPL planetary ephemerides (DE421, DE440, ...) are public domain.

    k = SPK("de421.bsp")
    pos, vel = k.state(target=301, center=3, et=seconds_past_J2000_TDB)   # km, km/s (ICRF)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


@dataclass
class Segment:
    target: int
    center: int
    frame: int
    kind: int
    start_et: float
    end_et: float
    init: float
    intlen: float
    rsize: int
    n: int
    coeffs: np.ndarray            # (n, rsize) view: MID, RADIUS, x..., y..., z... [, vx..., ...]

    def state(self, et):
        et = np.atleast_1d(np.asarray(et, dtype=float))
        idx = np.clip(((et - self.init) // self.intlen).astype(int), 0, self.n - 1)
        rec = self.coeffs[idx]                                   # (m, rsize)
        mid, radius = rec[:, 0], rec[:, 1]
        ncomp = 3 if self.kind == 2 else 6
        deg1 = (self.rsize - 2) // ncomp
        c = rec[:, 2:2 + ncomp * deg1].reshape(len(et), ncomp, deg1)
        tau = (et - mid) / radius
        # Chebyshev T_k(tau) and derivatives dT_k/dtau
        T = np.empty((len(et), deg1))
        dT = np.zeros((len(et), deg1))
        T[:, 0] = 1.0
        if deg1 > 1:
            T[:, 1] = tau
            dT[:, 1] = 1.0
        for k in range(2, deg1):
            T[:, k] = 2 * tau * T[:, k - 1] - T[:, k - 2]
            dT[:, k] = 2 * T[:, k - 1] + 2 * tau * dT[:, k - 1] - dT[:, k - 2]
        pos = np.einsum("mck,mk->mc", c[:, :3], T)
        if self.kind == 2:
            vel = np.einsum("mck,mk->mc", c[:, :3], dT) / radius[:, None]
        else:
            vel = np.einsum("mck,mk->mc", c[:, 3:6], T)
        return pos, vel


class SPK:
    """Memory-mapped SPK kernel. ``state(target, center, et)`` chains segments through common centres."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as fh:
            rec = fh.read(1024)
        if not rec[:7].startswith(b"DAF/SPK"):
            raise ValueError(f"{path}: not a DAF/SPK file")
        fmt = rec[88:96]
        self.endian = "<" if fmt.startswith(b"LTL") else ">"
        nd, ni = struct.unpack(self.endian + "ii", rec[8:16])
        fward, = struct.unpack(self.endian + "i", rec[76:80])
        self.data = np.memmap(path, dtype=self.endian + "f8", mode="r")
        self.segments: dict[tuple[int, int], Segment] = {}
        ss = nd + (ni + 1) // 2
        recno = fward
        while recno:
            base = (recno - 1) * 128                             # doubles per 1024-byte record
            nxt, _prev, nsum = self.data[base:base + 3]
            with open(path, "rb") as fh:
                fh.seek((recno - 1) * 1024)
                raw = fh.read(1024)
            for i in range(int(nsum)):
                off = 24 + i * ss * 8
                dc = struct.unpack(self.endian + f"{nd}d", raw[off:off + nd * 8])
                ic = struct.unpack(self.endian + f"{ni}i", raw[off + nd * 8: off + nd * 8 + ni * 4])
                target, center, frame, kind, a0, a1 = ic[:6]
                if kind not in (2, 3):
                    continue
                init, intlen, rsize, n = self.data[a1 - 4:a1]
                coeffs = self.data[a0 - 1:a0 - 1 + int(rsize) * int(n)].reshape(int(n), int(rsize))
                self.segments[(target, center)] = Segment(target, center, frame, kind, dc[0], dc[1],
                                                          float(init), float(intlen), int(rsize), int(n), coeffs)
            recno = int(nxt)
        self._parent = {t: c for (t, c) in self.segments}

    @property
    def coverage(self):
        s = next(iter(self.segments.values()))
        return s.start_et, s.end_et

    def _chain(self, body: int):
        chain = []
        while body in self._parent:
            chain.append((body, self._parent[body]))
            body = self._parent[body]
        return chain, body

    def state(self, target: int, center: int, et):
        """Position (km) and velocity (km/s) of ``target`` relative to ``center``, ICRF axes; arrays (m, 3)."""
        def to_root(body):
            chain, root = self._chain(body)
            et_ = np.atleast_1d(np.asarray(et, dtype=float))
            p = np.zeros((len(et_), 3))
            v = np.zeros((len(et_), 3))
            for t, c in chain:
                dp, dv = self.segments[(t, c)].state(et_)
                p += dp
                v += dv
            return p, v, root
        p1, v1, r1 = to_root(target)
        p0, v0, r0 = to_root(center)
        if r1 != r0:
            raise KeyError(f"no path between {target} and {center} in {self.path}")
        return p1 - p0, v1 - v0
