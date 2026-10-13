"""Pack each case folder into <case>/<case>.zip (everything in the folder except the zip itself and caches).

    python examples/handoff_cases/make_zips.py [airfoil_ld free_surface_trough]

Entries are stored under <case>/..., sorted, with a fixed timestamp, so the archive changes only when a file does.
"""
from __future__ import annotations

import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = ("airfoil_ld", "free_surface_trough")
LIMIT_MB = 20.0


def pack(case: str) -> tuple[str, int]:
    root = os.path.join(HERE, case)
    out = os.path.join(root, f"{case}.zip")
    files = []
    for d, dirs, names in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for n in sorted(names):
            p = os.path.join(d, n)
            if p != out and not n.endswith(".pyc"):
                files.append(p)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            info = zipfile.ZipInfo(os.path.join(case, os.path.relpath(p, root)), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if os.access(p, os.X_OK) else 0o644) << 16
            with open(p, "rb") as f:
                z.writestr(info, f.read(), compresslevel=9)
    return out, os.path.getsize(out)


def main():
    for case in sys.argv[1:] or CASES:
        out, size = pack(case)
        mb = size / 1e6
        print(f"{os.path.relpath(out, HERE)}: {mb:.2f} MB")
        if mb >= LIMIT_MB:
            raise SystemExit(f"{out} is {mb:.1f} MB, over the {LIMIT_MB} MB limit")


if __name__ == "__main__":
    main()
