"""python -m pinneapple_hub.zoo.web [DIR]: write the browser bundle of the zoo (default data/web/zoo)."""
from __future__ import annotations

import sys
from pathlib import Path

from . import export_bundle


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]) if argv else Path("data/web/zoo")
    res = export_bundle(out)
    for n, m in res["models"].items():
        print(f"{n}: {m['onnx']['bytes'] / 1e3:.0f} kB, parity vs PyTorch {m['onnx']['parity']:.1e}, "
              f"query checks {m['onnx']['checks_parity']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
