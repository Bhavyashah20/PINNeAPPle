"""Generate the files of the Hub-only black-hole accretion dataset (``blackhole_riaf_density_256``).

    python examples/black_hole_weather/simulate.py ...           # the runs listed in pinneapple_physics.blackhole.dataset.SPLITS
    python examples/black_hole_weather/convergence.py ... --out conv.json
    python scripts/zoo/build_blackhole.py --runs RUNS --convergence conv.json --out STAGING
    python scripts/zoo/build_assets.py --hub-only STAGING        # manifest entry from the catalog + the files

Writes ``STAGING/blackhole_riaf_density_256/<split>.npz`` (one file per split; a user downloads only the split they
load) and ``meta.json``. The data are built by ``pinneapple_physics.blackhole.dataset.build_arrays``; the dataset card
(title, provenance, the resolution study as verification) is the catalog entry in ``pinneapple_hub/zoo/_catalog.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pinneapple_hub.zoo import _artifacts as art  # noqa: E402
from pinneapple_physics.blackhole.dataset import NAME, T0, T1, build_arrays, write_split_files  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="folder of the simulate.py runs")
    ap.add_argument("--convergence", type=Path, required=True, help="report of examples/black_hole_weather/convergence.py")
    ap.add_argument("--out", type=Path, default=ROOT / "dist/zoo-staging")
    ap.add_argument("--t0", type=float, default=T0, help="(tests only) first frame time")
    ap.add_argument("--t1", type=float, default=T1, help="(tests only) last frame time")
    a = ap.parse_args(argv)
    arrays = build_arrays(a.runs, json.loads(a.convergence.read_text()), t0=a.t0, t1=a.t1)
    files = write_split_files(arrays, a.out / NAME)
    rows = {fn[:-4]: int(len(next(v for k, v in arrays.items() if k.startswith(fn[:-4] + "__")))) for fn in files}
    (a.out / NAME / "meta.json").write_text(json.dumps({"files": sorted(files), "rows": rows}, indent=1) + "\n")
    for p in files.values():
        print(f"{p}: {p.stat().st_size / 1e6:.1f} MB, sha256 {art.sha256_file(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
