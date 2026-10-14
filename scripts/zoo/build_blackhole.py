"""Build the black-hole accretion zoo dataset (and, when it passes the gate, the forecaster) from simulate.py runs.

    python scripts/zoo/build_blackhole.py dataset --runs RUNS --convergence examples/black_hole_weather/results/convergence.json \\
        --staging dist/zoo-staging
    python scripts/zoo/build_blackhole.py model --model runs/bh_256_res --eval runs/bh_256_res/eval --staging dist/zoo-staging

The data is too large to bundle with the package, so it is written as one file per split to
``<staging>/<name>/<split>.npz`` (Hub-only: ``publish_to_hub.py --staging`` uploads them; the publish workflow takes
them from the branch given as ``staging_ref``) and only its entry in ``pinneapple_hub/zoo/manifest.json`` changes; the
entries built by ``build_assets.py`` are left as they are.

Dataset layout (``pp.datasets.load("blackhole_riaf_density_256")``): one row per frame, 20 M apart, t in [2000, 10000] M,
for whole trajectories, so no trajectory is in two splits:

* ``train``: seeds 0-3, ``val``: seed 4, ``test``: seed 5 (alpha = 0.1); ``ood_alpha0.3``: alpha = 0.3, seed 0;
* columns ``run`` (seed), ``alpha``, ``t`` (M), ``mdot`` (accretion rate at r_in at that frame, code units) and
  ``log10_rho_q`` (uint16, the forecast window r < 126.5 M x theta in [pi/8, 7 pi/8]):
  ``log10 rho = Q_LO + q * (Q_HI - Q_LO) / 65535``, see :func:`decode`;
* ``grid_r`` / ``grid_theta``: cell centres and faces of the window;
* ``convergence``: the resolution-study scalars per grid and seed (the dataset card's verification).
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

VERSION = "1.0.0"

COLUMNS = {
    "run": ("-", "trajectory: the seed of the initial torus perturbation"),
    "alpha": ("-", "Shakura-Sunyaev viscosity parameter"),
    "t": ("GM/c^3", "time"),
    "mdot": ("code units", "accretion rate through r_in = 4 GM/c^2 at this frame (density max of the initial torus = 1)"),
    "log10_rho_q": ("-", "uint16 (frames, 192 r, 96 theta): log10 rho = -8 + q * 9 / 65535 in the forecast window "
                         "r < 126.5 GM/c^2, theta in [pi/8, 7 pi/8]"),
    "r": ("GM/c^2", "radius of the cell centres (volume weighted)"),
    "r_face_lo": ("GM/c^2", "inner radial face"), "r_face_hi": ("GM/c^2", "outer radial face"),
    "theta": ("rad", "polar angle of the cell centres"),
    "theta_face_lo": ("rad", "lower theta face"), "theta_face_hi": ("rad", "upper theta face"),
    "nr": ("-", "radial cells of the grid"), "ntheta": ("-", "theta cells of the grid"),
    "seed": ("-", "seed of the initial perturbation"),
    "mdot_mean": ("code units", "time-mean accretion rate, t in [2000, 10000] M"),
    "mdot_rms_over_mean": ("-", "rms variability of the accretion rate"),
    "mdot_psd_slope": ("-", "log-log slope of the accretion-rate power spectrum, 1/2000 to 1/50 per M"),
    "window_mass_mean": ("code units", "time-mean mass in the forecast window"),
    "persistence_mae_20M": ("-", "MAE of persistence at a 20 M lead (normalised log density, log10 rho in [-6, 0])"),
    "persistence_mae_100M": ("-", "same, 100 M lead"),
    "persistence_mae_1000M": ("-", "same, 1000 M lead"),
    "rho_slope_r10_60": ("-", "slope of the time-mean shell density, d ln rho / d ln r for r in [10, 60] M"),
    "ell_over_kepler_r10": ("-", "time-mean specific angular momentum over the Keplerian (Paczynski-Wiita) value, r = 10 M"),
    "ell_over_kepler_r30": ("-", "same at r = 30 M"),
}


def manifest_entry(arrays: dict, files: dict[str, Path], verification: str) -> dict:
    splits = {}
    descr = {"train": "seeds 0-3, alpha 0.1, t 2000-10000 M every 20 M (4 whole trajectories)",
             "val": "seed 4, alpha 0.1 (whole trajectory)", "test": "seed 5, alpha 0.1 (whole trajectory, held out)",
             "ood_alpha0.3": "alpha 0.3, seed 0: different physics, never used in training",
             "grid_r": "radial cells of the forecast window", "grid_theta": "theta cells of the forecast window",
             "convergence": "resolution study: one row per grid and seed (128x64, 181x90, 256x128, 362x181)"}
    for key, v in arrays.items():
        s = key.split("__")[0]
        splits.setdefault(s, {"n": int(len(v)), "description": descr[s]})
    cols = sorted({k.split("__", 1)[1] for k in arrays})
    columns = {c: {"unit": COLUMNS[c][0], "description": COLUMNS[c][1],
                   "dtype": str(next(v.dtype for k, v in arrays.items() if k.endswith("__" + c)))} for c in cols}
    return {
        "version": VERSION, "revision": f"v{VERSION}", "hf_repo": f"PINNeAPPle-Labs/{NAME.replace('_', '-')}",
        "repo_type": "dataset",
        "title": "Accretion onto a Schwarzschild black hole: 2.5-D viscous hydro density frames (256 x 128)",
        "description": (
            "Radiatively inefficient accretion flow grown from an equilibrium torus (Paczynski-Wiita potential, "
            "Shakura-Sunyaev alpha viscosity, gamma 5/3, r = 4-400 GM/c^2, log-spaced 256 x 128 grid), the set-up of "
            "Duarte, Nemmen & Navarro (2022) regenerated with pinneapple_physics.blackhole at the resolution chosen "
            "by a four-grid resolution study. Density in the forecaster's window every 20 GM/c^3 for whole "
            "trajectories; the training data of the black-hole weather forecaster (issue #399)."),
        "license": "cc-by-4.0",
        "source": ("examples/black_hole_weather/simulate.py --nr 256 --ntheta 128 --viscosity SS "
                   "--perturbation-kind modes --t-end 10000 --every 10 (seeds and alpha per split); "
                   "scripts/zoo/build_blackhole.py dataset"),
        "verification": verification,
        "split_rule": ("by trajectory (seed), never by time inside a trajectory: train seeds 0-3, val 4, test 5; "
                       "OOD alpha 0.3"),
        "citation": ("Duarte R., Nemmen R., Navarro J. P. (2022) MNRAS 512, 5848 (set-up and forecaster); "
                     "Almeida I., Nemmen R. (2020) MNRAS 492, 2553 (simulations); PINNeAPPle contributors (2026)."),
        "tags": ["astrophysics", "black hole", "accretion", "hydrodynamics", "forecasting"],
        "splits": splits, "columns": columns,
        "files": {fn: {"sha256": art.sha256_file(p), "size": p.stat().st_size} for fn, p in files.items()},
    }


def update_manifest(kind: str, name: str, entry: dict) -> None:
    """Insert or replace one entry, keeping everything else (and its order) as it is."""
    man = json.loads(art.MANIFEST.read_text())
    man[kind][name] = entry
    art.MANIFEST.write_text(json.dumps(man, indent=1, sort_keys=False) + "\n")


def verification_text(conv: dict) -> str:
    s = conv["scalars"]

    def row(q, fmt="{:.4g}"):
        bl = s[q]["by_level"]
        return ", ".join(f"{n}: " + fmt.format(bl[str(n)]["mean"]) + " +- " + fmt.format(bl[str(n)]["ci95"])
                         for n, _ in conv["levels"] if str(n) in bl)

    lines = ["Resolution study (scripts: examples/black_hole_weather/convergence.py): grids "
             + ", ".join(f"{a}x{b}" for a, b in conv["levels"]) + f" (ratio {conv['ratio']:.3f}), seeds "
             + ", ".join(map(str, conv["seeds"])) + ", same seeded perturbation on every grid, time averages over "
             f"t = {conv['window_M'][0]:.0f}-{conv['window_M'][1]:.0f} M with 95 % intervals (batch means and seeds).",
             "Mean accretion rate by grid (nr): " + row("mdot_mean") + ".",
             "Persistence MAE at 100 M (the forecaster's baseline): " + row("persistence_mae_100M") + "."]
    for v in conv.get("profiles", {}).values():
        lines.append(f"{v['label']}: relative L1 change between consecutive grids "
                     + ", ".join(f"{d:.3g}" for d in v["rel_L1_diff_consecutive"])
                     + (f", observed order {v['observed_order']:.2f}" if v.get("observed_order") else "")
                     + (f", GCI {100 * v['gci_fine']:.2g} %" if v.get("gci_fine") else "") + ".")
    if conv.get("verdict"):
        lines.append(conv["verdict"])
    lines.append("Solver checks (tests/test_blackhole_weather.py): mass and angular momentum conserved to 1e-16, "
                 "Bondi accretion steady within 0.5 %.")
    return " ".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--runs", type=Path, required=True)
    d.add_argument("--convergence", type=Path, required=True)
    d.add_argument("--staging", type=Path, default=ROOT / "dist/zoo-staging")
    d.add_argument("--t0", type=float, default=T0, help="(tests only) first frame time")
    d.add_argument("--t1", type=float, default=T1, help="(tests only) last frame time")
    d.add_argument("--no-manifest", action="store_true", help="(tests only) do not touch manifest.json")
    a = ap.parse_args(argv)
    if a.cmd == "dataset":
        conv = json.loads(a.convergence.read_text())
        arrays = build_arrays(a.runs, conv, t0=a.t0, t1=a.t1)
        files = write_split_files(arrays, a.staging / NAME)
        if not a.no_manifest:
            update_manifest("datasets", NAME, manifest_entry(arrays, files, verification_text(conv)))
        for p in files.values():
            print(f"{p}: {p.stat().st_size / 1e6:.1f} MB, sha256 {art.sha256_file(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
