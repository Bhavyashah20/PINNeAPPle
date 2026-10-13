"""Six-wheel rocker-bogie rover on the Moon with Bekker-Wong wheels (GRC-1 regolith, g = 1.62 m/s^2).

Runs the scenarios and the verification of ``pinneapple_simulation.rover`` and writes:

    outputs/rover3d/                 time series (CSV), the 3D viewer folder, the trained surrogate files
    docs/assets/lunar_rover/         PNG plots, GIF + MP4 animations, rover.glb, results.json
                                     (outputs/rover3d/media/ with --quick)

Scenarios
  A  level traverse at 0.05, 0.10 and 0.20 m/s (speed held, the drive finds its slip)
  B  drawbar-pull test on level ground at slip set points 0.05 ... 0.7
  C  climbing: steady state on uniform grades, the maximum grade (bisection) and a ramp steepening from 0 to 36 deg
     driven until traction fails
  D  crossing a crater (rim over the path, part of the path inside the bowl)
  E  the same rover with a 20 kg payload (level traverse, maximum grade)
Verification
  single wheel vs standalone solver, level-ground sinkage vs Bekker's closed form, force and moment balance,
  maximum grade vs "drawbar pull = grade resistance", the PINN surrogate vs the numerical law along the runs.

    python examples/use_cases/terramechanics/lunar_rover_3d.py            # about 10 min on a laptop CPU
    python examples/use_cases/terramechanics/lunar_rover_3d.py --quick    # fewer points, no surrogate training

Scenario inputs (speeds, slips, grades, crater size, payload) are choices made for this example, not data.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from pinneapple_simulation.particle_dynamics.terramechanics import SoilParams  # noqa: E402
from pinneapple_simulation.rover import (  # noqa: E402  # noqa: E402
    BekkerWongContact,
    Crater,
    LunarTerrain,
    RoverParams,
    RoverSimulator,
    SurrogateContact,
    checks,
    export_rover,
    random_rocks,
    viz,
)

OUT = HERE / "outputs" / "rover3d"
MEDIA = ROOT / "docs" / "assets" / "lunar_rover"
STEP = 0.05          # path advance per step [m]


def log(*a):
    print(*a, flush=True)


def run(sim, distance, v, **kw):
    return sim.run(distance=distance, v=v, dt=STEP / v, **kw)


# ---------------------------------------------------------------------------------------------- surrogates
def load_surrogates(train: bool) -> dict:
    """The use case's PINN (``terramechanics_rover_pinn.py``) on its own training box, and the same network and
    losses retrained on a box sized for this rover's wheel loads (sinkage 2-25 mm)."""
    sys.path.insert(0, str(HERE))
    out = {}
    specs = {"pinn_use_case_box": ("terramechanics_surrogate", "surrogate_metadata.json", (0.002, 0.058)),
             "pinn_rover_box": ("terramechanics_surrogate_rover_box", "surrogate_rover_box_metadata.json",
                                (0.002, 0.025))}
    tm_out = HERE / "outputs"
    for key, (stem, meta, zr) in specs.items():
        pt, js = tm_out / f"{stem}.pt", tm_out / meta
        if not pt.exists() and train:
            import terramechanics_rover_pinn as tp
            log(f"training the PINN surrogate ({key}, sinkage box {zr}) with terramechanics_rover_pinn.train ...")
            model, nx, ny, _, _ = tp.train(slip_range=(0.0, 0.75), sinkage_range=zr)
            tp.export_surrogate(model, nx, ny, stem=stem, meta_name=meta)
        if pt.exists():
            out[key] = SurrogateContact.from_torchscript(str(pt), str(js))
    return out


# ---------------------------------------------------------------------------------------------- main
def main(quick: bool = False, train_surrogate: bool = True):
    t_start = time.time()
    media = OUT / "media" if quick else MEDIA          # a quick run never overwrites the published figures
    OUT.mkdir(parents=True, exist_ok=True)
    media.mkdir(parents=True, exist_ok=True)
    rover = RoverParams(n_grousers=16)
    soil = SoilParams()                     # GRC-1, the values of the terramechanics use case
    R = {"rover": {k: getattr(rover, k) for k in ("wheel_radius", "wheel_width", "track", "wheelbase",
                                                  "pivot_height", "bogie_drop", "body_mass")},
         "soil": {k: getattr(soil, k) for k in ("c", "phi_deg", "K", "k_c", "k_phi", "n", "g")}}
    R["rover"]["inertia_kg_m2"] = np.diag(rover.body_inertia()).round(4).tolist()
    export_rover(rover, str(media / "rover.glb"))
    export_rover(rover, str(OUT / "rover.stl"))
    flat = LunarTerrain(soil=soil)
    sim = RoverSimulator(rover, flat)
    W = rover.mass * sim.g

    # ---------------------------------------------------------------- A: level traverse at three speeds
    log("A  level traverse")
    A, trajA = {}, {}
    for v in (0.05, 0.10, 0.20):
        tr = run(sim, 1.0 if quick else 2.0, v)
        tr.to_csv(str(OUT / f"A_flat_v{v:.2f}.csv"))
        st = tr.states[-1]
        A[f"{v:.2f}"] = {"energy_per_metre_J_m": tr.energy_per_metre(), "slip": float(st.s_cmd),
                         "sinkage_mm": (1e3 * st.sinkage).round(3).tolist(), "load_N": st.Fz.round(3).tolist(),
                         "torque_Nm": st.My.round(4).tolist(), "power_W": float(st.power.sum()),
                         "max_residual": tr.summary()["max_residual"]}
        trajA[f"v = {v:.2f} m/s"] = tr
        log(f"   v={v:.2f}  E/m={tr.energy_per_metre():.3f} J/m  slip={st.s_cmd:.4f}")
    R["A_level_traverse"] = A
    viz.plot_wheel_series(trajA["v = 0.10 m/s"], str(media / "A_level_wheels.png"),
                          "Level traverse at 0.10 m/s on GRC-1 (per wheel)")

    # ---------------------------------------------------------------- B: drawbar-pull test
    log("B  drawbar pull vs slip")
    B = []
    for s in ((0.05, 0.2, 0.5) if quick else (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)):
        tr = run(sim, 0.3, 0.1, mode="slip", s_cmd=s)
        st = tr.states[-1]
        B.append({"slip": s, "drawbar_pull_N": st.pull, "dp_over_weight": st.pull / W,
                  "sinkage_mean_mm": float(1e3 * st.sinkage.mean()), "power_W": float(st.power.sum()),
                  "single_wheel": {k: v for k, v in checks.single_wheel_check(st, rover, flat.soils()).items()
                                   if k != "rows"},
                  "balance": checks.balance_check(st, rover)})
        log(f"   s={s:.2f}  DP/W={st.pull / W:.4f}")
    R["B_drawbar_pull"] = B

    # ---------------------------------------------------------------- C: climbing
    log("C  climbing")
    grades = (0, 10, 20, 25) if quick else (0, 5, 10, 15, 20, 22.5, 25, 27.5)
    C_rows = []
    for a in grades:
        st = sim.steady_on_slope(a)
        C_rows.append({"grade_deg": a, "converged": st.converged, "slip": float(st.s_cmd),
                       "load_N": st.Fz.round(3).tolist(), "sinkage_mm": (1e3 * st.sinkage).round(3).tolist(),
                       "power_W": float(st.power.sum()), "energy_per_metre_J_m": float(st.power.sum() / st.v),
                       "balance": checks.balance_check(st, rover)})
        log(f"   grade {a:5.1f}  slip {st.s_cmd:.3f}  E/m {st.power.sum() / st.v:.2f}")
    a_max = sim.max_slope(tol=0.1 if quick else 0.05)
    est = checks.dp_grade_estimate(rover, soil, s_limit=sim.s_limit, g=sim.g)
    log(f"   max grade (full model) {a_max:.2f} deg, DP = grade resistance estimate {est['max_slope_deg']:.2f} deg")
    ramp = LunarTerrain(ramp=(1.0, 7.0, 0.0, 36.0), soil=soil)
    tr_ramp = run(RoverSimulator(rover, ramp), 8.0, 0.1)
    k_fail = tr_ramp.failed_at
    last_ok = tr_ramp.states[(k_fail - 1) if k_fail else -1]
    grade_fail = float(ramp.slope_along(last_ok.pose.x, last_ok.pose.y))
    contact_grades = [float(ramp.slope_along(p[0], p[1])) for p in last_ok.contact_point]
    log(f"   ramp: traction lost at x={last_ok.pose.x:.2f} m, grade under the body {grade_fail:.2f} deg")
    tr_ramp.to_csv(str(OUT / "C_ramp.csv"))
    R["C_climbing"] = {"steady": C_rows, "s_limit": sim.s_limit, "max_grade_full_model_deg": a_max,
                       "max_grade_dp_estimate_deg": est["max_slope_deg"],
                       "ramp": {"failed_at_step": k_fail, "x_at_failure_m": last_ok.pose.x,
                                "grade_under_body_deg": grade_fail,
                                "grade_under_wheels_deg": [round(g, 2) for g in contact_grades],
                                "slip_last_ok": float(last_ok.s_cmd)}}
    viz.plot_wheel_series(tr_ramp, str(media / "C_ramp_wheels.png"),
                          "Climbing a ramp steepening from 0 to 36 deg until traction fails")

    # ---------------------------------------------------------------- D: crater crossing
    log("D  crater crossing")
    rng = np.random.default_rng(7)
    craters = [Crater(4.0, 0.6, 4.0, rim_ratio=0.02, depth_ratio=0.1),         # degraded crater on the path
               Crater(5.0, -4.2, 3.0, rim_ratio=0.02)]                          # fresh crater (Pike d/D), scenery
    on_path = lambda x, y: abs(y) < 1.0 or math.hypot(x - 5.0, y + 4.2) < 1.8  # noqa: E731
    rocks = random_rocks(rng, 25, (-1, 9), (-6, 3), (0.05, 0.2), (0.4, 0.8), keep_out=on_path)
    crater_ter = LunarTerrain(craters=craters, rocks=rocks, soil=soil)
    tr_cr = run(RoverSimulator(rover, crater_ter), 8.0, 0.1)
    tr_cr.to_csv(str(OUT / "D_crater.csv"))
    R["D_crater"] = tr_cr.summary() | {"min_pitch_deg": float(np.degrees(tr_cr.pose_array("pitch")).min())}
    log(f"   {R['D_crater']}")
    viz.plot_wheel_series(tr_cr, str(media / "D_crater_wheels.png"),
                          "Crossing a crater (D = 4 m, depth 0.1 D, rim 0.02 D) off-centre")

    # ---------------------------------------------------------------- E: payload
    log("E  payload")
    heavy = replace(rover, payload_mass=20.0)
    sim_h = RoverSimulator(heavy, flat)
    tr_h = run(sim_h, 1.0 if quick else 2.0, 0.1)
    a_max_h = sim_h.max_slope(tol=0.1 if quick else 0.05)
    est_h = checks.dp_grade_estimate(heavy, soil, s_limit=sim.s_limit, g=sim.g)
    tr_hc = run(RoverSimulator(heavy, crater_ter), 8.0, 0.1)
    R["E_payload"] = {"payload_kg": 20.0, "energy_per_metre_J_m": tr_h.energy_per_metre(),
                      "slip": float(tr_h.states[-1].s_cmd),
                      "sinkage_mm": (1e3 * tr_h.states[-1].sinkage).round(3).tolist(),
                      "max_grade_full_model_deg": a_max_h, "max_grade_dp_estimate_deg": est_h["max_slope_deg"],
                      "crater": tr_hc.summary()}
    log(f"   {R['E_payload']}")
    viz.plot_rover_series({"level 0.10 m/s": trajA["v = 0.10 m/s"], "ramp": tr_ramp, "crater": tr_cr,
                           "crater + 20 kg": tr_hc}, str(media / "rover_series.png"),
                          "Rover attitude and drive energy")

    # ---------------------------------------------------------------- verification
    log("verification")
    V = {}
    st_sink = checks.static_sinkage_check(rover, soil)
    V["level_sinkage_vs_bekker"] = {k: v for k, v in st_sink.items() if k != "state"}
    V["single_wheel_vs_standalone"] = {f"slip {b['slip']}": b["single_wheel"] for b in B}
    bal = [checks.balance_check(s, rover) for tr in (*trajA.values(), tr_ramp, tr_cr, tr_hc) for s in tr.states
           if s.converged]
    V["balance_all_converged_steps"] = {k: max(b[k] for b in bal) for k in bal[0]} | {"n_states": len(bal)}
    V["max_grade"] = {"full_model_deg": a_max, "dp_estimate_deg": est["max_slope_deg"],
                      "payload_full_model_deg": a_max_h, "payload_dp_estimate_deg": est_h["max_slope_deg"]}
    surr = load_surrogates(train_surrogate and not quick)
    num = BekkerWongContact(rover.wheel_radius, rover.wheel_width)
    runs = {"A_level": list(trajA.values()), "C_ramp": [tr_ramp], "D_crater": [tr_cr]}
    V["surrogate"] = {}
    for key, sur in surr.items():
        V["surrogate"][key] = {"training_box": {"slip": sur.slip_range, "sinkage_m": sur.sinkage_range}}
        for rname, trs in runs.items():
            states = [s for tr in trs for s in tr.states if s.converged]
            V["surrogate"][key][rname] = checks.surrogate_vs_numerical(states, sur, num, flat.soils())
        t0 = time.time()
        tr_s = run(RoverSimulator(rover, LunarTerrain(slope_deg=10, soil=soil), contact=sur), 1.0, 0.1,
                   stop_on_failure=True)
        ref = run(RoverSimulator(rover, LunarTerrain(slope_deg=10, soil=soil)), 1.0, 0.1)
        V["surrogate"][key]["driving_10deg"] = {
            "completed": tr_s.failed_at is None, "failed_at_step": tr_s.failed_at, "wall_s": time.time() - t0,
            "energy_per_metre_J_m": tr_s.energy_per_metre() if tr_s.failed_at is None else None,
            "reference_energy_per_metre_J_m": ref.energy_per_metre()}
        log(f"   {key}: {V['surrogate'][key]['driving_10deg']}")
    R["verification"] = V

    # ---------------------------------------------------------------- visuals
    log("visuals")
    viz.animate(tr_cr, str(media / "crater_crossing.gif"), path_mp4=str(media / "crater_crossing.mp4"),
                color_by="sinkage", frames=60 if quick else 120, title="Crater crossing, wheels coloured by sinkage")
    viz.animate(tr_ramp, str(media / "ramp_climb.gif"), color_by="slip", frames=50 if quick else 90,
                title="Ramp climb until traction fails, wheels coloured by slip", clim=(0.0, 1.0),
                view=(18, -100))
    sc = viz.twin_scene(tr_cr, title="Lunar rover crossing a crater")
    sc.export(str(OUT / "viewer"))
    R["wall_time_s"] = time.time() - t_start
    with open(media / "results.json", "w") as f:
        json.dump(R, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    shutil.copy(media / "results.json", OUT / "results.json")
    log(f"done in {R['wall_time_s']:.0f} s -> {media}")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-train", action="store_true", help="skip training missing PINN surrogates")
    a = ap.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    main(quick=a.quick, train_surrogate=not a.no_train)
