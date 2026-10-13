"""NASA VIPER crossing a lunar crater, as an animation-centred dashboard (library features only).

The animation (planetary renderer: regolith, rocks and craters, a low hard Sun with long shadows, a black sky,
wheels rolling and leaving ruts with grouser marks, ruts coloured by the normal stress) sits in the middle. Around
it, charts synced to its time cursor:
- drawbar pull, driving torque and sinkage versus slip (testbed curves, each wheel's live operating point);
- per-wheel sinkage and normal load along the traverse;
- speed and slip over time;
- the PINN against the Wong-Reece reference, with its error.
Plus KPI tiles and a parameter panel.

    python examples/studio/07_lunar_rover_dashboard.py            # about 25 min on 4 CPU cores
    python examples/studio/07_lunar_rover_dashboard.py --quick    # short run, few frames, outputs in _out/

Writes docs/assets/studio/rover-dashboard.html (interactive: scrub a chart to move the rover; "Live 3D" view),
rover-dashboard.png, rover-dashboard.gif, rover-dashboard.mp4 and rover-frame.png (one rendered frame).
Needs pyvista (and xvfb-run on a headless machine) and ffmpeg for the videos.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from PIL import Image  # noqa: E402

from pinneapple_simulation.terramechanics import WongReeceModel, get_soil  # noqa: E402
from pinneapple_simulation.terramechanics.surrogate import train_contact_surrogate  # noqa: E402
from pinneapple_simulation.vehicle import (  # noqa: E402
    VehicleSimulator,
    build_vehicle,
    scenarios,
    viper,
)
from pinneapple_simulation.vehicle import dashboard as vdash  # noqa: E402


def main(quick: bool = False):
    t0 = time.time()
    out = os.path.join(HERE, "_out", "rover_dashboard") if quick else os.path.join(HERE, "..", "..", "docs",
                                                                                  "assets", "studio")
    os.makedirs(out, exist_ok=True)
    spec = viper()
    soil = get_soil("GRC-1")
    site = scenarios.crater_site()
    run = VehicleSimulator(build_vehicle(spec), site, soil).run(12.0 if quick else 45.0, speed=0.2)
    print("run:", {k: round(v, 3) for k, v in run.summary().items()}, flush=True)
    wr = WongReeceModel(soil, spec.wheel)
    sur, rep = train_contact_surrogate(wr, (-0.4, 0.9), (0.002, 0.06), epochs=600 if quick else 4000)
    print("PINN held-out:", {k: round(v["rmse_over_range"], 5) for k, v in rep["test"].items()}, flush=True)
    frames, ks, clim = vdash.render_frames(run, site, n=24 if quick else 150, size=(1280, 720),
                                           progress=lambda j, n: print(f"  frame {j}/{n}", flush=True)
                                           if j % 25 == 0 else None)
    Image.fromarray(frames[len(frames) * 2 // 3]).save(os.path.join(out, "rover-frame.png"))
    db = vdash.vehicle_dashboard(
        run, frames, ks, model=wr, surrogate=sur, clim=clim, scene3d=True, terrain=site,
        title="NASA VIPER crossing a lunar crater",
        subtitle="Multibody dynamics, Wong-Reece wheels on GRC-1 (lunar g), deformable regolith; "
                 "rendered with a low Sun (12 deg) under a black sky",
        footer="PINNeAPPle: pinneapple_simulation.vehicle, .terramechanics; pinneapple_tools.visualization."
               "dashboard. VIPER geometry from public NASA figures (estimates listed in VIPER_SOURCES).")
    db.to_html(os.path.join(out, "rover-dashboard.html"), max_width=1024, quality=30)
    db.to_png(os.path.join(out, "rover-dashboard.png"), k=int(len(ks) * 0.62))
    db.to_video(gif=os.path.join(out, "rover-dashboard.gif"), mp4=os.path.join(out, "rover-dashboard.mp4"), fps=12,
                gif_size=(800, 450), gif_every=1 if quick else 3)
    print(f"done in {time.time() - t0:.0f} s -> {os.path.abspath(out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(quick=ap.parse_args().quick)
