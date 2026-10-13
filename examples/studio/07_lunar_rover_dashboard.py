"""NASA VIPER crossing a lunar crater, as an animation-centred dashboard (library features only).

The animation (planetary renderer: regolith, rocks and craters, a low hard Sun with long shadows, a black sky and
the ground running out to the lunar horizon; wheels rolling and leaving 3-D ruts -- floor, walls, berms and
grouser marks deformed into the terrain mesh and shaded by the Sun, coloured by the peak normal stress they carried
through a semi-transparent map with its colour bar) sits in the middle. Around
it, charts synced to its time cursor:
- drawbar pull, driving torque and sinkage versus slip (testbed curves, each wheel's live operating point);
- per-wheel sinkage and normal load along the traverse;
- speed and slip over time;
- the pre-trained PINN of the model zoo (``rover_contact_grc1_viper``) against the run's Wong-Reece model, with its
  error at the states the wheels met.
Plus KPI tiles and a parameter panel.

    python examples/studio/07_lunar_rover_dashboard.py            # about 25 min on 4 CPU cores (zoo model download)
    python examples/studio/07_lunar_rover_dashboard.py --quick    # short run, few frames, outputs in _out/

Writes docs/assets/studio/rover-dashboard.html (interactive: scrub a chart to move the rover; "Live 3D" view),
rover-dashboard.png, rover-dashboard.mp4 (1920 x 1080), rover-dashboard.gif (1280 x 720, laid out natively at that
size so the chart text stays readable, about 6 MB for social feeds) and rover-frame.png (one rendered frame).
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
from pinneapple_simulation.terramechanics.surrogate import ZooContactSurrogate  # noqa: E402
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
    # the pre-trained wheel-contact PINN of the model zoo (downloaded once, checksum-verified)
    from pinneapple_hub import zoo
    sur = ZooContactSurrogate(zoo.load("rover_contact_grc1_viper"))
    print("PINN:", sur.meta["name"], "held-out (card):", sur.meta["test_errors"], flush=True)
    # ruts coloured by the peak normal stress they carried, semi-transparent over the 3-D relief
    frames, ks, clim = vdash.render_frames(run, site, n=24 if quick else 150, size=(1280, 720),
                                           field="normal_stress", field_alpha=0.6, field_glow=0.1,
                                           progress=lambda j, n: print(f"  frame {j}/{n}", flush=True)
                                           if j % 25 == 0 else None)
    # no edge of the ground mesh may show in any frame (the ground runs out past the lunar horizon)
    from pinneapple_simulation.vehicle import viz
    from pinneapple_tools.visualization.studio.planetary import border_visibility
    for k in ks[:: max(1, len(ks) // 10)]:
        view = viz.frame_view(run, site, int(k), field="normal_stress", clim=clim, sun_azimuth_deg=160.0,
                              half_window=(3.2, 3.4))
        vis = border_visibility(view, viz.follow_camera(run, int(k)), (1280, 720))
        assert vis["visible"] == 0, f"ground border visible at sample {k}"
    Image.fromarray(frames[len(frames) * 2 // 3]).save(os.path.join(out, "rover-frame.png"))
    db = vdash.vehicle_dashboard(
        run, frames, ks, model=wr, surrogate=sur, clim=clim, scene3d=True, terrain=site,
        field_label="Peak normal stress in the ruts [kPa]",
        pinn_note="zoo model trained on GRC-1 without slip-sinkage (n1 = 0)",
        title="NASA VIPER crossing a lunar crater",
        subtitle="Multibody dynamics, Wong-Reece wheels with slip-sinkage on GRC-1 (lunar g); 3-D ruts coloured by "
                 "normal stress; zoo PINN surrogate; low Sun (12 deg)",
        footer="PINNeAPPle: pinneapple_simulation.vehicle, .terramechanics; pinneapple_tools.visualization."
               "dashboard. VIPER geometry from public NASA figures (estimates listed in VIPER_SOURCES).")
    db.to_html(os.path.join(out, "rover-dashboard.html"), max_width=1024, quality=30)
    db.to_png(os.path.join(out, "rover-dashboard.png"), k=int(len(ks) * 0.62))
    gif = os.path.join(out, "rover-dashboard.gif")
    db.to_video(gif=gif, mp4=os.path.join(out, "rover-dashboard.mp4"), fps=12, size=(1920, 1080),
                gif_size=(1280, 720), gif_font_scale=1.0, gif_every=1 if quick else 2)
    print(f"GIF {os.path.getsize(gif) / 1e6:.1f} MB")
    print(f"done in {time.time() - t0:.0f} s -> {os.path.abspath(out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(quick=ap.parse_args().quick)
