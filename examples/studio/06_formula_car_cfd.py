"""A parametric Formula 1-style car in OpenFOAM, from geometry to pictures of the flow, with PINNeAPPle.

    python examples/studio/06_formula_car_cfd.py [coarse|medium|fine] [processes] [drs]

``pinneapple_design.geometry.vehicles3d.FormulaCar`` (monocoque, floor with diffuser, multi-element front and rear
wings, helmet, halo, exposed wheels; not any team's car) at 70 m/s (252 km/h) over a road moving with the flow, half
model, steady RANS (k-omega SST, wall functions). Wheels are static walls (not rotating). Prints drag, downforce,
L/D and the downforce of each component (front wing, rear wing, floor, body, wheels); writes the 3-D viewer, a glTF,
a Plotly page and a Rerun recording, and renders with PyVista: the car, the skin Cp from above and below, the
streamlines, the mid-plane Cp and the wake vorticity. ``drs`` opens the rear flap.
Needs OpenFOAM (FOAM_BASHRC pointing to its bashrc), pyvista; plotly and rerun-sdk optional.
"""
import json
import os
import sys

import pinneapple as pp
from pinneapple_design.geometry.vehicles3d import FormulaCar
from pinneapple_tools.visualization.studio.pyvista_render import render_scene
from pinneapple_tools.visualization.studio.scene import Scene

HERE = os.path.dirname(os.path.abspath(__file__))
res_name = sys.argv[1] if len(sys.argv) > 1 else "medium"
procs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
drs = len(sys.argv) > 3 and sys.argv[3] == "drs"
out = os.environ.get("F1_OUT") or os.path.join(HERE, "_out", f"formula_car{'_drs' if drs else ''}")
os.makedirs(out, exist_ok=True)

car = FormulaCar(drs_open=drs)
A = car.frontal_area()
flow = pp.cfd.ExternalFlow(car.cfd_bodies(), speed=70.0, ground=0.0, half_model=True, ref_area=A,
                           resolution=res_name, iterations=800, n_lines=60,
                           title=f"Formula car{' (DRS open)' if drs else ''}, 70 m/s")
result = flow.solve(os.path.join(out, f"case_{res_name}"), procs=procs)
c = result.coefficients
q = 0.5 * 1.225 * 70.0 ** 2
summary = {"frontal_area_m2": A, "CD": c["CD"], "CL": c["CL"], "CD_pressure": c["CD_pressure"],
           "CD_friction": c["CD_friction"], "drag_N": c["CD"] * q * A, "downforce_N": -c["CL"] * q * A,
           "L_over_D": -c["CL"] / c["CD"], "cells": c["cells"],
           "components": {k: {"CL": v["CL"], "CD": v["CD"], "downforce_N": -v["CL"] * q * A, "drag_N": v["CD"] * q * A}
                          for k, v in c["bodies"].items()}}
down = summary["downforce_N"]
print(f"frontal area {A:.2f} m2, {c['cells']:,} cells")
print(f"CD {c['CD']:.3f} (pressure {c['CD_pressure']:.3f} + friction {c['CD_friction']:.3f}), CL {c['CL']:.3f}, "
      f"L/D {summary['L_over_D']:.2f}")
print(f"at 70 m/s: drag {summary['drag_N']:.0f} N, downforce {down:.0f} N")
for k, v in summary["components"].items():
    print(f"  {k:<11s} downforce {v['downforce_N']:7.0f} N ({100 * v['downforce_N'] / down:5.1f} %), "
          f"drag {v['drag_N']:6.0f} N")
json.dump(summary, open(os.path.join(out, f"coefficients_{res_name}.json"), "w"), indent=1)

scene = Scene.from_parts(car.parts(), axes="z_up", title=flow.title)
scene = result.to_scene(scene)
scene.save(os.path.join(out, "formula_car.glb"))
pp.viz.web_viewer(scene, os.path.join(out, "viewer"))
render_scene(Scene.from_parts(car.parts(), axes="z_up"), os.path.join(out, "beauty.png"), view="iso", ground=True,
             zoom=1.6)
render_scene(scene, os.path.join(out, "cp_top.png"), field="cp", clim=(-2.0, 1.0), view="iso", zoom=1.6,
             hide=("halo",))
render_scene(scene, os.path.join(out, "cp_bottom.png"), field="cp", clim=(-2.0, 1.0), view=(-0.25, -0.35, -1.0),
             zoom=1.35)
render_scene(scene, os.path.join(out, "streamlines.png"), lines=True, view="iso", zoom=1.4)
render_scene(scene, os.path.join(out, "streamlines_side.png"), lines=True, view="side", zoom=1.4)
render_scene(scene, os.path.join(out, "midplane_cp.png"), slice_name="mid plane: Cp", view="side", zoom=1.3,
             cmap="RdBu_r", clim=(-1.5, 1.5), slice_crop=(0.12, 0.62, 0.0, 0.45))     # the car and its near wake
render_scene(scene, os.path.join(out, "wake_vorticity.png"), slice_name="wake: vorticity", view="back", zoom=1.5,
             cmap="RdBu_r", clim=(-6.0, 6.0), slice_crop=(0.27, 0.73, 0.0, 0.55))     # |y| < 1.8 m, z < 1.6 m
try:
    from pinneapple_tools.visualization.integrations.plotly_export import scene_html

    scene_html(scene, os.path.join(out, "formula_car_cp.html"), field="cp", title=flow.title, include_plotlyjs="cdn")
except ImportError:
    pass
try:
    from pinneapple_tools.visualization.integrations.rerun_bridge import RerunLogger

    log = RerunLogger("formula-car", path=os.path.join(out, "formula_car.rrd"))
    log.scene(scene, field="cp")
    log.text("summary", "```json\n" + json.dumps(summary, indent=1) + "\n```")
    log.close()
except ImportError:
    pass
print("wrote", out)
os._exit(0)
