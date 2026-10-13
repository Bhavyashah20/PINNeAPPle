"""Blender Cycles renders of the airliners run in OpenFOAM 3D, for the app's gallery and for posts.

    python apps/aero_optimizer/tools/render_airliner.py [--samples 128] [--width 1920 --height 1080] [--only balanced]

For every design in model/verification3d.json: in flight (paint), skin pressure Cp, skin friction Cf, and the
streamlines coloured by speed. CFD colours use the jet scale (blue low, red high) with a colour bar burnt into the
image. Output: model/renders/<label>_<kind>.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from pinneapple_design.aero.airframe import to_glb  # noqa: E402
from pinneapple_design.aero.airliner3d import airliner_from  # noqa: E402
from pinneapple_design.aero.case3d import surface_scalars  # noqa: E402
from pinneapple_design.aero.render_blender import render  # noqa: E402
from pinneapple_tools.visualization.studio.colormap import colorbar  # noqa: E402

MODEL = os.path.join(ROOT, "apps", "aero_optimizer", "model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--only", default=None, help="substring of the design label")
    ap.add_argument("--kinds", default="flight,cp,cf,lines")
    a = ap.parse_args()
    V = json.load(open(os.path.join(MODEL, "verification3d.json")))
    out_dir = os.path.join(MODEL, "renders")
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp()
    for d in V["designs"]:
        if a.only and a.only.lower() not in d["label"].lower():
            continue
        x = np.array(d["x"])
        af = airliner_from(x)
        parts = af.build("high")
        z = np.load(os.path.join(MODEL, "cfd3d", f"{d['id']}.npz"))
        fields = {f"_{k.upper()}": surface_scalars(af, parts, {"xyz": z["xyz"], k: z[k]}, key=k) for k in ("cp", "cf")}
        glb = os.path.join(tmp, f"{d['id']}.glb")
        open(glb, "wb").write(to_glb(parts, fields=fields))
        lines = os.path.join(tmp, f"{d['id']}_lines.npz")
        np.savez(lines, lines=z["lines"], line_speed=z["line_speed"])
        tag = d["label"].lower().replace(" ", "_")
        size = (a.width, a.height)
        for kind in a.kinds.split(","):
            out = os.path.join(out_dir, f"{tag}_{kind}.jpg")
            if kind == "flight":
                render(glb, out, "flight_front", a.samples, size, ground=False, sun_elevation=32, distance=1.1)
            elif kind in ("cp", "cf"):
                lo, hi = d["ranges"][kind]
                render(glb, out, "cfd", a.samples, size, field=kind, field_range=(lo, hi), studio=True, distance=1.08)
                if kind == "cp":
                    colorbar(out, "Pressure coefficient Cp (OpenFOAM)", lo, hi, 2, "suction", "stagnation")
                else:
                    colorbar(out, "Skin friction coefficient Cf (OpenFOAM)", lo, hi, 4, "low shear", "high shear")
            elif kind == "lines":
                lo, hi = float(np.percentile(z["line_speed"], 2)), float(np.percentile(z["line_speed"], 98))
                render(glb, out, "cfd", a.samples, size, lines_npz=lines, line_range=(lo, hi), studio=True, distance=1.08)
                colorbar(out, "Speed along the streamlines |U|/U∞ (OpenFOAM)", lo, hi, 2, "slower", "faster")
            print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)                                   # bpy can crash while tearing down at interpreter exit
