"""Wind on the CAARC standard tall building with OpenFOAM, through the generic STL path of ``pp.cfd``.

    python examples/geometry_library/01_caarc_building_wind_openfoam.py [coarse|medium|fine] [processes]

Loads the pre-generated light STL of ``caarc_building`` (``pp.geometries.load``: repository copy, Hub or rebuilt),
builds the isolated building (no surroundings) for the CFD, and runs steady RANS (k-omega SST) at 10 m/s on a fixed
ground with ``pp.cfd.ExternalFlow``: drag (along-wind) coefficient on the frontal area, skin Cp, wake slices.
Wind-tunnel references for the isolated CAARC building at 0 deg: mean along-wind force coefficient about 1.9-2.1
(Melbourne 1980). Without OpenFOAM on the machine the case is written and the commands to run it are printed.
"""
import os
import shutil
import sys

import pinneapple as pp

HERE = os.path.dirname(os.path.abspath(__file__))
res_name = sys.argv[1] if len(sys.argv) > 1 else "coarse"
procs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
out = os.path.join(HERE, "_out", "caarc")
os.makedirs(out, exist_ok=True)

print("pre-generated file:", pp.geometries.load("caarc_building", format="stl"))
info = pp.geometries.info("caarc_building")
print(info["title"], "-", "; ".join(s["ref"] for s in info["sources"][:1]))
g = pp.geometries.build("caarc_building", surroundings=False)
stl = g.export(os.path.join(out, "caarc_isolated.stl"))          # the generic STL path
d = g.dimensions()
flow = pp.cfd.ExternalFlow(stl, speed=10.0, ground=0.0, ground_moving=False, ref_area=d["width_y"] * d["height_z"],
                           ref_length=d["height_z"], resolution=res_name, iterations=500,
                           title="CAARC building, 10 m/s")
case = os.path.join(out, f"case_{res_name}")
if shutil.which("simpleFoam") or os.path.exists(os.environ.get("FOAM_BASHRC", "/usr/share/openfoam/etc/bashrc")):
    res = flow.solve(case, procs=procs)
    c = res.coefficients
    print(f"CD (along-wind, frontal area) {c['CD']:.2f}, CS {c.get('CS', 0):.2f}, {c['cells']:,} cells "
          "(wind tunnel: about 2.0)")
    pp.viz.web_viewer(res.to_scene(), os.path.join(out, "viewer"))
else:
    flow.write(case, procs=procs)
    print(f"OpenFOAM not found: wrote the case to {case}. Run it with OpenFOAM installed:\n"
          f"  FOAM_BASHRC=/path/to/openfoam/etc/bashrc python {sys.argv[0]} {res_name} {procs}")
