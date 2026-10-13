"""pp.cfd.FreeSurfaceFlow (interFoam VOF case writer, line sampling) and the 2-D runner section."""
import math
import os
import re

import numpy as np
import pytest

from pinneapple_design.geometry.trough import TroughSection
from pinneapple_simulation.numerical_solvers.free_surface import (
    FreeSurfaceFlow,
    Phase,
    SampleLine,
    cell_centres,
    split_surface,
)


def test_split_surface_by_boxes():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], float)
    F = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 4], [0, 2, 3]])
    parts = split_surface(V, F, {"top": ((-1, -1, 0.3), (2, 2, 2))})
    assert set(parts) == {"top", "walls"}
    assert sum(len(f) for _, f in parts.values()) == len(F)
    Vt, Ft = parts["top"]
    assert len(Ft) == 3 and np.all(Vt[Ft].mean(1)[:, 2] >= 0.3)


def test_trough_frame():
    sec = TroughSection(slope_deg=2.0)
    g = np.array(sec.gravity(9.8))
    assert np.linalg.norm(g) == pytest.approx(9.8) and g[0] > 0          # + slope: gravity pulls toward the weir
    up = -g / 9.8
    crest = np.array([sec.length, 0.0, sec.weir_height])
    assert crest @ up == pytest.approx(sec.level())                      # the bath surface passes the weir crest
    assert sec.depth_at(sec.length) == pytest.approx(sec.weir_height)
    assert sec.depth_at(0.0) == pytest.approx(0.5 - 8 * math.tan(math.radians(2)))
    v = sec.to_mesh((2.0, 0.0, -5.0))
    assert np.linalg.norm(v) == pytest.approx(math.hypot(2, 5))
    assert np.array(v) @ up == pytest.approx(np.array([2.0, 0, -5.0]) @ [0, 0, 1])   # vertical speed preserved
    with pytest.raises(ValueError):
        TroughSection(slope_deg=4.0).blockmesh()


def _fake_mesh(tmp_path):
    d = tmp_path / "mesh" / "constant" / "polyMesh"
    d.mkdir(parents=True)
    (d / "boundary").write_text("FoamFile { version 2.0; class polyBoundaryMesh; object boundary; }\n3\n(\n"
                                "    floor { type wall; nFaces 1; startFace 0; }\n"
                                "    jet { type patch; nFaces 1; startFace 1; }\n"
                                "    top { type patch; nFaces 1; startFace 2; }\n"
                                "    frontAndBack { type empty; nFaces 2; startFace 3; }\n)\n")
    return str(tmp_path / "mesh")


def test_write_case(tmp_path):
    with pytest.raises(ValueError):
        FreeSurfaceFlow(mesh="x", roles={"jet": "inlet"})                 # inlet without a velocity
    sec = TroughSection(slope_deg=-2.0)
    f = FreeSurfaceFlow(mesh=_fake_mesh(tmp_path), two_d=True, roles={"jet": "inlet", "top": "atmosphere"},
                        inlet_velocity={"jet": sec.to_mesh((2.0, 0.0, -5.0))}, level=sec.level(),
                        gravity=sec.gravity(9.8), liquid=Phase("metal", 7200.0, 8.472e-6), sigma=1.8,
                        lines=[SampleLine("L", (0, 0, 0.05), (8, 0, 0.05))])
    info = f.write(str(tmp_path / "case"))
    assert info["patches"] == {"floor": "wall", "jet": "inlet", "top": "atmosphere"}     # empty patch skipped
    c = tmp_path / "case"
    U = (c / "0" / "U").read_text()
    assert "noSlip" in U and "pressureInletOutletVelocity" in U and "empty" in U
    a = (c / "0" / "alpha.metal").read_text()
    assert re.search(r"jet\s*\{\s*type fixedValue;\s*value uniform 1;", a)
    k = (c / "0" / "k").read_text()
    kin = float(re.search(r"internalField uniform ([^;]+);", k).group(1))
    assert kin == pytest.approx(1.5 * (math.hypot(2, 5) * 0.01) ** 2)
    assert "rotatedBoxToCell" in (c / "system" / "setFieldsDict").read_text()           # tilted gravity
    assert float(re.search(r"value ([^;]+);", (c / "constant" / "hRef").read_text()).group(1)) == pytest.approx(
        sec.level())
    assert "phases (metal air);" in (c / "constant" / "transportProperties").read_text()
    fv = (c / "system" / "fvSolution").read_text()
    assert re.search(r"p_rgh \{ solver PCG; preconditioner DIC; tolerance 1e-07; relTol 0.05; \}", fv)
    assert "libs" not in (c / "system" / "controlDict").read_text()


def test_write_case_with_warm_started_pressure(tmp_path):
    from pinneapple_simulation.accelerated_foam.case import warm_solver

    f = FreeSurfaceFlow(mesh=_fake_mesh(tmp_path), two_d=True, roles={"top": "atmosphere"},
                        pressure_solver=warm_solver("PCG", 1e-7, 0.05, safeguard=True), libs=("/x/lib.so",))
    f.write(str(tmp_path / "case"))
    c = tmp_path / "case"
    fv = (c / "system" / "fvSolution").read_text()
    assert "p_rgh { solver pinneWarmStart; innerSolver PCG; guess projection; nSnapshots 8;" in fv
    assert "safeguard true;" in fv and "p_rghFinal { $p_rgh; relTol 0; }" in fv
    assert 'libs ("/x/lib.so");' in (c / "system" / "controlDict").read_text()


def _foam():
    from pinneapple_simulation.numerical_solvers.external_flow import openfoam_available

    if not openfoam_available():
        pytest.skip("OpenFOAM not available")


def test_short_run_holds_level_and_samples(tmp_path):
    _foam()
    sec = TroughSection(length=4.0, inlet_x=0.8, slope_deg=1.0, cell=0.05)
    f = FreeSurfaceFlow(mesh=sec.write_mesh(str(tmp_path / "mesh")), two_d=True,
                        roles={"inlet": "inlet", "outlet": "outlet", "atmosphere": "atmosphere"},
                        inlet_velocity={"inlet": sec.to_mesh((1.0, 0.0, -3.0))}, level=sec.level(),
                        gravity=sec.gravity(9.8), liquid=Phase("metal", 7200.0, 8.472e-6), sigma=1.8,
                        end_time=0.3, sample_every=0.1, keep_every=0.3,
                        lines=[SampleLine("floor", (0.0, 0.025, 0.05), (4.0, 0.025, 0.05), 41)])
    case = str(tmp_path / "case")
    f.write(case, procs=1)
    cc = cell_centres(os.path.join(case, "constant", "polyMesh"))
    f.run(case, procs=1, log=lambda s: None)
    prof = f.profiles(case)["floor"]
    assert np.allclose(prof["times"], [0.1, 0.2, 0.3])
    assert prof["k"].shape == (3, 41) and np.all(prof["k"] >= 0)
    assert prof["alpha.metal"][0].min() > 0.99                     # 5 cm above the floor stays in the bath
    assert not os.path.exists(os.path.join(case, "0.1")) and os.path.exists(os.path.join(case, "0.3"))
    assert len(cc) > 1000
