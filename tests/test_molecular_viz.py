"""Molecular visualisation: PDB reading, fcc slabs, orientation on a surface, analysis, renders, glTF, MP4."""
import io
import math

import numpy as np
import pytest

from pinneapple_tools.visualization.molecular import (
    LATTICE_CONSTANTS,
    Molecule,
    StepLog,
    contacts,
    fcc_slab,
    height_above,
    md_dashboard,
    orient_on_surface,
    residue_class,
    rmsd,
    slab_for_footprint,
)


def _pdb_text(n_res=12, models=1, shift=0.0):
    """A helix-like backbone (N, CA, C, O + CB) written as PDB, optionally several MODELs."""
    res = ["ALA", "LYS", "GLU", "SER", "GLY", "LEU"]
    lines = []
    for m in range(models):
        if models > 1:
            lines.append(f"MODEL     {m + 1:4d}")
        k = 1
        for i in range(n_res):
            ang = i * math.radians(100)
            base = np.array([2.3 * math.cos(ang), 2.3 * math.sin(ang), 1.5 * i]) + np.array([shift * m, 0, 0])
            for name, el, off in (("N", "N", (-0.5, 0, -0.4)), ("CA", "C", (0, 0, 0)), ("C", "C", (0.6, 0.3, 0.5)),
                                  ("O", "O", (1.2, 0.6, 0.6)), ("CB", "C", (0.2, -1.2, 0.1))):
                x, y, z = base + np.array(off)
                lines.append(f"ATOM  {k:5d} {name:<4s} {res[i % 6]:>3s} A{i + 1:4d}    "
                             f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}")
                k += 1
        if models > 1:
            lines.append("ENDMDL")
    lines.append("HETATM 9999  O   HOH A 900       0.000   0.000  50.000  1.00  0.00           O")
    return "\n".join(lines) + "\nEND\n"


@pytest.fixture
def mol(tmp_path):
    p = tmp_path / "pep.pdb"
    p.write_text(_pdb_text(models=3, shift=0.5))
    return Molecule.from_pdb(str(p))


def test_pdb_reading_models_and_chemistry(mol):
    assert len(mol) == 60 and len(mol.frames) == 2                       # water dropped, 3 models
    assert mol.elements[:2] == ["N", "C"] and mol.resnames[0] == "ALA"
    assert np.allclose(mol.frames[0] - mol.xyz, [0.5, 0, 0])
    assert residue_class("LYS") == "positive" and residue_class("GLU") == "negative" and residue_class("XYZ") == "other"
    trace = mol.ca_trace()
    assert len(trace) == 1 and len(trace[0]) == 12
    assert mol.masses[1] == pytest.approx(12.011) and mol.radii[0] == pytest.approx(1.55)


@pytest.mark.parametrize("facet,nn,dz", [("111", 1 / math.sqrt(2), 1 / math.sqrt(3)), ("100", 1 / math.sqrt(2), 0.5),
                                         ("110", 1 / math.sqrt(2), 1 / (2 * math.sqrt(2)))])
def test_fcc_slab_geometry(facet, nn, dz):
    a = LATTICE_CONSTANTS["Au"]
    P, cell = fcc_slab("Au", facet, 6, 6, 3)
    assert len(P) == 108 and P[:, 2].max() == pytest.approx(0.0)
    zs = np.unique(np.round(P[:, 2], 6))
    assert np.allclose(np.diff(zs), dz * a)                             # layer spacing
    from scipy.spatial import cKDTree
    d, _ = cKDTree(P).query(P, k=2)
    assert np.min(d[:, 1]) == pytest.approx(nn * a, rel=1e-6)           # nearest-neighbour distance a / sqrt(2)


def test_slab_covers_footprint_and_orientation(mol):
    rng = np.random.default_rng(0)
    blob = rng.normal(size=(500, 3)) * np.array([12.0, 6.0, 2.0])        # thin along its third axis
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    X = orient_on_surface(blob @ R.T, gap=3.0)
    ext = np.ptp(X, axis=0)
    assert ext[2] < ext[1] < ext[0]                                       # thinnest direction is the normal
    assert X[:, 2].min() == pytest.approx(3.0)
    site = np.argsort(X[:, 2])[:10]                                       # lowest atoms as the "active site"
    Y = orient_on_surface(blob @ R.T, active_site=site, gap=3.0)
    assert Y[site, 2].mean() > np.median(Y[:, 2])                         # turned away from the surface
    S, cell, (nx, ny) = slab_for_footprint(X, "Au", "111", padding=5.0)
    assert S[:, 0].min() < X[:, 0].min() - 3 and S[:, 0].max() > X[:, 0].max() + 3
    assert len(S) == nx * ny * 4


def test_analysis_helpers(mol):
    f0 = mol.xyz
    th = 0.7
    Rz = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    frames = [f0, f0 @ Rz.T + 5.0, f0 + np.array([0, 0, 0.1])]
    r = rmsd(frames, f0)
    assert r[0] == pytest.approx(0, abs=1e-9) and r[1] == pytest.approx(0, abs=1e-9)   # rigid motion removed
    assert rmsd([f0 + 1.0], f0, align=False)[0] == pytest.approx(math.sqrt(3))
    z0 = f0[:, 2].min() - 1.0
    assert height_above([f0], z0)[0] == pytest.approx(f0[:, 2].mean() - z0)
    assert contacts([f0], z0, cutoff=2.0)[0] == int(np.sum(f0[:, 2] - z0 < 2.0))


def test_steplog_plain_text():
    buf = io.StringIO()
    log = StepLog("Title", color=False, stream=buf)
    log.step("first")
    log.info("detail")
    log.ok("done")
    log.warn("careful")
    out = buf.getvalue()
    assert "[1] first" in out and "    ok: done" in out and "\033[" not in out and len(log.lines) == 5


def test_dashboard_glb_and_render(mol, tmp_path):
    pytest.importorskip("matplotlib")
    t = np.arange(5) * 0.5
    p = md_dashboard(str(tmp_path / "d.png"), t, potential=-1000 - t, temperature=300 + t, rmsd_A=t / 10,
                     height_A=10 - t, contacts_n=t * 2, target_temperature=300)
    assert (tmp_path / "d.png").stat().st_size > 10_000 and p.endswith("d.png")
    from pinneapple_tools.visualization.molecular import to_glb
    from pinneapple_tools.visualization.studio.scene import read_gltf

    slab, _, _ = slab_for_footprint(orient_on_surface(mol.xyz), "Au", "111", padding=2.0, layers=2)
    to_glb(mol, str(tmp_path / "s.glb"), xyz=orient_on_surface(mol.xyz), slab=slab)
    surfs = read_gltf(str(tmp_path / "s.glb"))
    assert {s.name for s in surfs} >= {"Au", "positive", "negative"}
    pv = pytest.importorskip("pyvista")
    from pinneapple_tools.visualization.molecular import render

    pv.OFF_SCREEN = True
    out = render(mol, str(tmp_path / "r.png"), xyz=orient_on_surface(mol.xyz), slab=slab, size=(320, 200),
                 style="cartoon")
    from PIL import Image

    im = np.asarray(Image.open(out).convert("RGB"), dtype=float)
    assert im.shape == (200, 320, 3) and im.std() > 5                    # not a blank frame


def test_mp4(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    from PIL import Image

    from pinneapple_tools.visualization.molecular import encode_mp4

    paths = []
    for k in range(6):
        p = tmp_path / f"frame_{k:04d}.png"
        Image.fromarray(np.full((64, 96, 3), 40 * k, np.uint8)).save(p)
        paths.append(str(p))
    out = encode_mp4(paths, str(tmp_path / "v.mp4"), fps=6)
    assert (tmp_path / "v.mp4").stat().st_size > 500 and out.endswith("v.mp4")
