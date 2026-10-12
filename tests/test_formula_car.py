"""FormulaCar geometry: closed outward surfaces, components, DRS, frontal area; PyVista scene renders."""
import numpy as np
import pytest

from pinneapple_design.geometry.vehicles3d import FormulaCar, _signed_volume, naca4


def _closed(V, F):
    e = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def test_cfd_bodies_closed_and_outward():
    car = FormulaCar()
    bodies = car.cfd_bodies()
    assert set(bodies) == {"body", "floor", "front_wing", "rear_wing", "helmet", "wheels"}
    for name, (V, F) in bodies.items():
        assert _signed_volume(V, F) > 0, name
    V, F = bodies["body"]
    assert _closed(V, F)
    allv = np.vstack([V for V, _ in bodies.values()])
    assert -0.02 < allv[:, 2].min() < 0.0               # tyres sink 1 cm into the road: a contact patch
    assert np.allclose(allv[:, 1].min(), -allv[:, 1].max(), atol=1e-6)      # symmetric about y = 0
    assert 5.0 < np.ptp(allv[:, 0]) < 6.0 and np.ptp(allv[:, 1]) < 2.1 and allv[:, 2].max() < 1.1


def test_naca_profile_and_wings():
    P = naca4(0.06, 0.4, 0.12, 40)
    assert P[:, 0].min() == pytest.approx(0.0, abs=2e-3) and P[:, 0].max() == pytest.approx(1.0, abs=1e-3)
    assert 0.12 < np.ptp(P[:, 1]) < 0.18                    # thickness 0.12 plus part of the 0.06 camber
    closed, drs = FormulaCar(), FormulaCar(drs_open=True)
    flap = [p for p in closed.parts() if p.name == "rear_wing_1"][0]
    flap_drs = [p for p in drs.parts() if p.name == "rear_wing_1"][0]
    # the open flap is flatter: a smaller vertical extent
    assert np.ptp(flap_drs.vertices[:, 2]) < 0.6 * np.ptp(flap.vertices[:, 2])
    fw = [p for p in closed.parts() if p.group == "front_wing"]
    assert len(fw) == 4                                                      # main plane, flap, two endplates


def test_frontal_area_reasonable():
    a = FormulaCar().frontal_area()
    assert 1.2 < a < 2.0
    assert FormulaCar(drs_open=True).frontal_area() < a


def test_pyvista_render(tmp_path):
    pv = pytest.importorskip("pyvista")
    from pinneapple_tools.visualization.studio.pyvista_render import render_scene
    from pinneapple_tools.visualization.studio.scene import Scene

    pv.OFF_SCREEN = True
    sc = Scene.from_parts(FormulaCar().parts(), axes="z_up")
    V = sc.surfaces[0].vertices
    sc.surfaces[0].fields["cp"] = V[:, 2]
    sc.add_lines([np.array([[0, 0.3, 0.2], [3, 0.3, 0.5], [6, 0.3, 0.4]])], [np.array([1.0, 0.8, 1.1])])
    out = render_scene(sc, str(tmp_path / "a.png"), view="iso", ground=True, size=(400, 240))
    render_scene(sc, str(tmp_path / "b.png"), field="cp", size=(400, 240))
    render_scene(sc, str(tmp_path / "c.png"), lines=True, size=(400, 240))
    from PIL import Image

    im = np.asarray(Image.open(out).convert("RGB"), float)
    assert im.shape == (240, 400, 3) and im.std() > 5
