"""Studio lighting presets, heightfield shadows, Lunar-Lambert photometry, the planetary renderer and the Blender /
three.js preset hooks."""
import math
import os
import types

import numpy as np
import pytest

from pinneapple_tools.visualization.studio.lighting import (
    PRESETS,
    blender_world,
    get_preset,
    threejs_config,
)
from pinneapple_tools.visualization.studio.shadows import (
    cast_shadow,
    lunar_lambert,
    points_in_terrain_shadow,
    shade_terrain,
    terrain_self_shadow,
)


def test_presets():
    p = get_preset("lunar_south_pole")
    assert p.background_top == p.background_bottom == "#000000"
    d = p.sun_direction()
    assert np.linalg.norm(d) == pytest.approx(1.0) and math.degrees(math.asin(d[2])) == pytest.approx(5.0)
    assert get_preset("lunar_low_sun").sun_elevation_deg == 12.0
    cfg = threejs_config("lunar_south_pole")
    assert cfg["background"] == ["#000000", "#000000"] and len(cfg["sun"]["direction"]) == 3
    assert set(PRESETS) >= {"studio", "lunar_south_pole", "lunar_low_sun", "lunar_day"}
    with pytest.raises(KeyError):
        get_preset("mars")


def _grid(n=101, L=10.0):
    x = np.linspace(-L / 2, L / 2, n)
    return np.meshgrid(x, x)


def test_wall_and_box_shadows_have_the_right_length():
    X, Y = _grid()
    Z = np.where(np.abs(X) < 0.25, 1.0, 0.0)                         # a 1 m wall across y at x = 0
    sun = np.array([-math.cos(math.radians(20)), 0.0, math.sin(math.radians(20))])   # Sun low in the -x direction
    sh = terrain_self_shadow(Z, X[0, 1] - X[0, 0], sun)
    row = sh[50]
    xs = X[50][row & (X[50] > 0.3)]
    assert xs.max() == pytest.approx(0.25 + 1 / math.tan(math.radians(20)), abs=0.15)   # shadow length h / tan(e)
    assert not row[X[50] < -0.3].any()
    flat = np.zeros_like(X)
    V = np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0], [0, 0, 1], [0.5, 0, 1], [0.5, 0.5, 1], [0, 0.5, 1.]])
    F = np.array([[4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6]])
    cs = cast_shadow([(V, F)], X, Y, flat, sun)
    assert cs.any() and X[cs].max() == pytest.approx(0.5 + 1 / math.tan(math.radians(20)), abs=0.15)
    soft = cast_shadow([(V, F)], X, Y, flat, sun, supersample=4)     # shadow fraction on the same grid
    assert soft.dtype.kind == "f" and 0.0 <= soft.min() and soft.max() == 1.0
    area = 0.25 + 0.5 / math.tan(math.radians(20))                  # the box's shadow: base + swept top edge
    cell2 = (X[0, 1] - X[0, 0]) ** 2
    assert ((soft > 0) & (soft < 1)).any()
    assert soft.sum() * cell2 == pytest.approx(area, rel=0.12) and cs.sum() * cell2 == pytest.approx(area, rel=0.12)
    pts = np.array([[1.0, 0.0, 0.1], [-1.0, 0.0, 0.1]])
    np.testing.assert_array_equal(points_in_terrain_shadow(pts, X, Y, Z, sun), [True, False])


def test_lunar_lambert_and_shading():
    n = np.array([[0.0, 0.0, 1.0]])
    s = np.array([1.0, 0.0, 0.2])
    s /= np.linalg.norm(s)
    v = np.array([0.0, 0.0, 1.0])
    lam = lunar_lambert(n, s, v, L=0.0)
    ls = lunar_lambert(n, s, v, L=1.0)
    assert lam[0] == pytest.approx(s[2]) and ls[0] > lam[0]             # Lommel-Seeliger brighter at grazing Sun
    X, Y = _grid(21, 2.0)
    rgb = shade_terrain(X, Y, np.zeros_like(X), s, (0, -5, 3), shadow=np.zeros(X.shape, bool))
    dark = shade_terrain(X, Y, np.zeros_like(X), s, (0, -5, 3), shadow=np.ones(X.shape, bool))
    assert rgb.shape == (21, 21, 3) and (rgb > dark).all() and rgb.max() <= 1.0
    half = shade_terrain(X, Y, np.zeros_like(X), s, (0, -5, 3), shadow=np.full(X.shape, 0.5))   # partial shadow
    assert (half < rgb).all() and (half > dark).all()


def test_blender_world_with_a_stub_bpy():
    class Obj(types.SimpleNamespace):
        pass
    created = {}

    def new_light(name, type):
        created["light"] = Obj(energy=0, angle=0, color=None)
        return created["light"]
    bg = Obj(inputs=[Obj(default_value=None), Obj(default_value=None)])
    world = Obj(use_nodes=False, node_tree=Obj(nodes={"Background": bg}))
    linked = []
    bpy = Obj(context=Obj(scene=Obj(world=world), collection=Obj(objects=Obj(link=linked.append))),
              data=Obj(worlds=Obj(new=lambda n: world), lights=Obj(new=new_light),
                       objects=Obj(new=lambda n, d: Obj(rotation_euler=None, data=d))))
    sun = blender_world("lunar_south_pole", bpy)
    assert bg.inputs[1].default_value == 0.0                           # no sky light on the Moon
    assert created["light"].angle == pytest.approx(math.radians(0.53))
    assert sun.rotation_euler[1] == pytest.approx(math.radians(85.0))
    assert linked


def test_scene_carries_the_lighting_for_the_browser():
    from pinneapple_tools.visualization.studio.scene import Scene
    sc = Scene.from_arrays(np.eye(3), [[0, 1, 2]]).set_lighting("lunar_low_sun")
    assert sc.extras()["lighting"]["name"] == "lunar_low_sun"


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="PyVista needs a display (xvfb-run)")
def test_planetary_render(tmp_path):
    pytest.importorskip("pyvista")
    from pinneapple_design.geometry.vehicle_cad import box
    from pinneapple_tools.visualization.studio.planetary import SurfaceView, render_surface
    X, Y = _grid(81, 8.0)
    Z = 0.02 * np.sin(X) * np.cos(Y)
    V, F = box((0.6, 0.4, 0.4), (0, 0, 0.3))
    view = SurfaceView(X, Y, Z, parts=[(V, F, "white_paint"), (V + [1, 1, 0], F, "grey", V[:, 2] * 10)],
                       terrain_field=np.where(np.abs(Y) < 0.3, 5.0, np.nan), clim=(0, 10), colorbar_title="x [kPa]")
    img = render_surface(view, None, ((-4, -4, 2), (0, 0, 0.3)), size=(320, 200), return_image=True)
    assert img.shape[:2] == (200, 320)
    assert img[:20].mean() < 10                                          # black sky
    assert img[150:].mean() > 20                                         # lit regolith
