"""Lighting and sky presets shared by the studio renderers (PyVista, Blender Cycles, three.js viewer).

    from pinneapple_tools.visualization.studio.lighting import PRESETS, apply_pyvista
    apply_pyvista(plotter, "lunar_south_pole", target=(0, 0, 0), extent=10.0)

Presets
-------
``studio``            the grey-blue gradient and soft three-point light of the CFD renders (Formula car, aircraft).
``lunar_south_pole``  a black sky and one hard, low Sun: at about 86 deg S (the VIPER landing region) the Sun stays
                      within about 5.5 deg of a level horizon (90 - 86 + 1.54 deg lunar obliquity), so shadows are
                      many times longer than the objects. No atmosphere: no sky light, only a faint bounce light
                      from the regolith. The Sun's disc is 0.53 deg across, so shadow edges are sharp.
``lunar_low_sun``     the same black sky with the Sun 12 deg high: still long, hard shadows, but the floors of
                      small craters are partly lit, which keeps data on them readable.
``lunar_day``         the same black sky with the Sun 30 deg high (mid latitudes).

Earth can be added (``earth=True``): a 1.9 deg blue-white disc (Earth's angular diameter seen from the Moon) in the
sky, by default low above the horizon.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class LightingPreset:
    name: str
    sun_elevation_deg: float
    sun_azimuth_deg: float           # direction the light comes FROM, measured from +x towards +y
    sun_intensity: float             # renderer light intensity; for low Suns it includes the camera exposure
    sun_color: tuple
    ambient: float                   # fill light intensity (regolith bounce / studio fill)
    background_top: str
    background_bottom: str
    sun_angle_deg: float = 0.53      # angular diameter of the light source (shadow softness)
    shadows: bool = True
    earth: bool = False
    earth_elevation_deg: float = 12.0
    earth_azimuth_deg: float = 200.0
    exposure: float = 0.0
    description: str = ""

    def sun_direction(self) -> np.ndarray:
        """Unit vector from the scene towards the Sun (z up)."""
        e, a = math.radians(self.sun_elevation_deg), math.radians(self.sun_azimuth_deg)
        return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])

    def earth_direction(self) -> np.ndarray:
        e, a = math.radians(self.earth_elevation_deg), math.radians(self.earth_azimuth_deg)
        return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])

    def with_(self, **kw) -> LightingPreset:
        return replace(self, **kw)


PRESETS = {
    "studio": LightingPreset("studio", 45.0, 225.0, 0.9, (1.0, 1.0, 1.0), 0.45, "#2B3340", "#11151C",
                             sun_angle_deg=8.0, shadows=False, description="CFD studio gradient"),
    "lunar_south_pole": LightingPreset("lunar_south_pole", 5.0, 210.0, 5.0, (1.0, 0.98, 0.95), 0.14, "#000000",
                                       "#000000", description="black sky, Sun 5 deg high, hard long shadows"),
    "lunar_low_sun": LightingPreset("lunar_low_sun", 12.0, 210.0, 3.0, (1.0, 0.98, 0.95), 0.10, "#000000",
                                    "#000000", description="black sky, Sun 12 deg high: hard shadows about 4.7 times "
                                                           "longer than the objects, crater floors partly lit"),
    "lunar_day": LightingPreset("lunar_day", 30.0, 210.0, 1.4, (1.0, 0.98, 0.95), 0.08, "#000000", "#000000",
                                description="black sky, Sun 30 deg high"),
}

EARTH_ANGULAR_DIAMETER_DEG = 1.9      # 12 742 km at 384 400 km


def get_preset(preset) -> LightingPreset:
    if isinstance(preset, LightingPreset):
        return preset
    try:
        return PRESETS[preset]
    except KeyError:
        raise KeyError(f"unknown lighting preset {preset!r}; available: {', '.join(PRESETS)}") from None


def apply_pyvista(plotter, preset="lunar_south_pole", target=(0.0, 0.0, 0.0), extent: float = 10.0,
                  earth: bool | None = None, shadow_resolution: int = 4096) -> LightingPreset:
    """Set the background, a directional Sun (with shadows when the preset asks for them), a weak fill light and,
    optionally, Earth in the sky of a PyVista ``plotter``."""
    import pyvista as pv
    p = get_preset(preset)
    if p.background_top == p.background_bottom:
        plotter.set_background(p.background_bottom)
    else:
        plotter.set_background(p.background_bottom, top=p.background_top)
    plotter.remove_all_lights()
    target = np.asarray(target, float)
    sun = pv.Light(position=tuple(target + p.sun_direction() * extent * 5), focal_point=tuple(target),
                   color=p.sun_color, intensity=p.sun_intensity, light_type="scene light")
    sun.positional = False
    plotter.add_light(sun)
    if p.ambient > 0:
        d = -p.sun_direction()
        d[2] = abs(d[2]) + 0.6
        fill = pv.Light(position=tuple(target + d / np.linalg.norm(d) * extent * 5), focal_point=tuple(target),
                        color=(0.85, 0.87, 0.9), intensity=p.ambient, light_type="scene light")
        fill.positional = False
        plotter.add_light(fill)
    if p.shadows:
        plotter.enable_shadows()
        sp = getattr(getattr(plotter, "_render_passes", None), "_shadow_map_pass", None)
        if sp is not None:
            sp.GetShadowMapBakerPass().SetResolution(shadow_resolution)      # fewer shadow-acne artefacts
    if earth if earth is not None else p.earth:
        add_earth_pyvista(plotter, p, target, extent)
    return p


def add_earth_pyvista(plotter, preset, target, extent: float):
    """Earth as a shaded sphere far away, with the right angular size, lit from the Sun's direction."""
    import pyvista as pv
    p = get_preset(preset)
    dist = 40.0 * extent
    r = dist * math.tan(math.radians(EARTH_ANGULAR_DIAMETER_DEG / 2))
    c = np.asarray(target, float) + p.earth_direction() * dist
    s = pv.Sphere(radius=r, center=tuple(c), theta_resolution=48, phi_resolution=48)
    n = (s.points - c) / r
    lit = np.clip(n @ p.sun_direction(), 0.0, 1.0) ** 0.8
    clouds = 0.5 + 0.5 * np.sin(9 * n[:, 0] + 4 * np.sin(7 * n[:, 2])) * np.cos(6 * n[:, 1])
    ocean = np.array([0.10, 0.25, 0.55])
    white = np.array([0.95, 0.96, 0.98])
    rgb = (ocean + (white - ocean) * (clouds[:, None] > 0.62)) * lit[:, None]
    s["rgb"] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    plotter.add_mesh(s, scalars="rgb", rgb=True, lighting=False, show_scalar_bar=False)


def threejs_config(preset="lunar_south_pole") -> dict:
    """The preset as plain data for the browser viewers (background, sun direction and intensity, fill)."""
    p = get_preset(preset)
    return {"name": p.name, "background": [p.background_top, p.background_bottom],
            "sun": {"direction": np.round(p.sun_direction(), 5).tolist(), "intensity": p.sun_intensity,
                    "color": list(p.sun_color)},
            "ambient": p.ambient, "shadows": p.shadows,
            "earth": {"on": p.earth, "direction": np.round(p.earth_direction(), 5).tolist(),
                      "angular_diameter_deg": EARTH_ANGULAR_DIAMETER_DEG}}


def blender_world(preset="lunar_south_pole", bpy=None):
    """Configure the current Blender scene: world colour, one Sun lamp with the preset's angular size and strength,
    and (for black-sky presets) no world light. Returns the Sun object."""
    if bpy is None:
        import bpy
    p = get_preset(preset)
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    col = tuple(int(p.background_bottom[i:i + 2], 16) / 255 for i in (1, 3, 5))
    bg.inputs[0].default_value = (*[c ** 2.2 for c in col], 1.0)
    bg.inputs[1].default_value = 0.0 if p.background_top == p.background_bottom == "#000000" else 1.0
    data = bpy.data.lights.new("Sun", type="SUN")
    data.energy = 4.0 * p.sun_intensity
    data.angle = math.radians(p.sun_angle_deg)
    data.color = p.sun_color
    sun = bpy.data.objects.new("Sun", data)
    bpy.context.collection.objects.link(sun)
    d = p.sun_direction()
    sun.rotation_euler = (0.0, math.acos(d[2]), math.atan2(d[1], d[0]))
    return sun


__all__ = ["LightingPreset", "PRESETS", "get_preset", "apply_pyvista", "add_earth_pyvista", "threejs_config",
           "blender_world", "EARTH_ANGULAR_DIAMETER_DEG"]
