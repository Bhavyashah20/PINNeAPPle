"""NREL 5-MW reference wind turbine (Jonkman et al. 2009): three blades from the published chord / twist / airfoil
table, hub and spinner, nacelle, tapered tubular tower."""
from __future__ import annotations

import math

import numpy as np

from .. import kit
from ..core import Body, GeometrySpec, Param, Source, register

JONKMAN = ("Jonkman, J., Butterfield, S., Musial, W., Scott, G. (2009). Definition of a 5-MW Reference Wind Turbine "
           "for Offshore System Development. NREL/TP-500-38060. https://doi.org/10.2172/947422")

# Table 2-1 of Jonkman et al. (2009): radius from the rotor centre [m], aerodynamic twist [deg], chord [m], t/c of
# the airfoil family at the node (Cylinder 1.0, DU40 0.405, DU35 0.35, DU30 0.30, DU25 0.25, DU21 0.21,
# NACA64-618 0.18)
BLADE = np.array([
    [2.8667, 13.308, 3.542, 1.000], [5.6000, 13.308, 3.854, 1.000], [8.3333, 13.308, 4.167, 1.000],
    [11.7500, 13.308, 4.557, 0.405], [15.8500, 11.480, 4.652, 0.350], [19.9500, 10.162, 4.458, 0.350],
    [24.0500, 9.011, 4.249, 0.300], [28.1500, 7.795, 4.007, 0.250], [32.2500, 6.544, 3.748, 0.250],
    [36.3500, 5.361, 3.502, 0.210], [40.4500, 4.188, 3.256, 0.210], [44.5500, 3.125, 3.010, 0.180],
    [48.6500, 2.319, 2.764, 0.180], [52.7500, 1.526, 2.518, 0.180], [56.1667, 0.863, 2.313, 0.180],
    [58.9000, 0.370, 2.086, 0.180], [61.6333, 0.106, 1.419, 0.180]])

MATERIALS = {
    "turbine_white": {"color": [0.93, 0.94, 0.94], "metallic": 0.0, "roughness": 0.35, "clearcoat": 0.6},
    "turbine_grey": {"color": [0.78, 0.80, 0.81], "metallic": 0.0, "roughness": 0.4},
    "tower_band": {"color": [0.80, 0.12, 0.10], "metallic": 0.0, "roughness": 0.4},
}


def _blade_stations(R_tip: float, hub_r: float, n_span: int, n_sec: int, pitch: float):
    """Stations of one blade along +r: (radius, chord, twist + pitch, section outline)."""
    r_tab = BLADE[:, 0]
    r = np.r_[hub_r, r_tab, R_tip]
    chord = np.r_[3.542, BLADE[:, 2], 0.55]
    twist = np.r_[13.308, BLADE[:, 1], 0.0]
    tc = np.r_[1.0, BLADE[:, 3], 0.18]
    # spanwise sampling: dense at the root transition and the tip
    u = np.linspace(0, 1, n_span)
    rr = hub_r + (R_tip - hub_r) * (0.5 * u + 0.25 * (1 - np.cos(np.pi * u)))
    rr = np.unique(np.r_[rr, r_tab[[2, 3]]])
    from scipy.interpolate import PchipInterpolator
    ch = PchipInterpolator(r, chord)(rr)
    tw = PchipInterpolator(r, twist)(rr)
    t = np.interp(rr, r, tc)
    circ = kit.round_section(n_sec)
    out = []
    for ri, ci, twi, ti in zip(rr, ch, tw, t, strict=False):
        if ti >= 0.999:
            sec = circ
            pivot = 0.5
        else:
            cam = 0.02 + 0.02 * np.clip((0.40 - ti) / 0.22, 0, 1)          # cambered outboard sections
            foil = kit.naca4(cam, 0.4, min(ti, 0.40), n_sec)
            foil = kit.thick_te(foil, 0.02 * np.clip((ti - 0.18) / 0.22, 0, 1) + 0.004)
            w = np.clip((ti - 0.405) / (1.0 - 0.405), 0, 1)               # cylinder -> DU40 transition
            sec = kit.blend_sections(foil, circ, w)
            pivot = 0.25 + 0.25 * w
        out.append((ri, ci, twi + pitch, sec, pivot))
    return out


def _blade(p, res, azimuth_deg):
    R_tip, hub_r = p["rotor_diameter"] / 2, p["hub_diameter"] / 2
    n_sec = max(14, int(round(30 * res)))
    st = _blade_stations(R_tip, hub_r, max(18, int(round(30 * res))), n_sec, p["pitch_deg"])
    rings = []
    for ri, ci, twi, sec, pv in st:
        # blade frame before rotation: span along +z, chord along +y (rotation direction), thickness along -x
        ring = kit.section_ring(sec, ci, twi, (0.0, -pv * ci, ri), frame=((0, 1.0, 0), (-1.0, 0, 0)), pivot=pv)
        rings.append(ring)
    V, F = kit.loft(np.asarray(rings), "cap", "cap")
    # precone (tilts the blade upwind, -x), then azimuth about the shaft (x axis)
    V = V @ kit.rot((0, 1, 0), p["precone_deg"]).T
    V = V @ kit.rot((1, 0, 0), azimuth_deg).T
    return V, F


def _build(p, res):
    H = p["hub_height"]
    tilt = p["shaft_tilt_deg"]
    overhang = p["overhang"]
    nt = max(24, int(round(40 * res)))
    # rotor, in the hub frame (x downwind along the shaft), then tilted and moved to the hub centre
    hub_c = np.array([-overhang, 0.0, H])
    Rt = kit.rot((0, 1, 0), -tilt)                                   # nose of the shaft up (upwind end raised)

    def to_world(mesh):
        V, F = mesh
        return V @ Rt.T + hub_c, F

    bodies = []
    for k in range(3):
        az = p["rotor_azimuth_deg"] + 120.0 * k
        bodies.append(Body(f"blade_{k + 1}", *to_world(_blade(p, res, az)), "turbine_white", "rotor"))
    rh = p["hub_diameter"] / 2
    xs = np.r_[np.linspace(-3.6, -1.2, max(8, int(10 * res))), np.linspace(-0.9, 2.2, 6)]
    rs = np.r_[rh * 1.02 * np.sqrt(np.clip(1 - ((xs[:max(8, int(10 * res))] + 1.2) / 2.4) ** 2, 0, 1)),
               np.full(6, rh * 1.02)]
    rs[0] = 0.0
    bodies.append(Body("hub_spinner", *to_world(kit.revolve(xs, rs, nt)), "turbine_white", "rotor"))
    # nacelle: rounded box on the tower top, shaft axis 1.96 m above the tower top (Jonkman Table 6-1)
    zt = H - 1.96256
    Ln, Wn, Hn = p["nacelle_length"], 3.8, 3.9
    xn0 = -overhang + 2.2
    nac = kit.rounded_box((Ln, Wn, Hn), (xn0 + Ln / 2, 0, zt + Hn / 2 - 0.4), radius=0.8,
                          n_side=max(3, int(3 * res)))
    bodies.append(Body("nacelle", *nac, "turbine_white", "nacelle"))
    cool = kit.rounded_box((2.2, 2.4, 0.7), (xn0 + Ln - 2.0, 0, zt + Hn - 0.4 + 0.3), radius=0.2, n_side=2)
    bodies.append(Body("nacelle_cooler", *cool, "turbine_grey", "nacelle"))
    # tower: linear taper 6 m -> 3.87 m from the base to 87.6 m (Jonkman section 6), red band near the base
    zs = np.unique(np.r_[np.linspace(0.0, zt, max(10, int(round(24 * res)))), 6.0, 10.0])
    rT = 3.0 + (1.935 - 3.0) * zs / 87.6
    ring = np.linspace(0, 2 * math.pi, nt * 2, endpoint=False)
    rings = [np.stack([r * np.cos(ring), r * np.sin(ring), np.full_like(ring, z)], 1) for z, r in zip(zs, rT, strict=False)]
    Vt, Ft = kit.loft(rings)
    zc = Vt[Ft].mean(1)[:, 2]
    mats = np.where((zc > 6.0) & (zc < 10.0), "tower_band", "turbine_grey")
    bodies.append(Body("tower", Vt, Ft, mats, "tower"))
    return bodies


register(GeometrySpec(
    name="wind_turbine_nrel5mw", title="NREL 5-MW reference wind turbine", category="energy",
    description=("Three-bladed upwind 5-MW turbine: blades lofted through the published chord / twist table with "
                 "a cylinder-to-airfoil root transition, 2.5 deg precone, 5 deg shaft tilt, hub with spinner, "
                 "nacelle and tapered tower (onshore, base at z = 0). Rotor azimuth, blade pitch and yaw-free "
                 "geometry for rotor CFD (actuator line / sliding mesh), wake and loads studies."),
    builder=_build,
    params=[
        Param("rotor_diameter", 126.0, "m", (100.0, 140.0), "rotor diameter (scales nothing else: tip extension)"),
        Param("hub_diameter", 3.0, "m", (2.5, 4.0), "hub diameter"),
        Param("hub_height", 90.0, "m", (80.0, 120.0), "hub height above the ground"),
        Param("pitch_deg", 0.0, "deg", (-5.0, 90.0), "collective blade pitch (90 = feathered)"),
        Param("rotor_azimuth_deg", 0.0, "deg", (0.0, 360.0), "rotor azimuth (0 = blade 1 up)"),
        Param("precone_deg", -2.5, "deg", (-5.0, 0.0), "precone (negative tilts the blades upwind)"),
        Param("shaft_tilt_deg", 5.0, "deg", (0.0, 8.0), "shaft tilt"),
        Param("overhang", 5.0, "m", (3.0, 7.0), "rotor centre upwind of the tower axis"),
        Param("nacelle_length", 14.0, "m", (10.0, 18.0), "nacelle length"),
    ],
    dimensions={"length_x": (24.49, 0.02), "width_y": (109.3, 0.02), "height_z": (152.5, 0.02)},
    frame=("right-handed, metres: x downwind, y lateral, z up; origin at the tower base on the ground; rotor "
           "upwind of the tower (x < 0)"),
    sources=[Source("rotor 126 m, hub 3 m, hub height 90 m, precone 2.5 deg, shaft tilt 5 deg, overhang 5 m, blade "
                    "chord / twist / airfoil table, tower 6 m -> 3.87 m over 87.6 m", JONKMAN),
             Source("airfoil shapes: the DU 40/35/30/25/21 and NACA 64-618 sections are approximated by NACA "
                    "4-digit sections with the same thickness ratio and 2-4 % camber",
                    "approximation (coordinates of the DU sections are not reproduced)", estimate=True),
             Source("bounding box: 109.3 m wide because at azimuth 0 the blades point up and at +-120 deg "
                    "(2 R sin 120 deg), 152.5 m high (hub 90 m + R cos(precone))", "model geometry"),
             Source("nacelle 14 x 3.8 x 3.9 m and spinner shape", "estimates from 5-MW-class nacelles",
                    estimate=True)],
    physics=["rotor aerodynamics (BEM, actuator line, blade-resolved CFD)", "wake and wind-farm interaction",
             "aeroelastic blade loads", "tower shadow, icing, erosion of the leading edge"],
    materials=MATERIALS, tags=("wind energy", "rotor", "reference turbine"),
    camera={"azimuth": 205.0, "elevation": 8.0},
))
