"""NASA VIPER (Volatiles Investigating Polar Exploration Rover) at full scale, from public figures.

Every published number used here is listed in :data:`VIPER_SOURCES` with its source. Numbers that could not be
found are marked ``"estimate: ..."`` with the reason; they were chosen to agree with the published envelope, track,
wheelbase and wheel size. The geometry is a parametric model built for this library from those numbers, not a copy
of any CAD model.

Caveat on the sources: from the environment this model was built in, the NASA documents could only be located with
a web search, which returned text excerpts; the PDFs themselves could not be opened, so slide or page numbers are not
given and the wording should be checked against the documents.

    from pinneapple_simulation.vehicle.viper import viper
    spec = viper()                       # VehicleSpec with the CAD (spec.cad), corners and the wheel
    spec.cad.export("viper.glb")
"""
from __future__ import annotations

import math

import numpy as np

from pinneapple_design.geometry.vehicle_cad import (
    VehicleCAD,
    beveled_box,
    box,
    camera_head,
    cylinder,
    merge,
    paraboloid_dish,
    radiator,
    rot,
    rover_wheel,
    solar_panel,
)

from ..terramechanics.wheel import Grousers, Wheel
from .spec import Actuator, CornerModule, VehicleSpec

LSIC_2024 = ("Bluethmann, W. (2024). VIPER Rover Overview and Mobility and Hardware. LSIC Tech Infusion panel, "
             "14 Nov 2024. https://ntrs.nasa.gov/api/citations/20240013903/downloads/"
             "LSIC_2024-Tech%20Infusion_Panel-VIPER_Mob-HW.pdf")
FRANCIS_2025 = ("Francis, P.L., Sobey, A.R. (2025). Design and Test of the VIPER Mobility Actuators. NASA NTRS "
                "20250000568. https://ntrs.nasa.gov/citations/20250000568")
FONG_2022 = ("Fong, T. et al. (2022). VIPER mobility, NASA NTRS 20220000441. https://ntrs.nasa.gov/api/citations/"
             "20220000441/downloads/viper-mobility-2022-01-27.pdf")
FONG_2025 = ("Fong, T. (2025). VIPER, NASA NTRS 20250004148. https://ntrs.nasa.gov/api/citations/20250004148/"
             "downloads/viper-2025-04-24.pdf")
PSJ_2025 = ("Investigating the geotechnical properties of the lunar south pole with NASA VIPER's mobility system. "
            "Planetary Science Journal (2025), doi:10.3847/PSJ/add13f")
ICES_2025 = ("McBryan, Francis, Sobey (2025). Dust Mitigation for the VIPER Mobility System, ICES-2025-147. "
             "https://ntrs.nasa.gov/citations/20250005201")
HU_2025 = ("Hu, W. et al. (2025), J. Field Robotics, doi:10.1002/rob.22597, arXiv:2405.11001 (VIPER-class wheel: "
           "radius 0.25 m, width 0.2 m, 24 grousers)")
MAST_2023 = ("NASA (2023). NASA VIPER robotic Moon rover team raises its mighty mast. https://www.nasa.gov/general/"
             "nasa-viper-robotic-moon-rover-team-raises-its-mighty-mast/")
ZACNY_2021 = ("Zacny, K. et al. (2021). TRIDENT drill for VIPER and PRIME1 missions to the Moon. LPSC 52, abstract "
              "2400. https://www.hou.usra.edu/meetings/lpsc2021/pdf/2400.pdf")
ANDREWS_2020 = ("Andrews, D. (2020). VIPER Mission Overview. NASA NTRS 20205000864. https://ntrs.nasa.gov/api/"
                "citations/20205000864/downloads/2020-04-14,%20VIPER%20Mission%20Overview%20(Andrews).pdf")

VIPER_SOURCES = {
    "envelope 1.7 x 1.7 x 2.5 m (L x W x H)": LSIC_2024,
    "mass 450 kg": f"{FRANCIS_2025} ('the four wheeled, 450-kilogram scientific platform'); {LSIC_2024} gives 447 kg "
                   "roving",
    "four wheels, one mobility module per corner (actuated suspension, steering and drive)": FRANCIS_2025,
    "wheel 50 cm diameter, 20 cm wide, rigid sheet metal with spokes and rim grousers": f"{FONG_2025}; {FONG_2022}",
    "24 grousers, 0.025 m tall": f"{HU_2025}; grouser height 2.5 cm also in {LSIC_2024}",
    "steering +/-50 deg (BLDC motor + Harmonic Drive)": ICES_2025,
    "suspension +/-40 deg, 200 N m": f"{FONG_2022}; {FONG_2025}",
    "track about 1.3 m at nominal ride height, wheelbase about 1 m": PSJ_2025,
    "suspension changes ride height and track; steering changes track and wheelbase": PSJ_2025,
    "top speed 20 cm/s, prospecting speed 10 cm/s": FONG_2022,
    "mast tip about 2.5 m above the wheel rims; mast head pans ~400 deg, tilts up to 75 deg": MAST_2023,
    "NavCam stereo pair and LED headlights on the mast head": MAST_2023,
    "4 hazard cameras (1 at each wheel), hazard lights on the sides and corners": FONG_2022,
    "gimballed high-gain antenna (gimbal at about 2.3 m) and a fixed low-gain antenna": LSIC_2024,
    "solar arrays on three sides; radiator on top": f"{LSIC_2024}; {FONG_2022}",
    "TRIDENT 1 m class drill, vertical, in the middle of the rover": ZACNY_2021,
    "NSS on the front; NIRVSS and MSolo under the rover": ANDREWS_2020,
    "design requirement: 15 deg slopes, 15 cm obstacles": LSIC_2024,
    # estimates (not found in the public sources reachable here)
    "suspension arm geometry (pivot, length), steering offset": "estimate: not published; pivots under the chassis "
        "and a 0.15 m steering offset chosen so that the nominal track is 1.3 m, the wheelbase 1.0 m and +/-40 deg of "
        "arm travel changes ride height and track, as the PSJ paper describes",
    "chassis box 1.40 x 1.30 x 0.66 m, belly 0.64 m above ground": "estimate: fits the 1.7 m envelope with the side "
        "arrays and keeps the 0.5 m wheels clear under the deck at +/-50 deg steering",
    "solar array panels 1.30 x 0.78 m": "estimate: panel area not found",
    "high-gain dish diameter 0.40 m": "estimate: not found",
    "mass split (per corner: wheel+drive 12 kg, steering 8 kg, arm 10 kg; chassis 330 kg)": "estimate: only the total "
        "is published; the split sets the centre of mass and inertia computed from the CAD",
    "drive actuator torque limit 80 N m, wheel speed limit 2.5 rad/s": "estimate: not published; 80 N m exceeds the "
        "soil torque of the wheel at 90 % slip under the static lunar load on GRC-1, so traction, not the motor, "
        "limits climbing",
    "steering actuator torque limit 60 N m": "estimate: not published",
    "grouser shape straight (across the tread)": "estimate: the shape (straight or chevron) was not found",
}


def _wheel() -> Wheel:
    g = Grousers(count=24, height=0.025, thickness=0.004, shape="straight", shear_radius_factor=1.0)
    return Wheel(radius=0.25, width=0.20, grousers=g, mass=12.0)


def viper(wheel: Wheel | None = None, total_mass: float = 450.0) -> VehicleSpec:
    """VIPER at nominal ride height on level ground (wheels just touching with their grousers)."""
    w = wheel or _wheel()
    hw = w.outer_radius                       # wheel centre height with the grousers on the surface
    corners = []
    for name, sx, sy in (("FL", 1, 1), ("FR", 1, -1), ("RL", -1, 1), ("RR", -1, -1)):
        P = np.array([0.50 * sx, 0.30 * sy, hw + 0.37])
        K = np.array([0.50 * sx, 0.50 * sy, hw + 0.31])
        C = np.array([0.50 * sx, 0.65 * sy, hw])
        corners.append(CornerModule(
            name, P, np.array([1.0, 0.0, 0.0]), K, np.array([0.0, 0.0, 1.0]), C,
            susp_limits=(-math.radians(40), math.radians(40)), steer_limits=(-math.radians(50), math.radians(50)),
            susp=Actuator(200.0, 0.5, kp=1500.0, ki=0.0, kd=300.0),
            steer=Actuator(60.0, 1.0, kp=800.0, kd=80.0),
            drive=Actuator(80.0, 2.5, kp=120.0, ki=60.0), side=sy, front=sx))
    spec = VehicleSpec("NASA VIPER", w, corners, total_mass, max_speed=0.20, sources=dict(VIPER_SOURCES))
    spec.cad = viper_cad(spec)
    return spec


def viper_cad(spec: VehicleSpec) -> VehicleCAD:
    """Parametric CAD of VIPER (bodies ``chassis``, ``arm_<c>``, ``steer_<c>``, ``wheel_<c>``) with masses."""
    w = spec.wheel
    hw = w.outer_radius
    cad = VehicleCAD("NASA VIPER (parametric, public figures)")
    zb, zt = hw + 0.37, hw + 1.03                       # chassis belly and deck
    zc = 0.5 * (zb + zt)
    H = zt - zb
    m_corner = {"arm": 10.0, "steer": 8.0, "wheel": w.mass}
    m_chassis = spec.total_mass - 4 * sum(m_corner.values())
    alloc = {"structure": 0.36, "avionics": 0.27, "arrays": 0.11, "radiator": 0.045, "mast": 0.075, "drill": 0.06,
             "instruments": 0.045, "susp_housings": 0.035}
    M = {k: v * m_chassis / sum(alloc.values()) for k, v in alloc.items()}
    # chassis: structure (white box with a gold MLI belly)
    cad.add("chassis_box", "chassis", beveled_box((1.40, 1.30, H - 0.05), (0, 0, zc + 0.025), 0.035), "white_paint",
            M["structure"] * 0.85)
    cad.add("chassis_belly", "chassis", box((1.32, 1.22, 0.05), (0, 0, zb + 0.025)), "mli_gold",
            M["structure"] * 0.15)
    cad.add("avionics", "chassis", box((0.9, 0.8, 0.25), (0.0, 0.0, zb + 0.18)), "mli_gold", M["avionics"])
    # solar arrays: left, right, aft
    pw, ph = 1.30, 0.78
    for nm, R, t in (("array_L", rot((0, 0, 1), math.pi), (0.0, 0.81, zc + 0.04)),
                     ("array_R", np.eye(3), (0.0, -0.81, zc + 0.04)),
                     ("array_aft", rot((0, 0, 1), -math.pi / 2), (-0.81, 0.0, zc + 0.04))):
        cad.add_assembly(nm, "chassis", solar_panel(pw, ph, cells=(8, 4)), M["arrays"] / 3, R=R, t=t)
        for k in (-1, 1):
            br = box((0.06, 0.12, 0.06), (k * 0.45, 0.07, ph / 2 - 0.1))
            cad.add(f"{nm}_bracket{k}", "chassis", (br[0] @ np.asarray(R).T + t, br[1]), "aluminium", 0.0)
    # radiator on the deck
    cad.add_assembly("radiator", "chassis", radiator(1.05, 0.95, fins=14), M["radiator"], t=(-0.08, 0.0, zt + 0.02))
    # mast with camera head, high-gain antenna, low-gain antenna
    mx, my = 0.52, 0.40
    mast_top = 2.50 - 0.20
    cad.add("mast", "chassis", cylinder((mx, my, zt), (mx, my, mast_top), 0.045, 28), "white_paint", M["mast"] * 0.35)
    cad.add("mast_base", "chassis", cylinder((mx, my, zt), (mx, my, zt + 0.10), 0.085, 28), "anodized_black",
            M["mast"] * 0.1)
    cad.add("mast_gimbal", "chassis", cylinder((mx, my, mast_top), (mx, my, mast_top + 0.06), 0.06, 24),
            "anodized_black", M["mast"] * 0.05)
    cad.add_assembly("camera_head", "chassis", camera_head(0.34, 0.16, 0.13, 0.20), M["mast"] * 0.30,
                     t=(mx, my, 2.50 - 0.065))
    hga_c = np.array([mx - 0.30, my, 2.30])
    cad.add("hga_boom", "chassis", cylinder((mx, my, 2.12), hga_c - [0.0, 0, 0.05], 0.022, 16), "aluminium",
            M["mast"] * 0.05)
    ax = np.array([-0.45, 0.0, 0.89])
    cad.add("hga_dish", "chassis", paraboloid_dish(0.40, 0.06, 0.006, hga_c, ax / np.linalg.norm(ax)), "white_paint",
            M["mast"] * 0.12)
    cad.add("hga_feed", "chassis", cylinder(hga_c, hga_c + 0.16 * ax / np.linalg.norm(ax), 0.012, 12), "aluminium",
            M["mast"] * 0.03)
    cad.add("lga", "chassis", merge([cylinder((-0.60, -0.50, zt), (-0.60, -0.50, zt + 0.16), 0.02, 16),
                                      cylinder((-0.60, -0.50, zt + 0.16), (-0.60, -0.50, zt + 0.24), (0.05, 0.01), 16)]),
            "white_paint", 0.0)
    # TRIDENT drill: vertical in the middle of the rover (housing above the deck, bit below the belly)
    cad.add("drill_tower", "chassis", beveled_box((0.16, 0.16, 0.42), (0.10, 0.0, zt + 0.19), 0.02), "white_paint",
            M["drill"] * 0.6)
    cad.add("drill_bit", "chassis", cylinder((0.10, 0.0, zb), (0.10, 0.0, zb - 0.16), (0.03, 0.012), 20),
            "steel", M["drill"] * 0.4)
    # instruments: NSS on the front, NIRVSS and MSolo under the rover
    cad.add("NSS", "chassis", beveled_box((0.10, 0.26, 0.20), (0.75, -0.25, zc - 0.05), 0.015), "mli_gold",
            M["instruments"] * 0.4)
    cad.add("NIRVSS", "chassis", beveled_box((0.24, 0.18, 0.08), (0.35, -0.30, zb - 0.04), 0.01), "anodized_black",
            M["instruments"] * 0.35)
    cad.add("MSolo", "chassis", beveled_box((0.22, 0.16, 0.08), (-0.30, 0.25, zb - 0.04), 0.01), "anodized_black",
            M["instruments"] * 0.25)
    # hazard cameras (one per wheel) and hazard lights at the corners
    for c in spec.corners:
        sx, sy = c.front, c.side
        p = np.array([0.70 * sx, 0.52 * sy, zb + 0.06])
        cad.add(f"hazcam_{c.name}", "chassis", beveled_box((0.07, 0.07, 0.05), p, 0.008), "anodized_black", 0.0)
        cad.add(f"hazcam_lens_{c.name}", "chassis", cylinder(p + [0.035 * sx, 0, 0], p + [0.045 * sx, 0, 0], 0.015,
                                                             16), "lens", 0.0)
        cad.add(f"hazlight_{c.name}", "chassis", box((0.01, 0.10, 0.03), (0.705 * sx, 0.32 * sy, zb + 0.06)), "led",
                0.0)
        # suspension actuator housing (fixed to the chassis, at the arm pivot)
        P = c.susp_pivot
        cad.add(f"susp_housing_{c.name}", "chassis", cylinder(P - [0.09, 0, 0], P + [0.09, 0, 0], 0.065, 28),
                "anodized_black", M["susp_housings"] / 4)
        cad.add(f"susp_mount_{c.name}", "chassis", box((0.20, 0.12, zb - P[2] + 0.06), (P[0], P[1],
                                                                                         0.5 * (zb + P[2]) + 0.03)),
                "aluminium", 0.0)
        _corner_cad(cad, c, w, m_corner)
    return cad


def _corner_cad(cad: VehicleCAD, c: CornerModule, w: Wheel, m: dict) -> None:
    P, K, C = c.susp_pivot, c.steer_origin, c.wheel_centre
    sy = c.side
    # arm: a box beam from the pivot to the steering head, plus clevis plates
    d = K - P
    L = np.linalg.norm(d)
    ang = math.atan2(d[2], abs(d[1]))
    Rarm = rot((1, 0, 0), -sy * ang) if sy > 0 else rot((1, 0, 0), sy * ang)
    beam = beveled_box((0.11, L + 0.08, 0.075), (0, 0, 0), 0.02)
    Rb = rot((1, 0, 0), ang if sy < 0 else -ang)
    beam = (beam[0] @ Rb.T + 0.5 * (P + K), beam[1])
    cad.add(f"arm_{c.name}_beam", f"arm_{c.name}", beam, "white_paint", m["arm"] * 0.8)
    cad.add(f"arm_{c.name}_hub", f"arm_{c.name}", cylinder(P - [0.10, 0, 0], P + [0.10, 0, 0], 0.045, 24), "aluminium",
            m["arm"] * 0.2)
    del Rarm
    # steering module: actuator drum on the arm end, a vertical column and a knuckle to the wheel hub
    top, bot = K + [0, 0, 0.07], np.array([K[0], K[1], C[2] + 0.06])
    cad.add(f"steer_{c.name}_drum", f"steer_{c.name}", cylinder(K - [0, 0, 0.03], top, 0.07, 28), "anodized_black",
            m["steer"] * 0.45)
    cad.add(f"steer_{c.name}_column", f"steer_{c.name}", cylinder(K, bot, 0.045, 24), "white_paint",
            m["steer"] * 0.3)
    hub_in = np.array([C[0], C[1] - sy * (0.5 * w.width + 0.02), C[2]])
    kn = beveled_box((0.09, abs(hub_in[1] - K[1]) + 0.09, 0.12), (C[0], 0.5 * (hub_in[1] + K[1]), C[2] + 0.03), 0.02)
    cad.add(f"steer_{c.name}_knuckle", f"steer_{c.name}", kn, "white_paint", m["steer"] * 0.25)
    # wheel with its drive actuator in the hub
    parts = rover_wheel(w.radius, w.width, grousers=w.grousers.count, grouser_height=w.grousers.height,
                        grouser_thickness=w.grousers.thickness, grouser_shape=w.grousers.shape,
                        chevron_angle_deg=w.grousers.chevron_angle_deg or 20.0, spokes=10, spoke_width=0.035,
                        hub_radius=0.075, hub_length=0.16, inner_side=-sy)
    cad.add_assembly(f"wheel_{c.name}", f"wheel_{c.name}", parts, m["wheel"], t=C)


__all__ = ["viper", "viper_cad", "VIPER_SOURCES"]
