"""NASA VIPER (Volatiles Investigating Polar Exploration Rover) at full scale, from public figures.

Every published number used here is listed in :data:`VIPER_SOURCES` with its source. Numbers that could not be
found are marked ``"estimate: ..."`` with the reason; they were chosen to agree with the published envelope, track,
wheelbase and wheel size, and the layout (where the mast, arrays, radiator, drill and instruments sit) follows NASA's
published photos and renders of the rover. The geometry is a parametric model built for this library from those
numbers, not a copy of any CAD model.

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
    actuator,
    box,
    box_beam,
    chamfered_box,
    chamfered_rect,
    cylinder,
    drill_assembly,
    extrude_polygon,
    merge,
    mli_blanket,
    pan_tilt_head,
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
TRIDENT_2024 = ("NASA (2024). Into the Belly of the Rover: VIPER's Final Science Instrument Installed. "
                "https://www.nasa.gov/general/into-the-belly-of-the-rover-vipers-final-science-instrument-installed/")
TRIDENT_LPSC_2024 = ("TRIDENT drill, LPSC 2024 (Honeybee Robotics / NASA). NASA NTRS 20240000585. "
                     "https://ntrs.nasa.gov/citations/20240000585")
TRIDENT_POSTER = ("Chu, P. et al. TRIDENT drill for VIPER and PRIME-1 (Honeybee Robotics), LSIC poster. "
                  "https://lsic.jhuapl.edu/uploadedDocs/posters/401-Poster%20PDF_8-Chu.pdf")
NASA_INSTRUMENTS = "NASA Science. VIPER: Rover and Instruments. https://science.nasa.gov/mission/viper/rover-and-instruments/"
WIKI_VIPER = ("VIPER (rover). Wikipedia, infobox (2.45 m tall, 1.53 m long and wide), retrieved 2026. "
              "https://en.wikipedia.org/wiki/VIPER_(rover)")

VIPER_SOURCES = {
    "envelope 1.7 x 1.7 x 2.5 m (L x W x H)": LSIC_2024,
    "body about 1.5 m long and wide, 2.45 m tall with the mast (golf-cart size)": f"{WIKI_VIPER}; {MAST_2023} gives "
        "8 ft (about 2.4 m) with the mast",
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
    "solar arrays on three sides (left, right, aft); radiator on top": f"{LSIC_2024}; {FONG_2022}; "
        f"{ANDREWS_2020} (solar array on 3 sides, heat-rejection radiator on top)",
    "TRIDENT: 1 m class rotary-percussive drill, vertical, 10 cm bites": f"{ZACNY_2021}; {TRIDENT_POSTER}",
    "TRIDENT integrated into the rover belly; stows inside the rover and extends 1 m to drill": TRIDENT_2024,
    "TRIDENT mass < 20 kg, 200 N maximum weight on bit": TRIDENT_LPSC_2024,
    "TRIDENT parts: deployment linear stage, drill feed linear stage, brush, sample chute, footpad": TRIDENT_POSTER,
    "cuttings brushed off the auger flutes into a chute, the pile analysed by NIRVSS and MSolo": NASA_INSTRUMENTS,
    "NSS on the front; NIRVSS and MSolo under the rover": ANDREWS_2020,
    "design requirement: 15 deg slopes, 15 cm obstacles": LSIC_2024,
    # estimates (not found in the public sources reachable here)
    "suspension arm geometry (pivot, length), steering offset": "estimate: not published; pivots under the chassis "
        "and a 0.15 m steering offset chosen so that the nominal track is 1.3 m, the wheelbase 1.0 m and +/-40 deg of "
        "arm travel changes ride height and track, as the PSJ paper describes",
    "body 1.44 x 1.48 x 0.58 m with 0.07 m chamfered vertical and top edges, belly 0.72 m above ground, a 1.20 x 0.40 "
    "x 0.15 m keel under it carrying the suspension actuators": "estimate: proportions read from NASA photos and "
        "renders of the rover; sized so that with the arrays it fills the 1.5 m published body width and 1.7 m "
        "envelope, and the 0.5 m wheels stay clear of it at +/-50 deg steering",
    "front face white/silver (MLI and radiator), not an array; MLI on the front and belly": "estimate: from NASA "
        "photos of the rover (the sources list arrays on three sides only)",
    "solar array panels 1.26 x 0.50 m (sides), 1.30 x 0.50 m (aft), on the flat body faces": "estimate: panel area "
        "not found; each panel covers the flat part of its body face",
    "radiator 0.92 x 1.24 m on the aft two thirds of the deck, raised rim around the deck": "estimate: not found",
    "mast at the front left of the deck, 0.09 m square section; head 0.36 x 0.15 x 0.12 m, NavCam baseline 0.24 m, "
    "6 LEDs, an aft-looking camera": "estimate: position and sizes read from NASA photos; dimensions not published",
    "high-gain dish diameter 0.40 m, pointing low (Earth near the horizon at the pole)": "estimate: diameter not "
        "found",
    "TRIDENT position: tower at the front face, right of centre (y = -0.18 m), footpad 0.36 m above ground, rail top "
    "about 1.66 m": "estimate: placement read from NASA renders and integration photos (no published coordinates); "
        "the published facts are the belly integration and the 1 m stroke. The bay inside the body is not modelled: "
        "the stowed rail stands on the front face and the auger reaches below the belly",
    "TRIDENT rail 1.30 m, auger 25 mm, flight 42 mm, footpad 0.12 m": "estimate: sized for the 1 m stroke; not "
        "published",
    "NSS box 0.12 x 0.30 x 0.20 m low on the front face; NIRVSS on the keel front near the drill; MSolo under the "
    "keel": "estimate: sizes not found; positions follow the published 'front' and 'under the rover'",
    "mass split (per corner: wheel+drive 12 kg, steering 8 kg, arm 10 kg; chassis 330 kg, of which drill 19.8 kg)":
        "estimate: only the total (and the < 20 kg drill) is published; the split sets the centre of mass and "
        "inertia computed from the CAD",
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
    zb, zt = hw + 0.45, hw + 1.03                       # body belly and deck
    m_corner = {"arm": 10.0, "steer": 8.0, "wheel": w.mass}
    m_chassis = spec.total_mass - 4 * sum(m_corner.values())
    alloc = {"structure": 0.36, "avionics": 0.27, "arrays": 0.11, "radiator": 0.045, "mast": 0.075, "drill": 0.06,
             "instruments": 0.045, "susp_housings": 0.035}
    M = {k: v * m_chassis / sum(alloc.values()) for k, v in alloc.items()}
    S = M["structure"]
    # body: a box with chamfered vertical edges and a chamfered top edge; the arrays sit flush on its flat faces
    bx0, bx1, by, ce, ct, cb = -0.74, 0.70, 0.74, 0.07, 0.07, 0.03
    L, Wd, xm = bx1 - bx0, 2 * by, 0.5 * (bx0 + bx1)
    cad.add("chassis_body", "chassis", extrude_polygon(chamfered_rect(L, Wd, ce), zb, zt, ct, cb, t=(xm, 0, 0)),
            "white_paint", S * 0.62)
    cad.add("chassis_belly_mli", "chassis", chamfered_box((L - 0.08, Wd - 0.08, 0.016), (xm, 0, zb - 0.008), ce),
            "mli_gold", S * 0.05)
    # keel under the body: carries the suspension actuators, houses the avionics (between the wheels)
    kz0 = zb - 0.016 - 0.15
    cad.add("chassis_keel", "chassis", chamfered_box((1.20, 0.40, 0.15), (0.0, 0.0, kz0 + 0.075), 0.04, 0.0, 0.03),
            "white_paint", S * 0.19)
    cad.add("avionics", "chassis", box((0.90, 0.80, 0.25), (xm, 0.0, zb + 0.16)), "mli_gold", M["avionics"])
    # front face: silver MLI, not an array
    zf0, zf1 = zb + cb, zt - ct                         # flat part of the vertical faces
    zfc, hf = 0.5 * (zf0 + zf1), zf1 - zf0
    zsplit = zf0 + 0.42 * hf                            # silver MLI below, a white radiator panel above
    cad.add("front_mli", "chassis", _place(mli_blanket(Wd - 2 * ce - 0.02, zsplit - zf0 - 0.01, 0.012, 0.005, seed=3),
                                           rot((0, 0, 1), math.pi / 2), (bx1 + 0.006, 0.0, 0.5 * (zf0 + zsplit))),
            "mli_silver", S * 0.02)
    cad.add("front_radiator", "chassis", chamfered_box((0.012, Wd - 2 * ce - 0.02, zf1 - zsplit - 0.01),
                                                       (bx1 + 0.006, 0.0, 0.5 * (zsplit + zf1)), 0.0, 0.004, 0.004),
            "radiator", S * 0.02)
    # solar arrays (vertical: the Sun stays low at the pole) on the left, right and aft faces
    th = 0.025
    for nm, R, t, pw in (("array_L", rot((0, 0, 1), math.pi), (xm, by + th / 2 + 0.002, zfc), L - 2 * ce - 0.04),
                         ("array_R", np.eye(3), (xm, -by - th / 2 - 0.002, zfc), L - 2 * ce - 0.04),
                         ("array_aft", rot((0, 0, 1), -math.pi / 2), (bx0 - th / 2 - 0.002, 0.0, zfc),
                          Wd - 2 * ce - 0.04)):
        cad.add_assembly(nm, "chassis", solar_panel(pw, hf - 0.04, thickness=th, cells=(10, 4)), M["arrays"] / 3,
                         R=R, t=t)
    # deck: raised rim on the top face, radiator over the aft two thirds, an MLI-covered box at the front
    tx0, tx1, ty = bx0 + ct, bx1 - ct, by - ct           # top face (inset by the top chamfer)
    rim = [box_beam((tx0, s * (ty - 0.0125), zt + 0.02), (tx1, s * (ty - 0.0125), zt + 0.02), (0.025, 0.04))
           for s in (-1, 1)]
    rim += [box_beam((x, -ty + 0.025, zt + 0.02), (x, ty - 0.025, zt + 0.02), (0.025, 0.04))
            for x in (tx0 + 0.0125, tx1 - 0.0125)]
    cad.add("deck_rim", "chassis", merge(rim), "white_paint", S * 0.04)
    rx0, rx1 = tx0 + 0.04, 0.30
    cad.add_assembly("radiator", "chassis", radiator(rx1 - rx0, 2 * ty - 0.08, fins=18, fin_height=0.02),
                     M["radiator"], t=(0.5 * (rx0 + rx1), 0.0, zt + 0.022))
    cad.add("deck_box", "chassis", chamfered_box((0.24, 0.80, 0.09), (0.47, -0.22, zt + 0.045), 0.03, 0.015),
            "mli_silver", S * 0.06)
    # mast at the front left, pan-tilt head with NavCams, AftCam and LEDs; HGA on a gimbal, LGA at the aft
    mx, my = 0.50, 0.42
    head_h = 0.20                                       # pan drum to the top of the head (pan_tilt_head defaults)
    mast_top = 2.50 - head_h
    cad.add("mast_foot", "chassis", chamfered_box((0.20, 0.20, 0.08), (mx, my, zt + 0.04), 0.04, 0.02),
            "white_paint", M["mast"] * 0.08)
    cad.add("mast", "chassis", box_beam((mx, my, zt + 0.06), (mx, my, mast_top), (0.10, 0.10), (0.075, 0.075),
                                        up=(1, 0, 0), chamfer=0.2), "white_paint", M["mast"] * 0.30)
    cad.add_assembly("mast_head", "chassis", pan_tilt_head(), M["mast"] * 0.30, t=(mx, my, mast_top))
    g = np.array([mx - 0.10, my - 0.22, 2.24])          # HGA gimbal
    cad.add("hga_boom", "chassis", box_beam((mx, my, 2.21), g + [0, 0, -0.03], (0.04, 0.04), up=(0, 0, 1)),
            "white_paint", M["mast"] * 0.05)
    cad.add_assembly("hga_gimbal", "chassis", actuator(g + [0, 0, -0.05], g + [0, 0, 0.0], 0.035), M["mast"] * 0.06)
    ax = np.array([-0.90, -0.15, 0.40])
    ax /= np.linalg.norm(ax)
    dc = g + 0.07 * ax
    cad.add("hga_yoke", "chassis", cylinder(g, dc, 0.025, 16), "anodized_black", M["mast"] * 0.02)
    cad.add("hga_dish", "chassis", paraboloid_dish(0.40, 0.06, 0.006, dc, ax), "white_paint", M["mast"] * 0.12)
    cad.add("hga_feed", "chassis", cylinder(dc, dc + 0.16 * ax, (0.012, 0.008), 12), "aluminium", M["mast"] * 0.02)
    cad.add("hga_feed_horn", "chassis", cylinder(dc + 0.16 * ax, dc + 0.19 * ax, (0.012, 0.03), 16), "white_paint",
            M["mast"] * 0.01)
    cad.add("lga", "chassis", merge([cylinder((-0.58, -0.55, zt + 0.04), (-0.58, -0.55, zt + 0.20), 0.018, 16),
                                      cylinder((-0.58, -0.55, zt + 0.20), (-0.58, -0.55, zt + 0.27), (0.05, 0.012),
                                               20)]),
            "white_paint", M["mast"] * 0.04)
    # TRIDENT drill tower on the front face, right of centre (stowed: the auger below the belly)
    dx, dy, dz = bx1 + 0.10, -0.18, 0.36
    cad.add_assembly("trident", "chassis", drill_assembly(rail_offset=0.08), M["drill"] * 0.92, t=(dx, dy, dz))
    for k, z in enumerate((zb + 0.08, zt - 0.10)):
        cad.add(f"trident_bracket{k}", "chassis", chamfered_box((0.07, 0.22, 0.05), (bx1 + 0.02, dy, z), 0.01),
                "aluminium", M["drill"] * 0.04)
    # instruments: NSS low on the front face; NIRVSS on the keel front, looking at the drill cuttings; MSolo under
    cad.add("NSS", "chassis", chamfered_box((0.12, 0.30, 0.20), (bx1 + 0.072, 0.32, zb + 0.15), 0.02, 0.015),
            "white_paint", M["instruments"] * 0.35)
    cad.add("NSS_detectors", "chassis", box((0.012, 0.24, 0.14), (bx1 + 0.137, 0.32, zb + 0.15)), "mli_gold",
            M["instruments"] * 0.05)
    nir = np.array([0.66, -0.02, kz0 + 0.06])
    cad.add("NIRVSS", "chassis", chamfered_box((0.12, 0.20, 0.10), nir, 0.015, 0.01), "anodized_black",
            M["instruments"] * 0.3)
    cad.add("NIRVSS_lens", "chassis", cylinder(nir + [0.06, 0.0, -0.01], nir + [0.075, 0.0, -0.03], 0.025, 20), "lens",
            M["instruments"] * 0.05)
    cad.add("MSolo", "chassis", chamfered_box((0.22, 0.16, 0.08), (0.30, 0.04, kz0 - 0.04), 0.02, 0.0, 0.01),
            "white_paint", M["instruments"] * 0.25)
    # hazard cameras (one per wheel, under the body corners) and hazard lights on the keel sides and corners
    for c in spec.corners:
        sx, sy = c.front, c.side
        p = np.array([0.60 * sx + xm, 0.56 * sy, zb - 0.05])
        cad.add(f"hazcam_{c.name}", "chassis", chamfered_box((0.08, 0.07, 0.06), p, 0.012), "anodized_black", 0.0)
        cad.add(f"hazcam_lens_{c.name}", "chassis", cylinder(p + [0, 0.035 * sy, 0], p + [0, 0.045 * sy, -0.005],
                                                             0.016, 16), "lens", 0.0)
        cad.add(f"hazlight_{c.name}", "chassis", box((0.10, 0.012, 0.025), (0.22 * sx, 0.206 * sy, kz0 + 0.04)),
                "led", 0.0)
        cad.add(f"cornerlight_{c.name}", "chassis",
                box((0.06, 0.012, 0.025), (xm + sx * (L / 2 - ce / 2), sy * (by - ce / 2), zb + 0.05),
                    rot((0, 0, 1), sx * sy * math.pi / 4)), "led", 0.0)
        _corner_cad(cad, c, w, m_corner, M["susp_housings"] / 4, zb)
    return cad


def _place(mesh, R, t):
    return mesh[0] @ np.asarray(R).T + np.asarray(t, float), mesh[1]


def _corner_cad(cad: VehicleCAD, c: CornerModule, w: Wheel, m: dict, m_housing: float, zb: float) -> None:
    P, K, C = c.susp_pivot, c.steer_origin, c.wheel_centre
    sx, sy = c.front, c.side
    ex = np.array([1.0, 0.0, 0.0])
    # suspension actuator (fixed to the chassis keel), output at the pivot, motor toward the middle of the rover
    a0, a1 = P - sx * 0.03 * ex, P - sx * 0.16 * ex
    cad.add_assembly(f"susp_act_{c.name}", "chassis", actuator(a0, a1, 0.08, cap_radius=0.055, cap_length=0.05),
                     m_housing * 0.85)
    mid = 0.5 * (a0 + a1)
    cad.add(f"susp_mount_{c.name}", "chassis", chamfered_box((0.10, 0.10, zb - P[2]), (mid[0], P[1] - 0.02 * sy,
                                                                                       0.5 * (zb + P[2])), 0.02),
            "white_paint", m_housing * 0.15)
    # arm: a chunky tapered box-section swing arm from the pivot hub to the steering head
    hub0, hub1 = P - sx * 0.035 * ex, P + sx * 0.085 * ex
    cad.add(f"arm_{c.name}_hub", f"arm_{c.name}", cylinder(hub0, hub1, 0.068, 28), "white_paint", m["arm"] * 0.25)
    root = P + sx * 0.025 * ex
    tip = K + 0.02 * (K - P) / np.linalg.norm(K - P)
    cad.add(f"arm_{c.name}_beam", f"arm_{c.name}", box_beam(root, tip, (0.17, 0.13), (0.12, 0.06), up=(0, 0, 1),
                                                            chamfer=0.22), "white_paint", m["arm"] * 0.55)
    cad.add(f"arm_{c.name}_clamp", f"arm_{c.name}", cylinder(K + [0, 0, -0.03], K + [0, 0, 0.045], 0.078, 28),
            "anodized_black", m["arm"] * 0.20)
    # steering module: actuator (BLDC + strain-wave gear) above the column, column down to the wheel knuckle
    cad.add_assembly(f"steer_{c.name}_act", f"steer_{c.name}", actuator(K + [0, 0, 0.045], K + [0, 0, 0.075], 0.06,
                                                                        cap_radius=0.045, cap_length=0.035),
                     m["steer"] * 0.35)
    bot = np.array([K[0], K[1], C[2] - 0.05])
    cad.add(f"steer_{c.name}_column", f"steer_{c.name}", box_beam(K + [0, 0, -0.03], bot, (0.08, 0.08), (0.07, 0.07),
                                                                  up=(0, 1, 0), chamfer=0.2), "white_paint",
            m["steer"] * 0.35)
    yk = K[1] + sy * 0.045
    kn = chamfered_box((0.17, 0.03, 0.17), (C[0], yk, C[2]), 0.04)
    cad.add(f"steer_{c.name}_knuckle", f"steer_{c.name}", kn, "white_paint", m["steer"] * 0.30)
    # wheel with its drive actuator in the hub
    parts = rover_wheel(w.radius, w.width, grousers=w.grousers.count, grouser_height=w.grousers.height,
                        grouser_thickness=w.grousers.thickness, grouser_shape=w.grousers.shape,
                        chevron_angle_deg=w.grousers.chevron_angle_deg or 20.0, spokes=10, spoke_width=0.035,
                        hub_radius=0.075, hub_length=0.14, inner_side=-sy)
    cad.add_assembly(f"wheel_{c.name}", f"wheel_{c.name}", parts, m["wheel"], t=C)


__all__ = ["viper", "viper_cad", "VIPER_SOURCES"]
