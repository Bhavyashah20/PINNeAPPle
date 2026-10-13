"""The first entries: the airliner, the road car, the Formula car, the two-stage launcher and NASA VIPER, wrapped
from the models that already exist in the repository (their classes stay where they are and keep working)."""
from __future__ import annotations

import numpy as np

from ..core import FRAME_VEHICLE, GeometrySpec, Param, Source, register
from ._adapt import bodies_from_items, close_sheets, drop_degenerate

AIRBUS_AC = ("Airbus (2023). A320 Aircraft Characteristics - Airport and Maintenance Planning, Rev. 41, section 2-2 "
             "(general dimensions). https://aircraft.airbus.com/en/customer-care/fleet-wide-care/airport-operations-"
             "and-technical-data/aircraft-characteristics")


def _aero_materials():
    from pinneapple_design.aero import airliner  # noqa: F401  (adds the airliner materials)
    from pinneapple_design.aero.airframe import MATERIALS
    return MATERIALS


def _vehicle_materials():
    from pinneapple_design.geometry import vehicles3d
    out = dict(vehicles3d.VEHICLE_MATERIALS)
    return out





# ============================================================================================ airliner
def _airliner(p, res):
    from pinneapple_design.aero.airliner import Airliner
    ac = Airliner(length=p["length"], wing_area=p["wing_area"], aspect_ratio=p["aspect_ratio"], taper=p["taper"],
                  sweep_le=p["sweep_le"], dihedral=p["dihedral"], twist=p["twist"], fan_diameter=p["fan_diameter"],
                  engine_station=p["engine_station"], sharklets=p["sharklets"])
    parts = _airliner_parts(ac, 0.28 if res < 2 else 1.0)
    items = [(q.name, q.vertices, q.faces, q.material, "", q.aero) for q in parts]
    return close_sheets(drop_degenerate(bodies_from_items(items)))


def _airliner_parts(ac, n: float):
    """``Airliner.build("high")`` with the resolution factor ``n`` (1.0 = "high"; the light level uses fewer
    stations so its files stay small)."""
    parts = ac._airliner_fuselage(int(440 * n), int(96 * max(n, 0.5)), windows=True)
    parts += ac._wing(int(80 * max(n, 0.4)), int(40 * max(n, 0.4)))
    parts += ac._airliner_tails(int(50 * max(n, 0.5)), int(16 * max(n, 0.5)))
    parts += ac._engines()
    if ac.sharklets:
        parts += ac._sharklets()
    for p in parts:
        if p.name.startswith("wing") and p.material == "paint":
            p.material = "wingmetal"
    return parts


register(GeometrySpec(
    name="airliner_a320", title="Single-aisle airliner (A320 class)", category="aircraft",
    description=("Circular fuselage with cabin and cockpit windows and a livery, low swept wing with dihedral, washout "
                 "and sharklets, two under-wing turbofans on pylons, swept tails sized by volume coefficient; in "
                 "cruise (gear up). Wraps pinneapple_design.aero.airliner.Airliner."),
    builder=_airliner,
    params=[
        Param("length", 37.57, "m", (30.0, 45.0), "overall fuselage length"),
        Param("wing_area", 122.6, "m^2", (100.0, 140.0), "reference wing area"),
        Param("aspect_ratio", 9.5, "-", (7.5, 12.0), "b^2 / S (span without sharklets)"),
        Param("taper", 0.24, "-", (0.15, 0.40), "tip / root chord"),
        Param("sweep_le", 28.6, "deg", (20.0, 35.0), "leading-edge sweep"),
        Param("dihedral", 5.0, "deg", (2.0, 8.0), "wing dihedral"),
        Param("twist", -4.0, "deg", (-6.0, 0.0), "tip washout"),
        Param("fan_diameter", 1.95, "m", (1.6, 2.2), "turbofan fan diameter (CFM56-5B: 1.73 m, LEAP-1A: 1.98 m)"),
        Param("engine_station", 0.34, "-", (0.25, 0.45), "engine spanwise station / half-span"),
        Param("sharklets", True, "", None, "wing-tip sharklets"),
    ],
    dimensions={"length_x": (38.75, 0.03), "width_y": (35.43, 0.03), "height_z": (10.23, 0.05)},
    frame=FRAME_VEHICLE.replace("z = 0 on the ground for ground vehicles", "z = 0 on the fuselage axis"),
    sources=[Source("length 37.57 m, span 35.80 m with sharklets (34.10 m without), fuselage width 3.95 m, wing "
                    "area 122.6 m^2, height 11.76 m on the gear", AIRBUS_AC),
             Source("LE sweep, dihedral, washout, tail volume coefficients, airfoil (NACA 2412 stand-in for the "
                    "supercritical section)", "engineering estimates from public three-views", estimate=True),
             Source("bounding box of the model: 38.75 m long (the swept horizontal-tail tips end 1.2 m behind the "
                    "37.57 m tail cone), 35.43 m span over the sharklets, 10.2 m high in cruise with the gear "
                    "retracted (11.76 m on the gear)", "model geometry", estimate=True)],
    physics=["external aerodynamics (RANS, cruise lift/drag, high-lift studies)", "vortex-lattice / panel methods",
             "wake-vortex and contrail studies", "aero-structural wing optimisation"],
    materials=_aero_materials, tags=("transport aircraft", "turbofan", "swept wing"), camera={"azimuth": 215.0, "elevation": 22.0},
    max_triangles={"light": 36000, "fine": 400000},
    self_intersection_exempt={b: "sub-millimetre crossings of the sharp trailing edge (legacy loft)" for b in ("wing_right", "wing_left", "sharklet_r", "sharklet_l")},
    legacy="pinneapple_design.aero.airliner.Airliner",
))


# ============================================================================================ road car
def _road_car(p, res):
    from pinneapple_design.geometry.vehicles3d import RoadCar
    car = RoadCar(length=p["length"], width=p["width"], roof_height=p["roof_height"], clearance=p["clearance"],
                  windshield_deg=p["windshield_deg"], slant_deg=p["slant_deg"], style=p["style"],
                  wheelbase=p["wheelbase"], wheel_radius=p["wheel_radius"],
                  n_x=int(140 * res), n_ring=int(64 * res))
    from pinneapple_design.geometry.vehicles3d import _split_by
    V, F, mats = car._body()
    n_side = 2 * (car.n_x - 1) * car.n_ring          # the two triangles of a quad get one material (no saw teeth)
    m = np.asarray(mats, dtype=object).copy()
    pair = m[0:n_side:2].copy()
    odd = m[1:n_side:2]
    keep_odd = (odd == "car_glass") & (pair != "car_glass")      # glass wins, so windows keep their full size
    pair[keep_odd] = odd[keep_odd]
    m[0:n_side:2] = pair
    m[1:n_side:2] = pair
    parts = _split_by(V, F, m, "body")
    for xc, yc in car.wheel_positions():
        parts += car._wheel(xc, yc)
    items = [(q.name, q.vertices, q.faces, q.material, q.group if q.group != "wheels" else q.name.rsplit("_", 1)[0],
              q.aero) for q in parts]
    return close_sheets(drop_degenerate(bodies_from_items(items)))


register(GeometrySpec(
    name="road_car", title="Four-door road car (fastback / notchback)", category="ground_vehicle",
    description=("Mid-size four-door saloon: super-elliptic body with glazing, lights, trim and wheel arches, four "
                 "tyres with rims, slant / rear-window angle as the main aerodynamic parameter (Ahmed-body physics on "
                 "a realistic shape). Wraps pinneapple_design.geometry.vehicles3d.RoadCar."),
    builder=_road_car,
    params=[
        Param("length", 4.6, "m", (4.0, 5.2), "overall length"),
        Param("width", 1.86, "m", (1.7, 2.0), "body width without mirrors"),
        Param("roof_height", 1.36, "m", (1.2, 1.55), "roof height above the road"),
        Param("clearance", 0.13, "m", (0.08, 0.20), "underbody ground clearance"),
        Param("windshield_deg", 28.0, "deg", (20.0, 40.0), "windscreen rake from the horizontal"),
        Param("slant_deg", 22.0, "deg", (10.0, 35.0), "fastback / rear-window angle (Ahmed critical angle ~30 deg)"),
        Param("style", "fastback", "", None, "roof style", choices=("fastback", "notchback")),
        Param("wheelbase", 2.80, "m", (2.5, 3.1), "axle distance"),
        Param("wheel_radius", 0.35, "m", (0.30, 0.40), "tyre outer radius"),
    ],
    dimensions={"length_x": (4.6, 0.02), "width_y": (1.86, 0.02), "height_z": (1.39, 0.02)},
    frame=FRAME_VEHICLE,
    sources=[Source("D-segment saloon envelope (length 4.6-4.9 m, width 1.82-1.87 m, height 1.40-1.45 m)",
                    "manufacturer data of mid-size saloons; generic shape, not a specific model", estimate=True),
             Source("slant-angle drag physics", "Ahmed, S.R., Ramm, G., Faltin, G. (1984), SAE 840300")],
    physics=["external aerodynamics with moving ground (drag, lift, wake)", "slant-angle / rear-wake studies",
             "aero-acoustics around the A-pillar", "vehicle soiling and cooling-drag surrogates"],
    materials=_vehicle_materials, tags=("automotive", "bluff body"), camera={"azimuth": 225.0, "elevation": 16.0},
    max_triangles={"light": 22000, "fine": 400000},
    legacy="pinneapple_design.geometry.vehicles3d.RoadCar",
))


# ============================================================================================ formula car
def _f1(p, res):
    from pinneapple_design.geometry.vehicles3d import FormulaCar
    car = FormulaCar(wheelbase=p["wheelbase"], ride_height=p["ride_height"], front_wing_aoa=p["front_wing_aoa"],
                     front_flap_aoa=p["front_flap_aoa"], rear_wing_aoa=p["rear_wing_aoa"],
                     rear_flap_aoa=p["rear_flap_aoa"], drs_open=p["drs_open"], diffuser_angle=p["diffuser_angle"])
    items = [(q.name, q.vertices, q.faces, q.material, q.name if q.group in ("front_wing", "rear_wing", "wheels")
              else q.group, q.aero) for q in car.parts() if q.group != "halo"]
    bodies = close_sheets(drop_degenerate(bodies_from_items(items)))
    # the halo hoop as a closed loop (FormulaCar sweeps it as an open tube whose two caps coincide)
    import math

    import numpy as np

    from .. import kit
    from ..core import Body
    t = np.linspace(-0.5 * math.pi, 1.5 * math.pi, 40, endpoint=False)
    ring = np.c_[2.30 + 0.28 * np.sin(t), 0.26 * np.cos(t), 0.84 + 0.04 * np.sin(t)]
    bodies.append(Body("halo", *kit.tube_loop(ring, 0.028, 16), "halo_ti", "halo", False))
    pillar = np.c_[np.linspace(1.86, 2.02, 6), np.zeros(6), np.linspace(0.60, 0.84, 6)]
    bodies.append(Body("halo_pillar", *kit.tube(pillar, 0.03, 16), "halo_ti", "halo", False))
    return bodies


register(GeometrySpec(
    name="f1_car", title="Open-wheel racing car (Formula 1 style)", category="ground_vehicle",
    description=("Monocoque with sidepods and engine cover, floor with diffuser, two-element front and rear wings "
                 "with endplates (DRS open or closed), driver helmet, halo and four exposed wheels. In the spirit of "
                 "the 2022+ regulations, not any team's car. Wraps pinneapple_design.geometry.vehicles3d.FormulaCar."),
    builder=_f1,
    params=[
        Param("wheelbase", 3.60, "m", (3.4, 3.6), "axle distance (2022+ technical regulations: max 3.6 m)"),
        Param("ride_height", 0.035, "m", (0.015, 0.06), "floor height above the road"),
        Param("front_wing_aoa", 4.0, "deg", (0.0, 10.0), "front main-plane angle"),
        Param("front_flap_aoa", 22.0, "deg", (10.0, 35.0), "front flap angle"),
        Param("rear_wing_aoa", 6.0, "deg", (0.0, 12.0), "rear main-plane angle"),
        Param("rear_flap_aoa", 32.0, "deg", (15.0, 45.0), "rear flap angle (DRS closed)"),
        Param("drs_open", False, "", None, "drag reduction system open"),
        Param("diffuser_angle", 13.0, "deg", (6.0, 18.0), "diffuser ramp angle"),
    ],
    dimensions={"length_x": (5.57, 0.02), "width_y": (1.97, 0.02), "height_z": (1.03, 0.02)},
    frame=FRAME_VEHICLE.replace("at the nose", "at the front-wing leading edge"),
    sources=[Source("max wheelbase 3.6 m, width 2.0 m, height 0.95 m above the reference plane, 18-inch wheels "
                    "(tyre diameter ~0.72 m)", "FIA (2022). Formula 1 Technical Regulations, articles 3 and 10"),
             Source("wing chords, sidepod and engine-cover shapes", "estimates from broadcast imagery",
                    estimate=True)],
    physics=["ground-effect aerodynamics (downforce, floor and diffuser)", "wake / dirty-air studies for following",
             "DRS open / closed drag delta", "tyre-wake interaction"],
    materials=_vehicle_materials, tags=("motorsport", "ground effect"), camera={"azimuth": 230.0, "elevation": 20.0},
    max_triangles={"light": 26000, "fine": 400000},
    legacy="pinneapple_design.geometry.vehicles3d.FormulaCar",
))


# ============================================================================================ launch vehicle
def _launcher(p, res):
    from pinneapple_design.geometry.vehicles3d import LaunchVehicle
    lv = LaunchVehicle(length=p["length"], diameter=p["diameter"], fin_span=p["fin_span"], n_engines=p["n_engines"],
                       plume_length=1.0)
    items = [(q.name, q.vertices, q.faces, q.material, "", q.aero) for q in lv.parts()
             if "flame" not in q.material]
    return close_sheets(drop_degenerate(bodies_from_items(items)))


register(GeometrySpec(
    name="launch_vehicle_two_stage", title="Two-stage launch vehicle with fins", category="space",
    description=("Two-stage launcher with a tangent-ogive payload fairing, interstage, four tail fins in + "
                 "configuration and a cluster of engine bells (plume off). Wraps "
                 "pinneapple_design.geometry.vehicles3d.LaunchVehicle (Barrowman stability included there)."),
    builder=_launcher,
    params=[
        Param("length", 42.0, "m", (25.0, 60.0), "overall length"),
        Param("diameter", 3.0, "m", (2.0, 4.0), "core diameter"),
        Param("fin_span", 2.4, "m", (0.8, 3.5), "fin span from the core"),
        Param("n_engines", 7, "-", (1, 9), "first-stage engines"),
    ],
    dimensions={"length_x": (43.5, 0.02), "width_y": (7.8, 0.02), "height_z": (7.8, 0.02)},
    frame="metres; x along the axis from the nose tip toward the engines (the free-stream direction in ascent), "
          "y and z across, fins in the x-y and x-z planes",
    sources=[Source("proportions of medium-lift launchers (fineness 12-15, fairing 1.2 x core)",
                    "generic; not a specific vehicle", estimate=True),
             Source("fin aerodynamics", "Barrowman, J. (1967). The practical calculation of the aerodynamic "
                    "characteristics of slender finned vehicles. NASA Goddard")],
    physics=["ascent aerodynamics (transonic / supersonic drag, Cp)", "static stability (Barrowman)",
             "fin flutter / aeroelastic surrogates"],
    materials=_vehicle_materials, tags=("rocket", "slender body"), camera={"azimuth": 200.0, "elevation": 14.0},
    max_triangles={"light": 36000, "fine": 400000},
    legacy="pinneapple_design.geometry.vehicles3d.LaunchVehicle",
))


# ============================================================================================ VIPER
def _viper(p, res):
    from pinneapple_simulation.terramechanics.wheel import Grousers, Wheel
    from pinneapple_simulation.vehicle.viper import viper
    w = Wheel(radius=p["wheel_radius"], width=p["wheel_width"],
              grousers=Grousers(count=p["grouser_count"], height=p["grouser_height"], thickness=0.004,
                                shape="straight", shear_radius_factor=1.0), mass=12.0)
    spec = viper(wheel=w)
    items = [(q.name, q.vertices, q.faces, q.material, "", True) for q in spec.cad.parts if q.name != "lga"]
    bodies = close_sheets(drop_degenerate(bodies_from_items(items)))
    lga = next((q for q in spec.cad.parts if q.name == "lga"), None)
    if lga is not None:            # mast + cone merged with coincident caps in viper_cad: one solid of revolution
        from .. import kit
        from ..core import Body
        x0, y0, z0 = lga.vertices[:, 0].mean(), lga.vertices[:, 1].mean(), lga.vertices[:, 2].min()
        h = float(lga.vertices[:, 2].max()) - z0              # mast 0.16 m (r 18 mm), then a cone 50 -> 12 mm
        zz = np.array([0.0, h - 0.07, h - 0.07, h])
        rr = np.array([0.018, 0.018, 0.05, 0.012])
        V, F = kit.revolve(zz, rr, 20, axis="z")
        bodies.append(Body("lga", V + [x0, y0, z0], F, lga.material, "lga", True))
    return bodies


register(GeometrySpec(
    name="viper_rover", title="NASA VIPER lunar rover", category="space",
    description=("Volatiles Investigating Polar Exploration Rover from public NASA figures: chassis with side and "
                 "aft solar arrays, deck radiator, mast with camera head, high- and low-gain antennas, TRIDENT drill, "
                 "four active-suspension corner modules with spoked, grousered wheels. Wraps "
                 "pinneapple_simulation.vehicle.viper.viper().cad."),
    builder=_viper,
    params=[
        Param("wheel_radius", 0.25, "m", (0.18, 0.32), "wheel radius to the rim (grousers excluded)"),
        Param("wheel_width", 0.20, "m", (0.12, 0.30), "wheel width"),
        Param("grouser_count", 24, "-", (0, 40), "grousers per wheel"),
        Param("grouser_height", 0.025, "m", (0.0, 0.04), "grouser height"),
    ],
    dimensions={"length_x": (1.692, 0.02), "width_y": (1.539, 0.02), "height_z": (2.50, 0.02)},
    frame="right-handed, metres: x forward, y to the left, z up; origin on the ground below the chassis centre",
    sources=[Source("1.5 m x 1.5 m x 2.5 m (mast up), ~450 kg, wheels ~0.5 m diameter, active suspension",
                    "NASA VIPER fact sheets and NTRS papers listed in pinneapple_simulation.vehicle.viper.VIPER_SOURCES"),
             Source("detail placement (antennas, hazcams, instruments)", "estimates from NASA imagery",
                    estimate=True),
             Source("bounding box of the model: 1.69 m long (the front instruments and bumpers stand proud of the "
                    "1.5 m chassis), 1.54 m wide over the wheels, 2.50 m to the top of the mast", "model geometry")],
    physics=["terramechanics (wheel-soil, slip, sinkage)", "multibody dynamics", "thermal (lunar night, radiator)",
             "lighting / perception at the lunar south pole"],
    tags=("planetary rover", "terramechanics"), camera={"azimuth": 35.0, "elevation": 20.0},
    max_triangles={"light": 26000, "fine": 400000},
    legacy="pinneapple_simulation.vehicle.viper.viper_cad",
))
