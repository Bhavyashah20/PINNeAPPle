"""Vortex-lattice aerodynamics of the library airliner plus a friction-drag estimate from its mesh.

    python examples/geometry_library/03_airliner_vlm.py

Rebuilds the ``airliner_a320`` entry's aircraft (the same parameters as ``pp.geometries.build``), solves the
vortex lattice of wing + horizontal tail (``pinneapple_design.aero.vlm``): lift slope, neutral point, trim at a
cruise lift coefficient, induced drag. The wetted area of the library mesh (fuselage, wing, tails, nacelles) gives
a turbulent flat-plate friction drag with form factors at Mach 0.78, 11 000 m: the zero-lift drag, and L/D.
Published values for comparison (A320 class): CL_alpha about 5.5-6 /rad, cruise L/D about 16-18. Seconds.
"""
import math

import pinneapple as pp
from pinneapple_design.aero.airliner import Airliner
from pinneapple_design.aero.vlm import solve
from pinneapple_design.geometry_library import meshops

g = pp.geometries.build("airliner_a320")
p = g.params
ac = Airliner(length=p["length"], wing_area=p["wing_area"], aspect_ratio=p["aspect_ratio"], taper=p["taper"],
              sweep_le=p["sweep_le"], dihedral=p["dihedral"], twist=p["twist"], fan_diameter=p["fan_diameter"],
              engine_station=p["engine_station"], sharklets=p["sharklets"])
x_ref = ac.mac_le_x + 0.25 * ac.mac
sol = solve(ac, x_ref)
print(g)
print(f"span {ac.span:.2f} m, S {ac.wing_area} m^2, MAC {ac.mac:.2f} m, reference x {x_ref:.2f} m")
print(f"CL_alpha {sol.CL[1]:.2f} /rad, neutral point x = {sol.neutral_point:.2f} m "
      f"(wing + tail only: the fuselage's destabilising moment is not in the lattice)")

# cruise: ISA 11 000 m, Mach 0.78, mass 64 t
T, rho, a_snd = 216.65, 0.3639, 295.07
V = 0.78 * a_snd
q = 0.5 * rho * V * V
CL = 64000 * 9.81 / (q * ac.wing_area)
alpha, it = sol.trim(CL)
c = sol.coefficients(alpha, it)
# friction drag from the library mesh: wetted area per component, Re on its length, form factors (Raymer)
mu = 1.458e-6 * T ** 1.5 / (T + 110.4)
groups = {"fuselage": (1.10, p["length"]), "wing": (1.35, ac.mac), "htail": (1.30, 0.7 * ac.mac),
          "vtail": (1.30, ac.mac), "nacelle": (1.25, 4.3), "sharklet": (1.35, 1.0), "pylon": (1.3, 3.0)}
CD0 = 0.0
for b in g.bodies:
    key = next((k for k in groups if b.name.startswith(k)), None)
    if key is None:
        continue
    k_form, Lref = groups[key]
    Re = rho * V * Lref / mu
    cf = 0.455 / (math.log10(Re) ** 2.58 * (1 + 0.144 * 0.78 ** 2) ** 0.65)
    CD0 += k_form * cf * meshops.area(b.vertices, b.faces) / ac.wing_area
# the bodies overlap (wing roots inside the fuselage, pylons in the nacelles): the wetted area is the area of
# their boolean union (Geometry.solid(), manifold3d), so scale the component sum to it
try:
    from pinneapple_design.geometry_library import meshops as _m
    S_wet = _m.area(*g.solid())
except ImportError:
    S_wet = g.area()
CD0 *= S_wet / g.area() * 1.08                                   # true wetted area; interference, excrescences
CD = CD0 + c["CDi"] / 0.85                                       # span efficiency of the real wing vs the VLM
print(f"cruise M 0.78 FL360, 64 t: CL {CL:.3f}, alpha {math.degrees(alpha):.2f} deg, tail incidence "
      f"{math.degrees(it):.2f} deg")
print(f"wetted area {S_wet:.0f} m^2 (sum of the body areas {g.area():.0f} m^2), CD0 {CD0:.4f}, CDi {c['CDi']:.4f}, CD {CD:.4f}, L/D {CL / CD:.1f}")
