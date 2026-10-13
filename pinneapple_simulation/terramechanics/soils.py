"""Soil database for wheel-soil models: Bekker, Janosi-Hanamoto, Wong-Reece, slip-sinkage and unloading parameters.

Every entry carries, per parameter, where the number comes from and how far it was checked (``Soil.sources``):

* ``"secondary"`` -- the value as another publication or open dataset reports it, with the original citation; it
  was not checked against the original table here (the original was not readable from this build environment).
* ``"estimate"`` -- no published value was found; the number is a stated engineering estimate and the reason is
  given. Change it with ``dataclasses.replace(soil, ...)`` when you have a measurement.

Units: c [Pa], phi [deg], K and K_y [m], Bekker k_c [N/m^(n+1)], k_phi [N/m^(n+2)], rho [kg/m^3], k_u0 [N/m^3],
A_u [N/m^4]. The databases quoted in kN units are converted (x 1000).

Entries: ``"GRC-1"`` (NASA Glenn lunar terramechanics simulant), ``"JSC-1A"`` (lunar mare simulant),
``"Ding2011-TYII"`` (the planetary simulant of the Ding et al. 2011 single-wheel tests, used for validation),
``"FJS-1/Hurrell2025"`` (the FJS-1 state of the Rashid-1 wheel tests) and ``"KLS-1"`` (Wang & Han 2016 tests), both used
for validation, and
``"Apollo-lunar"`` (the values the library used before for "GRC-1", kept for reproducibility). See
``docs/core_concepts/lunar_rover_terramechanics.md`` for the full reference list.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from ..particle_dynamics.terramechanics import SoilParams

ROVERDEVKIT = ("RoverDevKit soil catalogue, data/soil_simulants.csv, https://github.com/Coordinated-Autonomous-"
               "Systems-Lab/roverdevkit (MIT licence), commit ec0b198")
ORAVEC_2010 = ("Oravec, H.A., Zeng, X., Asnani, V.M. (2010). Design and characterization of GRC-1: a soil for lunar "
               "terramechanics testing in Earth-ambient conditions. J. Terramechanics 47(6), 361-377. "
               "doi:10.1016/j.jterra.2010.04.006")
HU_2025 = ("Hu, W. et al. (2025). Using physics-based simulation towards eliminating empiricism in extraterrestrial "
           "terramechanics applications. J. Field Robotics, doi:10.1002/rob.22597; arXiv:2405.11001 (GRC-1 states "
           "1660 / 1730 / 1760 kg/m^3 with friction angles 33.4 / 37.2 / 38.4 deg, after Oravec et al. 2010)")
ZENG_2010 = ("Zeng, X., He, C., Oravec, H., Wilkinson, A., Agui, J., Asnani, V. (2010). Geotechnical properties of "
             "JSC-1A lunar soil simulant. J. Aerospace Engineering 23(2), 111-116. "
             "doi:10.1061/(ASCE)AS.1943-5525.0000014")
DING_2011 = ("Ding, L., Gao, H., Deng, Z., Nagatani, K., Yoshida, K. (2011). Experimental study and analysis on "
             "driving wheels' performance for planetary exploration rovers moving in deformable soil. "
             "J. Terramechanics 48(1), 27-45. doi:10.1016/j.jterra.2010.08.001")
DING_2010 = ("Ding, L., Gao, H., Deng, Z., Tao, J. (2010). Wheel slip-sinkage and its prediction model of lunar rover. "
             "J. Central South University of Technology 17, 129-135. doi:10.1007/s11771-010-0021-7")
HURRELL_2025 = ("Hurrell et al. (2025). Traction performance evaluation for a Rashid-1 rover wheel. Space Science "
                "Reviews 221, article 37 (open access)")
WONG_2008 = "Wong, J.Y. (2008). Theory of Ground Vehicles, 4th ed. Wiley."
WONG_REECE_1967 = ("Wong, J.Y., Reece, A.R. (1967). Prediction of rigid wheel performance based on the analysis of "
                   "soil-wheel stresses, part I. J. Terramechanics 4(1), 81-98.")
ISHIGAMI_2007 = ("Ishigami, G., Miwa, A., Nagatani, K., Yoshida, K. (2007). Terramechanics-based model for steering "
                 "maneuver of planetary exploration rovers on loose soil. J. Field Robotics 24(3), 233-250. "
                 "doi:10.1002/rob.20187")

_EST_K = ("estimate: no Janosi-Hanamoto K found for this simulant; 0.018 m, the value the library used before, is of "
          "the order Ding et al. 2011 report for their planetary simulant (9.7-13.1 mm, as relayed by RoverDevKit)")
_EST_WR = ("estimate: Wong-Reece theta_m = (a0 + a1 s) theta_f coefficients were not found for this simulant; "
           "a0 = 0.40, a1 = 0.15 are the library's previous values, within the range reported for sands by "
           "Wong & Reece (1967)")
_EST_C3 = "estimate: exit angle theta_r = -c3 theta_f with c3 = 1/3, the library's previous value (not measured)"
_EST_N1 = "estimate: no slip-sinkage coefficient n1 was found for this simulant; 0 switches the effect off"
# n1 for the two lunar simulants used by the vehicle examples: no measurement exists, so the slip-sinkage exponent
# gradient is taken from the five single-wheel calibrations of testbed.calibrate_holdout (fitted n1 = 0.64-0.94)
_SLIP_SINKAGE_N1 = 0.8
_EST_N1_LUNAR = ("estimate: no slip-sinkage coefficient was published for this simulant; n1 = 0.8 is the rounded mean "
                 "of n1 fitted (n = n0 + n1 |s|, Ding et al. 2010, 2011) to the training points of the five bundled "
                 "single-wheel test series (0.64-0.94, testbed.calibrate_holdout); it makes the sinkage grow with "
                 "slip as all those tests do")
_EST_KU = ("estimate: no unloading-reloading modulus (Wong 2008, k_u = k_u0 + A_u z_u) was found for any lunar "
           "simulant; k_u0 = 2e8 N/m^3, A_u = 0 gives an elastic rebound of a few tenths of a millimetre, i.e. "
           "almost all of the sinkage is plastic, as observed for loose granular soils")
_EST_KY = "estimate: lateral shear modulus K_y taken equal to K (Ishigami et al. 2007 fit K_y separately)"


@dataclass(frozen=True)
class Soil:
    """Soil parameters with provenance; see the module docstring for units and the meaning of ``sources``."""
    name: str
    c: float                    # cohesion [Pa]
    phi_deg: float              # internal friction angle [deg]
    K: float                    # Janosi-Hanamoto shear deformation modulus [m]
    k_c: float                  # Bekker cohesive modulus [N/m^(n+1)]
    k_phi: float                # Bekker frictional modulus [N/m^(n+2)]
    n0: float                   # sinkage exponent at zero slip [-]
    rho: float                  # bulk density [kg/m^3]
    n1: float = 0.0             # slip-sinkage: n = n0 + n1 |s| (Ding et al. 2011)
    a0: float = 0.40            # Wong-Reece theta_m = (a0 + a1 s) theta_f
    a1: float = 0.15
    c3: float = 1.0 / 3.0       # exit angle theta_r = -c3 theta_f
    Ky: float | None = None     # lateral shear modulus [m] (None = K)
    k_u0: float = 2.0e8         # unloading-reloading modulus k_u = k_u0 + A_u z_u [N/m^3]
    A_u: float = 0.0            # [N/m^4]
    description: str = ""
    sources: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self):
        if self.Ky is None:
            object.__setattr__(self, "Ky", self.K)
        for k in ("K", "n0", "rho", "Ky"):
            if not getattr(self, k) > 0:
                raise ValueError(f"Soil.{k} must be > 0")
        if not 0 < self.phi_deg < 90:
            raise ValueError("Soil.phi_deg must be in (0, 90)")
        if self.c < 0 or not self.k_phi > 0:
            raise ValueError("Soil.c must be >= 0 and k_phi > 0 (k_c may be negative, as some bevameter fits give)")

    @property
    def tan_phi(self) -> float:
        return math.tan(math.radians(self.phi_deg))

    @property
    def n(self) -> float:
        return self.n0

    def with_(self, **kw) -> Soil:
        """A copy with some parameters changed; changed parameters are marked ``"user"`` in ``sources``."""
        src = dict(self.sources)
        for k in kw:
            src[k] = "user"
        return replace(self, sources=src, **kw)

    def to_soil_params(self, g: float = 1.62) -> SoilParams:
        """The legacy :class:`SoilParams` (zero-slip exponent) for ``BekkerWongSolver`` and the rover module."""
        return SoilParams(c=self.c, phi_deg=self.phi_deg, K=self.K, k_c=self.k_c, k_phi=self.k_phi, n=self.n0,
                          rho=self.rho, g=g, a0=self.a0, a1=self.a1)

    def provenance(self) -> list[tuple[str, float, str]]:
        """(parameter, value, source) for every parameter, for reports."""
        keys = ("c", "phi_deg", "K", "Ky", "k_c", "k_phi", "n0", "n1", "rho", "a0", "a1", "c3", "k_u0", "A_u")
        return [(k, getattr(self, k), self.sources.get(k, "not stated")) for k in keys]


def _common_estimates() -> dict:
    return {"a0": _EST_WR, "a1": _EST_WR, "c3": _EST_C3, "n1": _EST_N1, "k_u0": _EST_KU, "A_u": _EST_KU, "Ky": _EST_KY}


SOILS: dict[str, Soil] = {}


def register_soil(soil: Soil) -> Soil:
    SOILS[soil.name] = soil
    return soil


register_soil(Soil(
    "GRC-1", c=250.0, phi_deg=38.0, K=0.018, k_c=700.0, k_phi=505_000.0, n0=0.8, rho=1620.0, n1=_SLIP_SINKAGE_N1,
    description="NASA Glenn GRC-1 lunar terramechanics simulant (cohesionless, frictional)",
    sources={"c": f"secondary: 0.25 kPa as listed in the {ROVERDEVKIT}, attributed to {ORAVEC_2010}; "
                  "Oravec et al. describe GRC-1 as essentially cohesionless",
             "phi_deg": f"secondary: 38 deg ({ROVERDEVKIT}, attributed to Oravec et al. 2010); within the "
                        f"33.4-38.4 deg range of {HU_2025}",
             "k_c": f"secondary: 0.7 kN/m^(n+1) ({ROVERDEVKIT}, attributed to Oravec et al. 2010)",
             "k_phi": f"secondary: 505 kN/m^(n+2) ({ROVERDEVKIT}, attributed to Oravec et al. 2010)",
             "n0": f"secondary: 0.8 ({ROVERDEVKIT}, attributed to Oravec et al. 2010)",
             "rho": f"secondary: 1620 kg/m^3 ({ROVERDEVKIT}); {HU_2025} test at 1660-1760 kg/m^3",
             "K": _EST_K, **_common_estimates(), "n1": _EST_N1_LUNAR}))

register_soil(Soil(
    "JSC-1A", c=1000.0, phi_deg=45.0, K=0.018, k_c=1400.0, k_phi=820_000.0, n0=1.0, rho=1600.0, n1=_SLIP_SINKAGE_N1,
    description="JSC-1A lunar mare simulant (Orbitec), compacted",
    sources={"c": f"secondary: 1.0 kPa ({ROVERDEVKIT}, attributed to {ZENG_2010}); reported cohesion ranges from "
                  "0 to 1 kPa as the density rises from 1.62 to 1.96 g/cm^3",
             "phi_deg": f"secondary: 45 deg ({ROVERDEVKIT}, attributed to Zeng et al. 2010); triaxial peak "
                        "friction angles of 42.9-48.8 deg depending on density are reported for JSC-1A",
             "k_c": f"secondary: 1.4 kN/m^(n+1) ({ROVERDEVKIT}, attributed to Zeng et al. 2010). Caution: k_c, "
                    "k_phi and n equal the classic Apollo lunar-soil set (Bekker; Wong 2008), so they are probably "
                    "not a JSC-1A bevameter measurement",
             "k_phi": f"secondary: 820 kN/m^(n+2) ({ROVERDEVKIT}); same caution as k_c",
             "n0": f"secondary: 1.0 ({ROVERDEVKIT}); same caution as k_c",
             "rho": f"secondary: 1600 kg/m^3 ({ROVERDEVKIT})",
             "K": _EST_K, **_common_estimates(), "n1": _EST_N1_LUNAR}))

register_soil(Soil(
    "Ding2011-TYII", c=250.0, phi_deg=31.9, K=0.011, k_c=15_600.0, k_phi=2_407_400.0, n0=1.10, rho=1605.0,
    description="Planetary soil simulant of the Ding et al. (2011) single-wheel tests (validation soil)",
    sources={**{k: f"secondary: {DING_2011}, Table 2, as relayed by the {ROVERDEVKIT}"
                for k in ("c", "phi_deg", "k_c", "k_phi", "n0", "rho")},
             "K": f"secondary: K = 9.7-13.1 mm reported by {DING_2011} (as relayed by RoverDevKit); mid value",
             **_common_estimates()}))

register_soil(Soil(
    "FJS-1/Hurrell2025", c=2400.0, phi_deg=38.0, K=0.018, k_c=1370.0, k_phi=820_000.0, n0=1.0, rho=1740.0,
    description="FJS-1 in the state of the Rashid-1 wheel tests (validation soil, pressure-sinkage proxied)",
    sources={"c": f"secondary: 2.4 kPa, {HURRELL_2025} Table 2 (via Ozaki et al. 2023), as relayed by RoverDevKit",
             "phi_deg": f"secondary: angle of repose 38 deg, {HURRELL_2025}, used as friction angle",
             "k_c": "secondary: catalogue FJS-1 value (Kanamori et al. 1998) as a proxy -- Hurrell et al. do not "
                    "report pressure-sinkage parameters",
             "k_phi": "secondary: catalogue FJS-1 proxy (Kanamori et al. 1998), see k_c",
             "n0": "secondary: catalogue FJS-1 proxy (Kanamori et al. 1998), see k_c",
             "rho": f"secondary: {ROVERDEVKIT}",
             "K": "secondary: 0.018 m used by RoverDevKit for this case", **_common_estimates()}))

register_soil(Soil(
    "KLS-1", c=1716.0, phi_deg=40.6, K=0.018, k_c=-44_055.4, k_phi=3_581_810.6, n0=1.2594, rho=1600.0,
    description="Korean lunar simulant KLS-1, relative density about 60 % (validation soil)",
    sources={"c": f"secondary: 1.716 kPa, Wang & Han (2016) J. Korean Geotech. Soc. 32(11) Table 2, as relayed by "
                  f"the {ROVERDEVKIT}",
             "phi_deg": "secondary: 40.6 deg, Wang & Han (2016) Table 2, as relayed by RoverDevKit",
             **{k: "secondary: bevameter fit of Lim et al. (2021) J. Astron. Space Sci. 38(4):237, as relayed by "
                   "RoverDevKit (negative k_c is a least-squares artefact; k_c/b + k_phi stays positive)"
                for k in ("k_c", "k_phi", "n0")},
             "rho": "secondary: RoverDevKit", "K": "secondary: 0.018 m used by RoverDevKit for this case",
             **_common_estimates()}))

register_soil(Soil(
    "Apollo-lunar", c=1400.0, phi_deg=30.0, K=0.018, k_c=1370.0, k_phi=814_000.0, n0=1.0, rho=1650.0,
    description="The parameters SoilParams() carries (labelled GRC-1 before this database existed)",
    sources={k: "library default of pinneapple_simulation.particle_dynamics.terramechanics.SoilParams, source "
                "not recorded" for k in ("c", "phi_deg", "K", "k_c", "k_phi", "n0", "rho")} | _common_estimates()))


def get_soil(name: str) -> Soil:
    try:
        return SOILS[name]
    except KeyError:
        raise KeyError(f"unknown soil {name!r}; available: {', '.join(SOILS)}") from None


def list_soils() -> list[str]:
    return list(SOILS)


__all__ = ["Soil", "SOILS", "get_soil", "list_soils", "register_soil", "ORAVEC_2010", "ZENG_2010", "DING_2011",
           "HURRELL_2025", "DING_2010", "WONG_2008", "WONG_REECE_1967", "ISHIGAMI_2007", "HU_2025", "ROVERDEVKIT"]
