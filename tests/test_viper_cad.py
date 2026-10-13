"""Parametric NASA VIPER CAD: closed meshes, envelope, mass, layout (mast, TRIDENT drill) and steering clearance."""
import math
from collections import Counter

import numpy as np
import pytest

pytest.importorskip("scipy")

from pinneapple_design.geometry.vehicle_cad import (  # noqa: E402
    actuator,
    box_beam,
    chamfered_box,
    drill_assembly,
    extrude_polygon,
    helix_sweep,
    ladder_rail,
    mli_blanket,
    offset_polygon,
    pan_tilt_head,
    rot,
    signed_volume,
)


def _closed_and_consistent(F) -> bool:
    """Every directed edge appears once and its reverse once: a closed, consistently wound surface."""
    e = Counter()
    for a, b, c in np.asarray(F):
        for u, v in ((a, b), (b, c), (c, a)):
            e[(int(u), int(v))] += 1
    return all(n == 1 and e.get((v, u), 0) == 1 for (u, v), n in e.items())


@pytest.fixture(scope="module")
def spec():
    from pinneapple_simulation.vehicle import viper
    return viper()


def test_new_primitives_are_closed_and_outward():
    meshes = {
        "chamfered_box": chamfered_box((1.0, 0.8, 0.5), (0, 0, 1), 0.07, 0.05, 0.02),
        "extrude": extrude_polygon([[0, 0], [1, 0], [1.2, 0.5], [0.3, 0.9]], 0.0, 0.4, 0.05, 0.02),
        "box_beam": box_beam((0, 0, 0), (0, 0.3, -0.1), (0.16, 0.1), (0.1, 0.07), chamfer=0.15),
        "ladder_rail": ladder_rail((0, 0, 0), (0, 0, 1), 0.16),
        "helix": helix_sweep((0, 0, 0), (0, 0, 1), 0.01, 0.02, 0.004, 0.035, 0.9, 16),
        "mli": mli_blanket(1.2, 0.5),
    }
    for name, (V, F) in meshes.items():
        assert _closed_and_consistent(F), name
        assert signed_volume(V, F) > 0, name
    V, F = chamfered_box((1.0, 0.8, 0.5), edge=0.0)
    assert signed_volume(V, F) == pytest.approx(0.4)
    assert offset_polygon([[0, 0], [2, 0], [2, 1], [0, 1]], 0.1) == pytest.approx(
        np.array([[0.1, 0.1], [1.9, 0.1], [1.9, 0.9], [0.1, 0.9]]))
    for asm in (drill_assembly(), pan_tilt_head(), actuator((0, 0, 0), (0, 0, 0.1), 0.07)):
        assert sum(p[3] for p in asm) == pytest.approx(1.0)
        for suffix, (V, F), _mat, _share in asm:
            assert _closed_and_consistent(F) and signed_volume(V, F) > 0, suffix


def test_drill_assembly_feeds_down():
    stowed = np.concatenate([m[0] for s, m, *_ in drill_assembly(carriage=1.0) if s == "bit"])
    fed = np.concatenate([m[0] for s, m, *_ in drill_assembly(carriage=0.0) if s == "bit"])
    assert stowed[:, 2].min() > 0.0                          # stowed: the bit stays above the footpad bottom
    assert stowed[:, 2].min() - fed[:, 2].min() > 0.95       # a 1 m class feed stroke


def test_every_viper_part_is_closed_and_outward(spec):
    for p in spec.cad.parts:
        assert _closed_and_consistent(p.faces), p.name
        assert signed_volume(p.vertices, p.faces) > 0, p.name


def test_viper_envelope_mass_and_bodies(spec):
    cad = spec.cad
    lo, hi = cad.bounds()
    size = hi - lo
    assert size[0] <= 1.71 and size[1] <= 1.71 and size[2] <= 2.6
    assert hi[2] == pytest.approx(2.5, abs=0.01) and lo[2] == pytest.approx(0.0, abs=0.01)
    assert cad.total()["mass"] == pytest.approx(450.0)
    names = {f"{k}_{c.name}" for c in spec.corners for k in ("arm", "steer", "wheel")} | {"chassis"}
    assert set(cad.bodies()) == names
    mp = cad.mass_properties()
    for c in spec.corners:
        assert mp[f"arm_{c.name}"]["mass"] == pytest.approx(10.0)
        assert mp[f"steer_{c.name}"]["mass"] == pytest.approx(8.0)
        assert mp[f"wheel_{c.name}"]["mass"] == pytest.approx(spec.wheel.mass)
    assert spec.track() == pytest.approx(1.3) and spec.wheelbase() == pytest.approx(1.0)


def test_mast_head_and_drill_layout(spec):
    cad = spec.cad
    parts = {p.name: p for p in cad.parts}
    head = np.concatenate([p.vertices for p in cad.parts if p.name.startswith("mast_head_")])
    assert head[:, 2].max() == pytest.approx(2.5, abs=0.01) and head[:, 0].mean() > 0.3   # front of the deck
    belly = parts["chassis_body"].vertices[:, 2].min()
    drill = [p for p in cad.parts if p.name.startswith("trident_")]
    V = np.concatenate([p.vertices for p in drill])
    assert all(p.body == "chassis" for p in drill)
    assert V[:, 0].min() > 0.4                                 # at the front of the rover
    assert V[:, 2].min() < belly - 0.2                          # the auger and footpad reach below the belly
    assert V[:, 2].max() > parts["chassis_body"].vertices[:, 2].max()   # the rail stands above the deck
    assert sum(p.mass for p in drill) < 20.0                    # TRIDENT < 20 kg
    for key in ("NSS", "NIRVSS", "MSolo", "hga_dish", "lga"):
        assert key in parts
    assert parts["NSS"].vertices[:, 0].min() > 0.6               # NSS on the front face
    assert not any("middle of the rover" in k for k in spec.sources)


def test_suspension_and_steering_meshes_belong_to_their_bodies(spec):
    for c in spec.corners:
        arm = [p.name for p in spec.cad.body_parts(f"arm_{c.name}")]
        steer = [p.name for p in spec.cad.body_parts(f"steer_{c.name}")]
        assert any("beam" in n for n in arm) and any("_act_" in n for n in steer)


@pytest.mark.parametrize("angle_deg", [-50.0, 0.0, 50.0])
def test_wheels_clear_the_chassis_when_steered(spec, angle_deg):
    cad = spec.cad
    chassis = [(p.vertices.min(0) + 0.003, p.vertices.max(0) - 0.003) for p in cad.body_parts("chassis")]
    for c in spec.corners:
        W = np.concatenate([p.vertices for p in cad.body_parts(f"wheel_{c.name}")])
        R = rot(c.steer_axis, math.radians(angle_deg))
        Wr = (W - c.steer_origin) @ R.T + c.steer_origin
        for lo, hi in chassis:
            assert not np.any(np.all((Wr > lo) & (Wr < hi), axis=1)), c.name
