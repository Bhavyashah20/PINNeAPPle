"""UsdPhysics in the twin3d USD export, checked with the official OpenUSD
parser (``usd-core``): the applied API schemas, attribute values, the physics
material binding, gravity, and the time-sampled rigid motion. The reference
is OpenUSD's own ``UsdPhysics.LoadUsdPhysicsFromRange``, which must recognise
the same bodies, colliders and scene that were declared."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_twin3d import Scene
from pinneapple_twin3d.scene import PhysicsMaterial
from pinneapple_twin3d.usd import export_usd

pxr = pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdPhysics, UsdShade  # noqa: E402

CUBE_V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                   [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
CUBE_F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                   [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])


def _scene():
    times = [0.0, 0.5, 1.0]
    sc = Scene("drop test", times=times)
    sc.set_gravity((0, 0, -9.81))
    sc.add_part("floor", CUBE_V * [10, 10, 0.1] - [5, 5, 0.1], CUBE_F)
    sc.add_part("box", CUBE_V, CUBE_F)
    sc.add_part("replay", CUBE_V + 3, CUBE_F)
    steel = PhysicsMaterial("steel", static_friction=0.6, dynamic_friction=0.4, restitution=0.2, density=7850)
    sc.add_physics("floor", rigid_body=False, collider=True, approximation="none", material=steel)
    sc.add_physics("box", mass=2.5, center_of_mass=(0.5, 0.5, 0.5), velocity=(1, 0, 0), material=steel)
    sc.add_physics("replay", kinematic=True, collider=False)
    pos = [[0, 0, 0], [0, 0, 1], [0, 0, 2]]
    ang = np.deg2rad([0, 45, 90])
    quats = [[np.cos(a / 2), 0, 0, np.sin(a / 2)] for a in ang]
    sc.add_motion("replay", pos, quats)
    return sc


def test_usd_physics_schemas_and_values(tmp_path):
    path = export_usd(_scene(), str(tmp_path / "twin.usda"))
    stage = Usd.Stage.Open(path)
    assert UsdPhysics.GetStageKilogramsPerUnit(stage) == 1.0

    floor, box, replay = (stage.GetPrimAtPath(f"/Twin/{n}") for n in ("floor", "box", "replay"))
    assert floor.HasAPI(UsdPhysics.CollisionAPI) and not floor.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MeshCollisionAPI(floor).GetApproximationAttr().Get() == "none"

    assert box.HasAPI(UsdPhysics.RigidBodyAPI) and box.HasAPI(UsdPhysics.MassAPI)
    mass = UsdPhysics.MassAPI(box)
    assert mass.GetMassAttr().Get() == pytest.approx(2.5)
    assert tuple(mass.GetCenterOfMassAttr().Get()) == pytest.approx((0.5, 0.5, 0.5))
    rb = UsdPhysics.RigidBodyAPI(box)
    assert tuple(rb.GetVelocityAttr().Get()) == pytest.approx((1, 0, 0))
    assert rb.GetKinematicEnabledAttr().Get() is False
    assert UsdPhysics.MeshCollisionAPI(box).GetApproximationAttr().Get() == "convexHull"

    mat, _ = UsdShade.MaterialBindingAPI(box).ComputeBoundMaterial(materialPurpose="physics")
    assert mat and mat.GetPrim().GetPath() == "/Twin/PhysicsMaterials/steel"
    pm = UsdPhysics.MaterialAPI(mat.GetPrim())
    assert (pm.GetStaticFrictionAttr().Get(), pm.GetDynamicFrictionAttr().Get(),
            pm.GetRestitutionAttr().Get(), pm.GetDensityAttr().Get()) == pytest.approx((0.6, 0.4, 0.2, 7850))

    assert UsdPhysics.RigidBodyAPI(replay).GetKinematicEnabledAttr().Get() is True
    assert not replay.HasAPI(UsdPhysics.CollisionAPI)

    scene = UsdPhysics.Scene(stage.GetPrimAtPath("/Twin/physicsScene"))
    assert scene.GetGravityMagnitudeAttr().Get() == pytest.approx(9.81)
    assert tuple(scene.GetGravityDirectionAttr().Get()) == pytest.approx((0, 0, -1))


def test_official_parser_recognises_bodies_colliders_and_scene(tmp_path):
    path = export_usd(_scene(), str(tmp_path / "twin.usda"))
    stage = Usd.Stage.Open(path)
    load = getattr(UsdPhysics, "UsdPhysicsLoadStageFromPrimRange", None) or UsdPhysics.LoadUsdPhysicsFromRange
    parsed = load(stage, ["/"])
    kinds = {}
    for obj_type, (paths, _descs) in parsed.items():
        kinds[str(obj_type).split(".")[-1]] = sorted(str(p) for p in paths)
    assert kinds.get("Scene") == ["/Twin/physicsScene"]
    assert kinds.get("RigidBody") == ["/Twin/box", "/Twin/replay"]
    collider_paths = sorted(p for k, v in kinds.items() if k.endswith("Shape") for p in v)
    assert collider_paths == ["/Twin/box", "/Twin/floor"]
    assert kinds.get("RigidBodyMaterial") == ["/Twin/PhysicsMaterials/steel"]


def test_motion_time_samples_reproduce_the_recorded_pose(tmp_path):
    """Closed form: at step k the vertex v is drawn at pos[k] + R(q[k]) v."""
    path = export_usd(_scene(), str(tmp_path / "twin.usda"))
    stage = Usd.Stage.Open(path)
    from pxr import UsdGeom
    replay = UsdGeom.Xformable(stage.GetPrimAtPath("/Twin/replay"))
    for k, (z, ang) in enumerate([(0, 0), (1, 45), (2, 90)]):
        m = replay.ComputeLocalToWorldTransform(Usd.TimeCode(k))
        p = m.Transform(Gf.Vec3d(1, 0, 0))      # local point (1,0,0)
        a = np.deg2rad(ang)
        assert tuple(p) == pytest.approx((np.cos(a), np.sin(a), z), abs=1e-6)


def test_validation_of_physics_inputs():
    sc = Scene("x")
    sc.add_part("p", CUBE_V, CUBE_F)
    with pytest.raises(ValueError):
        sc.add_physics("p", mass=-1)
    with pytest.raises(ValueError):
        sc.add_physics("p", approximation="sphere")
    with pytest.raises(ValueError):
        sc.add_physics("p", rigid_body=False, kinematic=True)
    with pytest.raises(ValueError):
        sc.add_physics("p", material=PhysicsMaterial("bad", restitution=1.5))
    with pytest.raises(ValueError):
        sc.set_gravity((0, 9.81))


def test_scene_without_physics_is_unchanged(tmp_path):
    sc = Scene("plain")
    sc.add_part("p", CUBE_V, CUBE_F)
    text = open(export_usd(sc, str(tmp_path / "p.usda"))).read()
    assert "physics:" not in text and "apiSchemas" not in text and "PhysicsScene" not in text
    assert Usd.Stage.Open(str(tmp_path / "p.usda")).GetPrimAtPath("/Twin/p").IsValid()
