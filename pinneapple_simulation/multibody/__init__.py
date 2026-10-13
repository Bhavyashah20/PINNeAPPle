"""Rigid multibody dynamics in reduced coordinates: a floating base with trees of revolute joints.

    from pinneapple_simulation.multibody import MultibodyTree, Body
    tree = MultibodyTree(gravity=(0, 0, -1.62))
    tree.add_body(Body("chassis", 300.0, inertia=np.diag([40, 50, 60])))
    tree.add_revolute("arm_FL", "chassis", Body("arm_FL", 12.0), origin=(0.5, 0.4, 0.0), axis=(1, 0, 0))
    q, v = tree.initial_state(p=(0, 0, 1))
    q, v, kin = tree.step(q, v, 1e-3, lambda kin, q, v: (np.zeros(tree.nv), np.zeros((tree.nv, tree.nv))))

Used by :mod:`pinneapple_simulation.vehicle` for wheeled rovers; validated in ``tests/test_vehicle_multibody.py`` against
free fall, a pendulum's period, energy conservation and momentum conservation of a free tree.
"""
from .tree import (
           Body,
           Kinematics,
           MultibodyTree,
           RevoluteJoint,
           axis_angle,
           quat_from_matrix,
           quat_to_matrix,
           skew,
)

__all__ = ["MultibodyTree", "Body", "RevoluteJoint", "Kinematics", "skew", "quat_to_matrix", "quat_from_matrix",
           "axis_angle"]
