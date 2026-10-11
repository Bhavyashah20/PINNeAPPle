from dataclasses import replace

import numpy as np
import pytest

import pinneapple as pp
from pinneapple_physics.solving import Solution

NU = 0.01 / np.pi


def test_compare_accepts_precomputed_solutions_and_mixed_method_inputs():
    exact = pp.solve("burgers_1d", "analytic", preset_kwargs={"nu": NU})

    def precomputed(problem):
        return Solution(problem, "precomputed", exact.predict, exact.wall_time_s, kind="analytic")

    comparison = pp.compare(
        "burgers_1d",
        methods=["analytic", precomputed, exact, exact],
        reference=exact,
        preset_kwargs={"nu": NU},
        points=np.array([[0.0, 0.25], [0.5, 0.75]]),
    )

    assert [row["method"] for row in comparison.rows] == [
        "analytic",
        "precomputed",
        "analytic (2)",
        "analytic (3)",
    ]
    assert all(row["status"] == "ok" for row in comparison.rows)
    assert all(row["kind"] == "analytic" for row in comparison.rows)
    assert all(row["wall_time_s"] == exact.wall_time_s for row in comparison.rows)
    assert all(row["metrics"]["relative_l2"]["u"] == pytest.approx(0.0) for row in comparison.rows)


def test_compare_rejects_precomputed_solutions_for_different_coordinates():
    exact = pp.solve("burgers_1d", "analytic", preset_kwargs={"nu": NU})
    incompatible = Solution(
        replace(exact.problem, coords=("y", "t")),
        exact.method,
        exact.predict,
        exact.wall_time_s,
        kind=exact.kind,
    )

    with pytest.raises(ValueError, match="has coordinates .* but problem"):
        pp.compare("burgers_1d", methods=[incompatible], reference=exact, preset_kwargs={"nu": NU})
