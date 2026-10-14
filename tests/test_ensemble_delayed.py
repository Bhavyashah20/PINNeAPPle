"""PhysicsEnsemble.run_delayed: prequential evaluation when each label arrives some time after its prediction."""
import numpy as np
import pytest

from pinneapple_physics.ensemble import PhysicsEnsemble, from_callable

X = np.linspace(0.0, 1.0, 32)


def truth(k):
    return np.sin(2 * np.pi * X + 0.3 * k) + 1.5


def _experts():
    """A perfect model, a biased one and a noisy one (fixed costs, so nothing depends on wall time)."""
    noise = np.random.default_rng(0).normal(0.0, 0.2, (64, X.size))
    return [from_callable("perfect", truth, cost=1.0),
            from_callable("biased", lambda k: truth(k) + 0.8, cost=1.0),
            from_callable("noisy", lambda k: truth(k) + noise[k], cost=2.0)]


def _assert_same_run(a, b):
    for f in ("errors", "ensemble_errors", "weights", "coverage", "costs", "suspect"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f), err_msg=f)
    assert a.active == b.active


@pytest.mark.parametrize("mode", ["combine", "select"])
def test_labels_available_immediately_equal_run(mode):
    n = 40
    queries = list(range(n))
    refs = [truth(k) if k % 7 != 3 else None for k in queries]      # some cases never get a label
    kw = dict(mode=mode, residual_fn=lambda p, k: float(np.mean(np.abs(np.diff(p, 2)))), residual_weight=0.1)
    ref_run = PhysicsEnsemble(_experts(), **kw).run(queries, refs, keep_predictions=True)
    delayed = PhysicsEnsemble(_experts(), **kw).run_delayed(queries, refs, available_at=queries, keep_predictions=True)
    _assert_same_run(ref_run, delayed)
    for p, q in zip(ref_run.predictions, delayed.predictions, strict=True):
        np.testing.assert_array_equal(p["prediction"], q["prediction"])
        np.testing.assert_array_equal(p["upper"], q["upper"])


@pytest.mark.parametrize("d", [1, 3, 6])
def test_weights_only_reflect_labels_already_available(d):
    n = 30
    queries = list(range(n))
    refs = [truth(k) for k in queries]
    immediate = PhysicsEnsemble(_experts()).run(queries, refs)
    delayed = PhysicsEnsemble(_experts()).run_delayed(queries, refs, available_at=[k + d for k in queries])
    # no label exists before step d: uniform weights; the first shift towards the perfect model happens at step d
    np.testing.assert_allclose(delayed.weights[:d], 1 / 3)
    assert delayed.weights[d, 0] > 1 / 3 + 1e-3
    # at step i the labels 0..i-d have been learned, exactly what run() has learned before its step i-d+1
    for i in range(d, n):
        np.testing.assert_array_equal(delayed.weights[i], immediate.weights[i - d + 1])
    # every label is scored in the end (flush), against the prediction made at the time
    np.testing.assert_array_equal(delayed.errors, immediate.errors)


def test_issue_times_and_no_flush():
    n = 12
    queries = list(range(n))
    refs = [truth(k) for k in queries]
    t = [10.0 * k for k in queries]
    ens = PhysicsEnsemble(_experts())
    run = ens.run_delayed(queries, refs, available_at=[tk + 25.0 for tk in t], issue_times=t, flush=False)
    np.testing.assert_allclose(run.weights[:3], 1 / 3)               # label 0 arrives at t=25: used from case 3 on
    assert run.weights[3, 0] > 1 / 3
    assert np.isnan(run.errors[-3:]).all() and np.isfinite(run.errors[:-3]).all()
    with pytest.raises(ValueError):
        ens.run_delayed(queries, refs[:-1], available_at=t)


def test_update_against_a_stored_prediction():
    a, b = PhysicsEnsemble(_experts()), PhysicsEnsemble(_experts())
    a.predict(0)
    a.update(0, truth(0))
    b.predict(0)
    state = b.last_state()
    b.predict(1)                                                      # a later prediction before the label arrives
    b.update(0, truth(0), state=state)
    np.testing.assert_array_equal(a.grid.expert_weights(), b.grid.expert_weights())
    with pytest.raises(RuntimeError):
        PhysicsEnsemble(_experts()).last_state()
