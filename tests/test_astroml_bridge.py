"""Tests for `pinneapple_physics.pde_environment.presets.astroml_bridge`.

The first group runs without astroML: the module must import, and using it must raise an ImportError that says
how to install astroML. The rest needs astroML (skipped otherwise) and runs the synthetic, offline checks:
period recovery from an irregularly sampled noisy sinusoid, XD means of a two-component mixture with
heteroscedastic errors, a Bayesian-Blocks change point, and line fits with errors. No dataset is downloaded.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from pinneapple_physics.pde_environment.presets import astroml_bridge as ab

# ---------------------------------------------------------------------------
# Without astroML
# ---------------------------------------------------------------------------


def test_module_imports_without_astroml():
    code = (
        "import sys; sys.modules['astroML'] = None\n"
        "from pinneapple_physics.pde_environment.presets import astroml_bridge as ab\n"
        "assert not ab.astroml_available()\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_clear_import_error_without_astroml(monkeypatch):
    for name in ("astroML", "astroML.time_series", "astroML.density_estimation", "astroML.linear_model",
                 "astroML.datasets"):
        monkeypatch.setitem(sys.modules, name, None)
    t = np.linspace(0, 10, 20)
    with pytest.raises(ImportError, match="pip install astroML"):
        ab.lomb_scargle_periodogram(t, np.sin(t), 0.1)
    with pytest.raises(ImportError, match="pip install astroML"):
        ab.xd_fit(np.zeros((5, 2)), 0.1, 1)
    with pytest.raises(ImportError, match="pip install astroML"):
        ab.load_dataset("linear_sample")


def test_frequency_grid_and_light_curve_sample_need_no_astroml():
    t = np.array([0.0, 1.3, 2.0, 7.5, 10.0])
    f = ab.frequency_grid(t, period_range=(0.5, 5.0), oversample=4)
    assert f[0] == pytest.approx(0.2) and f[-1] == pytest.approx(2.0)
    assert np.allclose(np.diff(f), np.diff(f)[0])
    assert np.diff(f)[0] <= 1.0 / (4 * 10.0) + 1e-12
    with pytest.raises(ValueError):
        ab.frequency_grid(np.zeros(4))
    s = ab.light_curve_sample(t, np.ones(5), 0.1, band="V")
    assert s.domain_type() == "points"
    assert s.schema["units"]["time"] == "d"


def test_unknown_dataset_name():
    with pytest.raises(KeyError, match="unknown dataset"):
        ab.load_dataset("not_a_dataset")


# ---------------------------------------------------------------------------
# With astroML (synthetic data only)
# ---------------------------------------------------------------------------

astroml = pytest.mark.skipif(not ab.astroml_available(), reason="astroML not installed")


def _sinusoid(seed=0, period=0.613, n=150, noise=0.15):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 200.0, n))
    return t, 15.0 + 0.3 * np.sin(2 * np.pi * t / period + 0.7) + rng.normal(0, noise, n)


@astroml
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_period_recovery(seed):
    t, y = _sinusoid(seed)
    P = ab.best_period(t, y, 0.15, period_range=(0.2, 10.0))
    assert abs(P - 0.613) / 0.613 < 1e-3


@astroml
def test_periodogram_significance_and_upd():
    t, y = _sinusoid(0)
    pg = ab.lomb_scargle_periodogram(t, y, 0.15, period_range=(0.2, 10.0))
    assert set(pg.false_alarm_levels) == {0.01, 0.001}
    assert pg.false_alarm_levels[0.001] > pg.false_alarm_levels[0.01]
    assert pg.peak_power > pg.false_alarm_levels[0.001]
    assert pg.peak_false_alarm < 1e-6
    # pure noise: the highest peak is far below the signal's and not significant at 0.1 %.
    # (The M = 2 N approximation is optimistic on wide grids: see the docs, ~6 % of noise draws exceed the
    # nominal 1 % level on this grid, so the test does not use the 1 % level.)
    noise = 15 + np.random.default_rng(5).normal(0, 0.15, t.size)
    pn = ab.lomb_scargle_periodogram(t, noise, 0.15, period_range=(0.2, 10.0))
    assert pn.peak_power < 0.5 * pg.peak_power
    assert pn.peak_false_alarm > 1e-3
    s = pg.to_upd()
    assert s.provenance["source"] == "astroML"
    assert s.provenance["best_period"] == pytest.approx(pg.best_period)
    assert s.domain["coords"]["freq"].shape == s.state["power"].shape


@astroml
def test_multiterm_and_bootstrap():
    t, y = _sinusoid(0)
    assert abs(ab.best_period(t, y, 0.15, period_range=(0.2, 10.0), n_terms=2) - 0.613) / 0.613 < 1e-3
    f = np.linspace(0.1, 5.0, 300)
    D = ab.bootstrap_peak_distribution(t, y, 0.15, f, n_bootstraps=10, random_state=0)
    assert D.shape == (10,) and np.all((D >= 0) & (D <= 1))


@astroml
def test_xd_recovers_means_and_deconvolves_width():
    row = ab.check_xd_means(seed=0)
    assert row["passed"], row
    sd = np.asarray(row["intrinsic_sd"])
    # true intrinsic sd is 0.5; a plain GMM on the noisy points gives ~0.75-0.85
    assert np.all(np.abs(sd - 0.5) < 0.15)


@astroml
def test_bayesian_blocks_finds_step():
    row = ab.check_block_step(seed=0)
    assert row["passed"], row
    assert row["n_inner"] == 1


@astroml
def test_knuth_histogram():
    c, e, w = ab.knuth_histogram(np.random.default_rng(0).normal(0, 1, 1000))
    assert c.sum() == 1000 and e.size == c.size + 1 and 0.05 < w < 1.0


@astroml
def test_line_fits():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 80)
    y = 2.5 * x - 1.0 + rng.normal(0, 0.3, x.size)
    w = ab.fit_line_with_errors(x, y, 0.3)
    assert abs(w.slope - 2.5) < 4 * w.slope_err
    assert abs(w.intercept + 1.0) < 4 * w.intercept_err
    assert w.dof == 78 and 0.5 < w.chi2 / w.dof < 1.5
    np.testing.assert_allclose([w.intercept, w.slope], np.polyfit(x, y, 1)[::-1], rtol=1e-8)
    row = ab.check_line_fits(seed=0)
    assert row["passed"], row


@astroml
def test_validate_all_pass():
    rows = ab.validate(log=None)
    assert [r["case"] for r in rows] == ["period recovery", "XD means", "Bayesian-block step", "line fit (WLS, TLS)"]
    assert all(r["passed"] for r in rows), rows
