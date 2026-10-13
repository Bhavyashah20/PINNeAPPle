"""Dataset recipes of the operator-learning benchmarks (operator_datasets.py): band-limited inputs, exact Fourier
padding, worker-independent generation, split/shard layout."""
import json

import numpy as np
import pytest

from pinneapple_simulation.numerical_solvers import operator_datasets as od


def test_fourier_pad_is_exact_for_band_limited_inputs():
    w = od.ns_inputs(2, 32, seed=0)                      # Nyquist modes dropped: exactly band-limited
    assert np.abs(od.fourier_pad(w, 128, axes=(1, 2))[:, ::4, ::4] - w).max() < 1e-13
    u = od.burgers_inputs(2, 64, seed=0)
    assert np.abs(od.fourier_pad(u, 256, axes=(1,))[:, ::4] - u).max() < 1e-13


def test_darcy_refined_solution_is_close_to_the_coarse_one():
    a = od.darcy_inputs(1, 16, seed=3)
    u1, u2 = od.solve_darcy(a, refine=1), od.solve_darcy(a, refine=2)
    assert u1.shape == u2.shape == (1, 16, 16) and u2.min() > 0
    assert od.rel_l2(u1, u2).max() < 0.05


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr(od, "BURGERS", {**od.BURGERS, "n": 64, "dt": 1e-3, "n_save": 2, "n_train": 5, "n_test": 3})
    monkeypatch.setattr(od, "DARCY", {**od.DARCY, "n": 8, "refine": 2, "n_train": 3, "n_test": 2})
    monkeypatch.setattr(od, "NS", {**od.NS, "n_solve": 16, "n_store": 8, "t_end": 1.0, "n_save": 2, "dt": 0.05,
                                   "n_train": 6, "n_test": 3})
    monkeypatch.setattr(od, "SHARD_ROWS", {**od.SHARD_ROWS, "navier_stokes_2d_li2021_nu1e-3": 4})


def test_generate_layout_and_worker_independence(tmp_path, tiny):
    od.generate(tmp_path / "a", workers=1)
    od.generate(tmp_path / "b", workers=2)
    for name, files in (("burgers_1d_li2021", ["train.npz", "test.npz"]),
                        ("darcy_2d_li2021", ["train.npz", "test.npz"]),
                        ("navier_stokes_2d_li2021_nu1e-3", ["train-00.npz", "train-01.npz", "test.npz"])):
        meta = json.loads((tmp_path / "a" / name / "meta.json").read_text())
        assert meta["files"] == files
        for fn in files:                                 # byte-identical whatever the number of workers
            assert (tmp_path / "a" / name / fn).read_bytes() == (tmp_path / "b" / name / fn).read_bytes()
    with np.load(tmp_path / "a" / "burgers_1d_li2021" / "train.npz") as z:
        assert z["train__u"].shape == (5, 3, 64) and np.array_equal(z["train__u"][:, 0], z["train__u0"])
    with np.load(tmp_path / "a" / "navier_stokes_2d_li2021_nu1e-3" / "train-01.npz") as z:
        assert list(z["train__index"]) == [4, 5] and z["train__w"].shape == (2, 3, 8, 8)
    with np.load(tmp_path / "a" / "darcy_2d_li2021" / "test.npz") as z:
        assert set(np.unique(z["test__a"])) <= {3.0, 12.0} and z["test__u"].min() > 0
