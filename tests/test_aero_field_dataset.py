"""The airfoil field dataset writer (pinneapple_design.aero.field_dataset) on the campaign plan and split."""
import json

import numpy as np
import pytest

from pinneapple_design.aero import dataset as c
from pinneapple_design.aero.field_dataset import write_field_dataset
from pinneapple_design.aero.train import split

HELD_OUT = ["s003", "s007", "s008", "s015", "s016", "s021", "s026", "s031", "s042", "s047", "s054", "s060", "s069",
            "s072", "s079"]                                   # the test shapes of the shipped airfoil models


def test_plan_is_the_436_run_campaign():
    jobs = c.plan(100, 4)
    assert len(jobs) == 436 and len({j["id"] for j in jobs}) == 436
    assert sum(j["set"] == "lhs" for j in jobs) == 400 and jobs[-1]["id"] == "NACA4415_a+14"
    for b, (lo, hi) in enumerate(c.ALPHA_BINS):
        a = [j["alpha"] for j in jobs if j["set"] == "lhs" and j["id"].endswith(f"_a{b}")]
        assert len(a) == 100 and lo <= min(a) and max(a) <= hi
    assert c.plan(100, 4) == jobs                                    # seeded


def _fake_runs(tmp_path, jobs):
    from pinneapple_design.aero.mesh import GridSpec
    n = GridSpec().ni * GridSpec().nj
    (tmp_path / "runs").mkdir()
    for j in jobs:
        rec = {**j, "reynolds": c.RE, "Cl": 0.1 * j["alpha"], "Cd": 0.01, "Cm": -0.05, "status": "converged"}
        (tmp_path / "runs" / f"{j['id']}.json").write_text(json.dumps(rec))
        z = np.full(n, j["alpha"], np.float32)
        np.savez(tmp_path / "runs" / f"{j['id']}.npz", p=z, Ux=z, Uy=z, nut=z)


def test_split_holds_out_the_shipped_test_shapes(tmp_path):
    jobs = c.plan(100, 4)
    recs = [{**j, "Cl": 0.0} for j in jobs]
    tr, te, ref = split(recs)
    assert sorted({r["shape_id"] for r in te}) == HELD_OUT
    assert len(te) == 60 and len(ref) == 36 and len(tr) == 340
    assert not {r["shape_id"] for r in tr} & set(HELD_OUT)


@pytest.mark.parametrize("n_lhs_shapes", [20])
def test_field_dataset_layout(tmp_path, n_lhs_shapes):
    jobs = [j for j in c.plan(100, 4) if j["set"] == "reference" or int(j["shape_id"][1:]) < n_lhs_shapes]
    _fake_runs(tmp_path, jobs)
    meta = write_field_dataset(str(tmp_path), str(tmp_path / "out"))
    assert meta["files"] == ["train.npz", "test.npz", "naca_reference.npz"]
    assert meta["rows"]["naca_reference"] == 36 and sum(meta["rows"].values()) == len(jobs)
    with np.load(tmp_path / "out" / meta["name"] / "naca_reference.npz") as z:
        assert z["naca_reference__p"].shape == (36, meta["n_cells"]) and z["naca_reference__x"].dtype == np.float32
        assert np.allclose(z["naca_reference__p"][:, 0], z["naca_reference__alpha_deg"])
        assert list(z["naca_reference__run_id"])[:2] == ["NACA0012_a+00", "NACA0012_a+02"]


def test_field_dataset_shards_large_splits(tmp_path):
    jobs = [j for j in c.plan(100, 4) if j["set"] == "reference" or int(j["shape_id"][1:]) < 20]
    _fake_runs(tmp_path, jobs)
    meta = write_field_dataset(str(tmp_path), str(tmp_path / "out"), shard_rows=30)
    assert meta["files"] == ["train-00.npz", "train-01.npz", "train-02.npz", "test.npz",
                             "naca_reference-00.npz", "naca_reference-01.npz"]
    d = tmp_path / "out" / meta["name"]
    ids = [r for f in ("naca_reference-00.npz", "naca_reference-01.npz")
           for r in np.load(d / f)["naca_reference__run_id"]]
    assert len(ids) == 36 and ids == sorted(ids)
