"""Snapshot matrices from arrays, tables, mesh series and OpenFOAM time directories (pinneapple_data.snapshots)."""
from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pytest

from pinneapple_data.snapshots import FormatError, load_snapshots, snapshots_from_arrays

CAVITY = os.path.join(os.path.dirname(__file__), "..", "apps", "simulation_comparator", "examples",
                      "cavity_Re100_20x20.zip")


def test_npz_round_trip_keeps_fields_times_and_locations():
    rng = np.random.default_rng(0)
    s = snapshots_from_arrays({"T": rng.random((6, 40)), "U": rng.random((6, 40, 2))}, times=np.arange(6) * 0.5,
                              locations={"point": rng.random((40, 3))}, meta={"hf_seconds": 2.5})
    back = load_snapshots([("s.npz", s.to_npz_bytes())])
    np.testing.assert_allclose(back.X, s.X)
    assert [f["name"] for f in back.fields] == ["T", "U"] and back.fields[1]["components"] == 2
    assert back.meta["hf_seconds"] == 2.5 and back.kind == "time" and back.locations["point"].shape == (40, 3)
    np.testing.assert_allclose(back.field_values("U"), s.field_values("U"))


def test_table_with_parameter_columns():
    csv = "param_L,param_k,v1,v2,v3\n1,2,0.1,0.2,0.3\n2,3,0.4,0.5,0.6\n3,1,0.7,0.8,0.9\n"
    s = load_snapshots([("runs.csv", csv.encode())])
    assert s.kind == "parametric" and s.param_names == ["L", "k"] and s.X.shape == (3, 3)
    t = load_snapshots([("ts.csv", b"t,a,b\n0.2,1,2\n0.1,3,4\n")])
    assert t.kind == "time" and t.times.tolist() == [0.1, 0.2] and t.X[0].tolist() == [3, 4]   # sorted by time


def test_series_of_vtu_files_ordered_by_the_number_in_the_name():
    meshio = pytest.importorskip("meshio")
    pts = np.c_[np.random.default_rng(1).random((20, 2)), np.zeros(20)]
    files = []
    for k in (10, 2, 5):
        m = meshio.Mesh(pts, [("vertex", np.arange(20)[:, None])], point_data={"T": np.full(20, float(k))})
        buf = io.BytesIO()
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.vtu")
            meshio.write(p, m)
            buf.write(open(p, "rb").read())
        files.append((f"run_{k}.vtu", buf.getvalue()))
    s = load_snapshots(files)
    assert s.times.tolist() == [2, 5, 10] and s.X[:, 0].tolist() == [2, 5, 10]


def test_openfoam_case_with_several_time_directories():
    z = zipfile.ZipFile(CAVITY)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as w:
        for n in z.namelist():
            b = z.read(n)
            w.writestr(n, b)
            parts = n.split("/")
            if len(parts) > 2 and parts[1] == "5":                    # copy the time 5 as 6 and 7, U scaled
                for t, _f in (("6", 2.0), ("7", 3.0)):
                    w.writestr("/".join([parts[0], t] + parts[2:]), b)
    s = load_snapshots([("case.zip", out.getvalue())], fields=["U"])
    assert s.times.tolist() == [5.0, 6.0, 7.0] and s.source["format"] == "openfoam"
    assert s.field_values("U").shape == (3, 400, 3) and s.locations["cell"].shape == (400, 3)


def test_one_time_directory_is_not_a_snapshot_set():
    with pytest.raises(FormatError, match="fewer than two"):
        load_snapshots([("c.zip", open(CAVITY, "rb").read())])
