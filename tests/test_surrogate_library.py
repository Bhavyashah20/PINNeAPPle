"""pinneapple_analysis.surrogate: data, splits, candidates, study, limitations, packaging."""
import io
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from pinneapple_analysis import surrogate as S
from pinneapple_analysis.cost.equal_budget import ComputeBudget
from pinneapple_analysis.surrogate.methods import TrainContext, fit_candidate


def _scalar(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    Y = np.stack([np.sin(2 * X[:, 0]) + X[:, 1] ** 2, X[:, 0] * X[:, 1]], 1)
    return S.SurrogateData(X, Y, ["a", "b"], ["f", "g"], groups=np.repeat(np.arange(n // 4), 4))


def _fields(n=24):
    return S.examples.burgers_fields(n=n, nx=16, nt=6)


def test_read_table_formats_and_fields(tmp_path):
    df = pd.DataFrame({"p": [1.0, 2.0, 3.0], "q": [0.1, 0.2, 0.3], "u_0": [1, 2, 3], "u_1": [4, 5, 6],
                       "u_2": [7, 8, 9], "u_3": [0, 1, 2]})
    for name, raw in [("t.csv", df.to_csv(index=False).encode()), ("t.json", df.to_json(orient="records").encode())]:
        back = S.read_table(raw, name)
        assert list(back.columns) == list(df.columns)
    buf = io.BytesIO()
    df.to_parquet(buf)
    assert S.read_table(buf.getvalue(), "t.parquet").shape == df.shape
    np.savez(tmp_path / "a.npz", p=df.p.to_numpy(), u=df[["u_0", "u_1", "u_2", "u_3"]].to_numpy())
    assert S.field_columns(S.read_table(tmp_path / "a.npz").columns, "u") == ["u_0", "u_1", "u_2", "u_3"]
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "a.h5", "w") as f:
        f["p"] = df.p.to_numpy()
        f["out/u"] = df[["u_0", "u_1"]].to_numpy()
    assert set(S.read_table(tmp_path / "a.h5").columns) == {"p", "out.u_0", "out.u_1"}
    d = S.from_table(df, ["p", "q"], ["u"], grid_shape=(2, 2))
    assert d.kind == "field" and d.grid_shape == (2, 2) and d.coords.shape == (4, 2)
    s = S.from_table(df, ["p"], ["q", "u_0"])
    assert s.kind == "scalar" and s.Y.shape == (3, 2)
    with pytest.raises(ValueError, match="not found"):
        S.from_table(df, ["p"], ["missing"])


def test_inspect_table_and_grid_coords():
    df = pd.DataFrame({"case": ["a", "a", "b"], "p": [1.0, 2.0, 3.0], "y": [0.1, 0.2, 0.4],
                       **{f"u_{i}": [i, i, i] for i in range(6)}})
    info = S.inspect_table(df)
    assert info["fields"]["u"]["columns"] == 6 and info["suggested"]["outputs"] == ["u"]
    assert info["suggested"]["inputs"] == ["p", "y"] and info["suggested"]["group"] == "case"
    r = S.inspect_table(df, roles={"p": "input", "y": "output", "case": "id"})
    assert r["suggested"] == {"inputs": ["p"], "outputs": ["y"], "group": "case"}
    c = S.grid_coords((3, 2), [(-1, 1), (0, 0.5)])
    assert c.shape == (6, 2) and c[0].tolist() == [-1, 0] and c[1].tolist() == [-1, 0.5] and c[-1].tolist() == [1, 0.5]


def test_splits_are_disjoint_and_shift_is_unseen():
    d = _scalar()
    for shift in ("edge", "group", {"column": "a", "side": "low"}):
        sp = S.make_splits(d, shift=shift)
        sets = [set(sp.train), set(sp.val), set(sp.test), set(sp.shift)]
        assert sum(map(len, sets)) == d.n and len(set().union(*sets)) == d.n
        if shift == "group":
            assert not set(d.groups[sp.shift]) & set(d.groups[sp.train])
        if isinstance(shift, dict):
            assert d.X[sp.shift, 0].max() <= d.X[sp.train, 0].min()
    few = S.make_splits(d).few_data(0.25)
    assert len(few.train) < len(S.make_splits(d).train)


def test_candidate_applicability():
    a = S.applicable("scalar", 100)
    assert a["ridge"] is None and a["gbt"] is None and "scalars" in a["fno"] and "field" in a["pinn"]
    f = S.applicable("field", 100, (16, 6), has_pde=False)
    assert f["fno"] is None and f["deeponet"] is None and "no PDE" in f["pinn"] and f["gbt"]
    assert S.applicable("field", 100, (16, 6), has_pde=True)["pinn"] is None
    assert "O(n^3)" in S.applicable("scalar", 5000)["gp"]


def test_pde_declaration_and_fd_residual():
    d = _fields()
    with pytest.raises(ValueError, match="unknown PDE"):
        S.parse_pde({"type": "nope"}, d.coord_names, d.input_names)
    with pytest.raises(ValueError, match="neither an input"):
        S.parse_pde({"type": "burgers", "coefficients": {"nu": "viscosity"}}, d.coord_names, d.input_names)
    p = S.parse_pde(d.meta["pde"], d.coord_names, d.input_names)
    exact = p.fd_residual(d.Y, d.coords, d.X).mean()
    wrong = p.fd_residual(1.3 * d.Y, d.coords, d.X).mean()
    assert wrong > 1.2 * exact


def test_scalar_study_end_to_end(tmp_path):
    d = _scalar()
    st = S.run_study(d, methods=["ridge", "gp", "mlp"], budget={"steps": 150}, seeds=2)
    assert st.best in ("ridge", "gp", "mlp")
    rk = st.ranking("test")
    assert [r["error"] for r in rk] == sorted(r["error"] for r in rk)
    gp = next(m for m in st.methods if m["id"] == "gp")
    assert gp["summary"]["test_error_mean"] < 0.1               # smooth 2-D function: a GP interpolates it
    assert {"fno", "deeponet", "pinn"} <= {m["id"] for m in st.methods if not m["applicable"]}
    assert st.trust["decision"] in ("APPROVED", "REVIEW", "REJECT") and st.trust["checks"]
    lim = st.limitations
    assert set(lim["validity_bounds"]) == {"a", "b"} and "concentration" in lim and lim["worst_samples"]
    json.dumps(st.to_dict(), default=str)
    md = st.to_markdown()
    assert "## Comparison" in md and "not run:" in md and st.trust["decision"] in md
    m = st.model
    out = m.predict({"a": [0.0, 5.0], "b": 0.0})
    assert out["ood"][1]["ood"] and not out["ood"][0]["ood"] and out["outputs"]["f"][1] is not None
    ref = m.predict({"a": [0.0, 5.0], "b": 0.0}, ood_policy="refuse")
    assert ref["outputs"]["f"][1] is None and ref["refused"] == 1
    m.save(tmp_path / "m")
    back = S.SurrogateModel.load(tmp_path / "m")
    assert np.allclose(back.predict_array(d.X[:5]), m.predict_array(d.X[:5]))
    (tmp_path / "m" / "model.pkl").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256"):
        S.SurrogateModel.load(tmp_path / "m")


def test_field_study_with_declared_pde_offers_pinn_and_checks_residual():
    d = _fields()
    st = S.run_study(d, methods=["gp", "deeponet", "pinn"], budget={"steps": 40}, seeds=1,
                     options={"pinn": {"n_data": 64, "n_colloc": 64}})
    pinn = next(m for m in st.methods if m["id"] == "pinn")
    assert pinn["applicable"] and pinn["summary"]["n_ok"] == 1
    assert st.limitations["pde_residual"]["ratio"] > 0
    assert any("PDE residual" in c["name"] for c in st.trust["checks"])
    assert len(st.limitations["field_error"]["map"]) == 16 * 6
    assert st.model.predict({"nu": 0.02})["outputs"]["u"][0] is not None


def test_ood_gate_flags_range_and_holes():
    X = np.concatenate([np.linspace(0, 1, 50), np.linspace(2, 3, 50)])[:, None]
    g = S.OODGate(X, ["x"])
    a = g.assess(np.array([[0.5], [1.5], [4.0]]))
    assert not a[0]["ood"] and a[1]["ood"] and a[1]["knn_distance"] > 3 and a[2]["outside"] == ["x"]


def test_equal_budget_iterative_methods_stop_at_the_budget():
    d = _scalar()
    sp = S.make_splits(d)
    ctx = TrainContext(d.X[sp.train], d.Y[sp.train], d.X[sp.val], d.Y[sp.val], ComputeBudget(steps=37))
    f, _ = fit_candidate("mlp", ctx)
    assert f.info["steps"] == 37 and f.info["stopped_by"] == "steps"


def test_export_torchscript_matches(tmp_path):
    torch = pytest.importorskip("torch")
    d = _scalar()
    st = S.run_study(d, methods=["mlp"], budget={"steps": 30}, seeds=1)
    p = st.model.export(tmp_path / "m.pt", "torchscript")
    ts = torch.jit.load(str(p))
    x = d.X[:4]
    assert np.allclose(ts(torch.tensor(x, dtype=torch.float32)).detach().numpy(), st.model.predict_array(x),
                       atol=1e-5)


def test_airfoil_example_is_the_zoo_campaign():
    d = S.examples.airfoil_rans()
    assert d.input_names[-1] == "alpha_deg" and d.output_names == ["Cl", "Cd"]
    assert d.groups is not None and len(np.unique(d.groups)) == 100 and (d.Y[:, 1] > 0).all()
