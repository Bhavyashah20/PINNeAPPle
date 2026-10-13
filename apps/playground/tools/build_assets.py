"""Generate the Playground's committed assets from the zoo and the geometry library.

    python apps/playground/tools/build_assets.py            # everything (needs torch, onnx, onnxruntime)
    python apps/playground/tools/build_assets.py --only geometries
    python apps/playground/tools/build_assets.py --check    # fail if a committed asset is out of date

Outputs (all under apps/playground/):

* ``site/models/<name>.onnx``: the zoo model as one ONNX graph, from the model's physical inputs (in the card's input
  order) to its physical outputs. Pre-processing (CST/log features, normalisation) and post-processing (denormalising,
  exp, ensemble mean and spread, conformal bound) are inside the graph, so the browser runs exactly what is tested.
* ``site/models/<name>.json``: the model card, its stored trust report, and what the page needs to run live checks
  (bounds, thresholds, reference points to overlay).
* ``site/data/geometries.json`` and ``site/data/thumbs/*.jpg``: the geometry gallery index.
* ``tests/fixtures/*.json``: Python reference values for the JS unit tests and the browser smoke test.

The build is deterministic: re-running it on the same weights gives byte-identical files.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "apps" / "playground"
SITE = APP / "site"
FIX = APP / "tests" / "fixtures"
sys.path.insert(0, str(ROOT))

MODELS = ("airfoil_cst_rans_mlp", "heatsink_forced_mlp", "rover_contact_grc1_viper")
OPSET = 17
GITHUB = "https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main"


# ---------------------------------------------------------------------------------------------------- ONNX wrappers
def _wrappers():
    import torch
    from torch import nn

    from pinneapple_design.aero.geometry import BOUNDS

    class Airfoil(nn.Module):
        """[T0..M2, alpha] -> mean (n, 3) and ensemble std (n, 3) of (Cl, Cd, Cm), as AirfoilMLP._forward."""

        def __init__(self, zm):
            super().__init__()
            self.net = zm.module
            self.register_buffer("lo", torch.tensor(BOUNDS[:, 0], dtype=torch.float32))
            self.register_buffer("span", torch.tensor(BOUNDS[:, 1] - BOUNDS[:, 0], dtype=torch.float32))
            self.register_buffer("ym", torch.tensor(zm.config["y_mean"], dtype=torch.float32))
            self.register_buffer("ys", torch.tensor(zm.config["y_std"], dtype=torch.float32))

        def forward(self, x):
            s = (x[:, 0:6] - self.lo) / self.span * 2 - 1
            a = x[:, 6:7]
            z = torch.cat([s, (a - 6) / 8, torch.sin(a * (math.pi / 180)) * 4], 1)
            y = self.net(z) * self.ys + self.ym                                  # (K, n, 3): Cl, log Cd, Cm
            y = torch.cat([y[..., 0:1], torch.exp(y[..., 1:2]), y[..., 2:3]], 2)
            m = y.mean(0)
            sd = torch.sqrt(((y - m) ** 2).mean(0))                              # population std, as numpy
            members = y.permute(1, 0, 2).reshape(-1, y.shape[0] * 3)             # (n, K*3): L/D per member
            return m, sd, members

    class Heatsink(nn.Module):
        """12 physical inputs (card order) -> [R, dP, R upper 95 %]."""

        def __init__(self, zm):
            super().__init__()
            self.net = zm.module
            c = zm.config
            for k in ("x_mean", "x_std", "y_mean", "y_std"):
                self.register_buffer(k, torch.tensor(c[k], dtype=torch.float32))
            self.q95 = float(math.exp(c["conformal_log_q"]["95"]))

        def forward(self, x):
            W, D, tb, N, t, H, k, sw, sd, Ta, V, Q = (x[:, i:i + 1] for i in range(12))
            gap = (W - N * t) / torch.clamp(N - 1, min=1.0)
            f = torch.cat([torch.log(W), torch.log(D), torch.log(tb), torch.log(torch.clamp(gap, min=1e-3)),
                           torch.log(t), torch.log(H), torch.log(N), torch.log(k), torch.log(sw / W),
                           torch.log(sd / D), Ta, torch.log(V), torch.log(Q)], 1)
            y = self.net((f - self.x_mean) / self.x_std)
            y = (y.y if hasattr(y, "y") else y) * self.y_std + self.y_mean
            r = torch.exp(y[:, 0:1])
            return torch.cat([r, torch.exp(y[:, 1:2]), r * self.q95], 1)

    class Rover(nn.Module):
        """[slip, sinkage_m] -> [Fx_N, Fz_N, My_Nm]."""

        def __init__(self, zm):
            super().__init__()
            self.net = zm.module
            c = zm.config
            for k in ("x_lo", "x_hi", "y_lo", "y_hi"):
                self.register_buffer(k, torch.tensor(c[k], dtype=torch.float32))

        def forward(self, x):
            xn = 2.0 * (x - self.x_lo) / (self.x_hi - self.x_lo) - 1.0
            return (self.net(xn) + 1.0) * (self.y_hi - self.y_lo) / 2.0 + self.y_lo

    return {"airfoil_cst_rans_mlp": (Airfoil, ["mean", "std", "members"]),
            "heatsink_forced_mlp": (Heatsink, ["y"]),
            "rover_contact_grc1_viper": (Rover, ["y"])}


def onnx_outputs(name: str, zm, X: np.ndarray) -> dict[str, np.ndarray]:
    """Zoo predictor outputs (``ZooModel._forward``) laid out like the ONNX graph's outputs."""
    out = zm._forward(X)
    if name == "airfoil_cst_rans_mlp":
        import torch
        with torch.no_grad():
            y = zm.module(torch.from_numpy(zm._enc(X))).numpy() * np.asarray(zm.config["y_std"]) + zm.config["y_mean"]
        y[..., 1] = np.exp(y[..., 1])
        return {"mean": np.stack([out["Cl"], out["Cd"], out["Cm"]], 1),
                "std": np.stack([out["Cl_std"], out["Cd_std"], out["Cm_std"]], 1),
                "members": y.transpose(1, 0, 2).reshape(len(X), -1)}
    if name == "heatsink_forced_mlp":
        return {"y": np.stack([out["r_base_k_w"], out["pressure_drop_pa"], out["r_base_upper95_k_w"]], 1)}
    return {"y": np.stack([out["Fx_N"], out["Fz_N"], out["My_Nm"]], 1)}


def output_scale(name: str, zm) -> dict[str, np.ndarray]:
    """Per-output scale for the parity tolerance: relative for positive log-modelled outputs (Cd, R, dP), the
    standardisation scale of the network for the others. |ONNX - PyTorch| / scale must be <= 1e-5."""
    if name == "airfoil_cst_rans_mlp":
        sc = np.array([zm.config["y_std"][0], np.nan, zm.config["y_std"][2]])
        return {"mean": sc, "std": sc, "members": np.tile(sc, zm.config["k"])}
    if name == "heatsink_forced_mlp":
        return {"y": np.array([np.nan, np.nan, np.nan])}
    lo, hi = np.asarray(zm.config["y_lo"]), np.asarray(zm.config["y_hi"])
    return {"y": (hi - lo) / 2}


def parity_error(ref: np.ndarray, got: np.ndarray, scale: np.ndarray, rel: np.ndarray | None = None) -> float:
    """Max of |got - ref| / scale, with NaN scales meaning 'relative to |rel|' (default ``ref``; positive outputs).
    The ensemble spread of Cd is measured relative to Cd itself."""
    s = np.where(np.isnan(scale), np.abs(ref if rel is None else rel), scale)
    return float(np.max(np.abs(got - ref) / np.maximum(s, 1e-12)))


def export_onnx(name: str, zm) -> bytes:
    import torch
    cls, outs = _wrappers()[name]
    w = cls(zm).eval()
    x = torch.zeros(4, len(zm.input_names))
    buf = io.BytesIO()
    kw = dict(input_names=["x"], output_names=outs, opset_version=OPSET,
              dynamic_axes={"x": {0: "n"}, **{o: {0: "n"} for o in outs}}, do_constant_folding=True)
    try:
        torch.onnx.export(w, (x,), buf, dynamo=False, **kw)
    except TypeError:                                                  # torch < 2.5 has no dynamo flag
        torch.onnx.export(w, (x,), buf, **kw)
    import onnx
    m = onnx.load_from_string(buf.getvalue())
    m.producer_name, m.producer_version, m.doc_string = "pinneapple-playground", "1", f"{name} v{zm.card.version}"
    onnx.checker.check_model(m)
    return m.SerializeToString()


def run_onnx(data: bytes, X: np.ndarray) -> dict[str, np.ndarray]:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    s = ort.InferenceSession(data, so, providers=["CPUExecutionProvider"])
    res = s.run(None, {"x": X.astype(np.float32)})
    return {o.name: r.astype(np.float64) for o, r in zip(s.get_outputs(), res, strict=True)}


def parity_inputs(name: str, zm) -> np.ndarray:
    """The model's held-out (and independent-reference) inputs plus random in-range points."""
    splits = zm.eval_splits
    X = np.concatenate([zm.reference(s, offline=True)[0] for s in splits])
    rng = np.random.default_rng(7)
    b = zm.bounds()
    if name == "heatsink_forced_mlp":
        return X
    lo = np.array([b[k][0] for k in zm.input_names])
    hi = np.array([b[k][1] for k in zm.input_names])
    return np.concatenate([X, lo + rng.random((500, len(lo))) * (hi - lo)])


# ---------------------------------------------------------------------------------------------------- model metadata
def _round(o, nd=6):
    if isinstance(o, float):
        return float(f"{o:.{nd}g}") if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    if isinstance(o, np.generic):
        return _round(o.item(), nd)
    return o


def model_meta(name: str, zm, onnx_bytes: bytes) -> dict:
    card = zm.card.to_dict()
    keep = ("name", "description", "task", "version", "inputs", "outputs", "validity_range", "known_limitations",
            "architecture", "equations", "training_config", "data_lineage", "validation_metrics", "reference_source",
            "citation", "license", "trust_report")
    meta = {"card": {k: card.get(k) for k in keep},
            "onnx": {"file": f"{name}.onnx", "bytes": len(onnx_bytes), "sha256": hashlib.sha256(onnx_bytes).hexdigest(),
                     "inputs": zm.input_names, "outputs": _wrappers()[name][1], "opset": OPSET},
            "hub": f"https://huggingface.co/PINNeAPPle-Labs/{name}",
            "code": {"adapter": f"{GITHUB}/pinneapple_hub/zoo/_models.py", "catalog": f"{GITHUB}/pinneapple_hub/zoo/_catalog.py",
                     "export": f"{GITHUB}/apps/playground/tools/build_assets.py"}}
    if name == "airfoil_cst_rans_mlp":
        meta["live"] = _airfoil_live(zm)
    elif name == "heatsink_forced_mlp":
        meta["live"] = {"conformal_log_q": zm.config["conformal_log_q"]}
    else:
        meta["live"] = _rover_live(zm)
    return _round(meta)


def _airfoil_live(zm) -> dict:
    """Thresholds for the live checks and the reference points the page overlays on the polar."""
    from pinneapple_design.aero.geometry import REFERENCE
    from pinneapple_hub.zoo import datasets
    X, _ = zm.reference("test", offline=True)
    sd = zm._forward(X)
    # ensemble spread seen on the held-out shapes: above its 99th percentile the page flags the query
    spread = {"Cl_std_p99": float(np.percentile(sd["Cl_std"], 99)), "Cd_std_rel_p99":
              float(np.percentile(sd["Cd_std"] / sd["Cd"], 99))}
    refs = []
    for split, label in (("naca_reference", "OpenFOAM RANS, not trained on"),
                         ("xfoil_naca2412", "XFOIL 6.99, independent solver")):
        d = datasets.load("airfoil_rans_cst_re4e6", split=split, offline=True)
        S = np.stack([np.asarray(d[k], float) for k in ("T0", "T1", "T2", "M0", "M1", "M2")], 1)
        for preset, shape in REFERENCE.items():
            hit = np.all(np.abs(S - shape) < 2e-4, 1)
            if hit.any():
                refs.append({"preset": preset, "source": label, "split": split,
                             "alpha": np.asarray(d["alpha_deg"], float)[hit].tolist(),
                             "Cl": np.asarray(d["Cl"], float)[hit].tolist(), "Cd": np.asarray(d["Cd"], float)[hit].tolist(),
                             "Cm": np.asarray(d["Cm"], float)[hit].tolist()})
    return {"spread": spread, "presets": {k: v.tolist() for k, v in REFERENCE.items()}, "references": refs,
            "design_constraints": {"min_thickness": 0.11, "max_abs_cm": 0.10,
                                   "source": f"{GITHUB}/pinneapple_design/aero/aircraft.py"},
            "thin_airfoil_slope_per_deg": 2 * math.pi * math.pi / 180, "slope_tolerance": 0.15}


def _rover_live(zm) -> dict:
    X, Y = zm.reference("test", offline=True)
    return {"physics": zm.config["physics"],
            "heldout": {"slip": X[:, 0].round(4).tolist(), "sinkage_m": X[:, 1].round(5).tolist(),
                        "Fx_N": Y[:, 0].round(2).tolist(), "Fz_N": Y[:, 1].round(2).tolist(),
                        "My_Nm": Y[:, 2].round(2).tolist()}}


# ---------------------------------------------------------------------------------------------------- fixtures
def smoke_cases() -> dict:
    """Fixed inputs the browser smoke test types into each mode, with the zoo predictor's answer."""
    from pinneapple_design.aero.geometry import naca4
    from pinneapple_hub import zoo
    out = {}
    a = zoo.load("airfoil_cst_rans_mlp", offline=True)
    cases = [("NACA 2412", naca4("2412").tolist(), 4.0), ("NACA 0012", naca4("0012").tolist(), 0.0),
             ("NACA 4415", naca4("4415").tolist(), 8.0)]
    out["airfoil"] = []
    for label, s, al in cases:
        p = a._forward(np.array([[*s, al]]))
        out["airfoil"].append({"preset": label, "alpha": al, "shape": s, "in_range": bool(a.in_range([[*s, al]])[0]),
                               **{k: float(v[0]) for k, v in p.items()}})
    h = zoo.load("heatsink_forced_mlp", offline=True)
    hs = {"base_width_mm": 80, "base_depth_mm": 100, "base_thickness_mm": 6, "n_fins": 14, "fin_thickness_mm": 1.5,
          "fin_height_mm": 35, "k_w_mk": 200, "source_width_mm": 40, "source_depth_mm": 40, "t_ambient_c": 25,
          "air_velocity_m_s": 2.5, "power_w": 60}
    p = h._forward(h.as_array({k: [v] for k, v in hs.items()}))
    out["heatsink"] = [{"inputs": hs, "in_range": bool(h.in_range(h.as_array({k: [v] for k, v in hs.items()}))[0]),
                        **{k: float(v[0]) for k, v in p.items()}}]
    r = zoo.load("rover_contact_grc1_viper", offline=True)
    out["rover"] = []
    for s, z in ((0.2, 0.02), (-0.1, 0.035), (0.6, 0.05)):
        p = r._forward(np.array([[s, z]]))
        out["rover"].append({"slip": s, "sinkage_m": z, "in_range": bool(r.in_range([[s, z]])[0]),
                             **{k: float(v[0]) for k, v in p.items()}})
    return _round(out, 10)


def geometry_fixtures() -> dict:
    from pinneapple_design.aero import geometry as g
    from pinneapple_design.aero.surrogate import MLPEnsemble
    rng = np.random.default_rng(11)
    b = g.BOUNDS
    shapes = [b[:, 0] + rng.random(6) * (b[:, 1] - b[:, 0]) for _ in range(6)]
    shapes += [np.array([0.3, 0.02, 0.3, 0.0, 0.0, 0.0]), np.array([0.12, 0.10, 0.11, 0.02, 0.25, -0.2]),
               np.array([0.12, 0.3, -0.3, 0.0, 0.0, 0.0])]                     # surfaces cross
    x = np.array([0.0, 1e-4, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    cases = []
    for s in shapes:
        yu, yl = g.surfaces(s, x)
        ok, why = g.valid(s)
        cases.append({"shape": s.tolist(), "x": x.tolist(), "upper": yu.tolist(), "lower": yl.tolist(),
                      "properties": g.properties(s), "valid": [bool(ok), why],
                      "encode_alpha": 5.5, "encode": MLPEnsemble.inputs(s, 5.5)[0].astype(float).tolist()})
    naca = {c: g.naca4(c).tolist() for c in ("0012", "2412", "4412", "4415", "6409", "0021")}
    return _round({"cases": cases, "naca4": naca}, 12)


def heatsink_range_fixtures() -> dict:
    from pinneapple_hub import zoo
    h = zoo.load("heatsink_forced_mlp", offline=True)
    X, _ = h.reference("test", offline=True)
    rng = np.random.default_rng(5)
    X = X[rng.choice(len(X), 40, replace=False)].copy()
    X[::3, 3] = np.maximum(2, X[::3, 3] * 3).round()                    # many fins: gap may fall below 0.6 mm
    X[1::5, 10] = 8.0                                                     # air speed above the range
    X[2::7, 7] = X[2::7, 0] * 0.1                                         # source too narrow
    return {"input_names": h.input_names, "X": X.tolist(), "in_range": h.in_range(X).tolist()}


def onnx_fixtures(name: str, data: bytes, zm) -> dict:
    X = parity_inputs(name, zm)
    rng = np.random.default_rng(3)
    X = X[rng.choice(len(X), min(24, len(X)), replace=False)]
    return _round({"inputs": X.tolist(), "outputs": {k: v.tolist() for k, v in run_onnx(data, X).items()},
                   "scale": {k: v.tolist() for k, v in output_scale(name, zm).items()}}, 12)


# ---------------------------------------------------------------------------------------------------- geometries
def geometry_gallery() -> tuple[dict, dict[str, bytes]]:
    import inspect

    from PIL import Image

    from pinneapple_design import geometry_library as gl
    from pinneapple_design.geometry_library import core
    from pinneapple_design.geometry_library.store import THUMBS, hub_url, read_manifest
    man = read_manifest()["geometries"]
    items, thumbs = [], {}
    for n in gl.list():
        info, spec = gl.info(n), core.get_spec(n)
        src = Path(inspect.getsourcefile(spec.builder)).resolve().relative_to(ROOT).as_posix()
        line = inspect.getsourcelines(spec.builder)[1]
        m = man.get(n, {})
        glb = f"{n}.light.glb"
        items.append({
            "name": n, "title": info["title"], "category": info["category"], "description": info["description"],
            "dimensions_m": m.get("dimensions_m", {}), "triangles": m.get("n_triangles", {}).get("light"),
            "glb": f"geometries/{glb}", "glb_bytes": m.get("files", {}).get(glb, {}).get("bytes"),
            "glb_hub": hub_url(n, glb), "frame": info["frame"], "physics": info["physics"], "tags": info["tags"],
            "sources": info["sources"], "parameters": [{k: p.get(k) for k in ("name", "default", "unit", "description")
                                                        if k in p} for p in info["parameters"]],
            "code": f"{GITHUB}/{src}#L{line}",
            "regenerate": f"import pinneapple as pp\ng = pp.geometries.build(\"{n}\", lod=\"light\")\n"
                          f"g.export(\"{n}.glb\")",
        })
        im = Image.open(THUMBS / f"{n}.png").convert("RGB")
        im.thumbnail((360, 240), Image.LANCZOS)
        b = io.BytesIO()
        im.save(b, "JPEG", quality=82, optimize=True, progressive=False)
        thumbs[f"{n}.jpg"] = b.getvalue()
    return _round({"version": 1, "geometries": items}), thumbs


# ---------------------------------------------------------------------------------------------------- driver
def _dump(o) -> bytes:
    return (json.dumps(o, indent=1, sort_keys=False, ensure_ascii=False) + "\n").encode()


def build(only: set[str]) -> dict[Path, bytes]:
    files: dict[Path, bytes] = {}
    if "models" in only:
        from pinneapple_hub import zoo
        for n in MODELS:
            zm = zoo.load(n, offline=True)
            data = export_onnx(n, zm)
            X = parity_inputs(n, zm)
            ref, got = onnx_outputs(n, zm, X), run_onnx(data, X)
            sc = output_scale(n, zm)
            err = max(parity_error(ref[k], got[k], sc[k], ref["mean"] if k == "std" else None) for k in ref)
            print(f"{n}: ONNX {len(data) / 1e3:.0f} kB, parity vs PyTorch on {len(X)} inputs: {err:.2e}")
            if err > 1e-5:
                raise SystemExit(f"{n}: ONNX parity {err:.2e} > 1e-5")
            files[SITE / "models" / f"{n}.onnx"] = data
            files[SITE / "models" / f"{n}.json"] = _dump(model_meta(n, zm, data))
            files[FIX / f"onnx_{n}.json"] = _dump(onnx_fixtures(n, data, zm))
        files[FIX / "smoke.json"] = _dump(smoke_cases())
        files[FIX / "heatsink_range.json"] = _dump(heatsink_range_fixtures())
    if "fixtures" in only or "models" in only:
        files[FIX / "geometry.json"] = _dump(geometry_fixtures())
    if "geometries" in only:
        idx, thumbs = geometry_gallery()
        files[SITE / "data" / "geometries.json"] = _dump(idx)
        for k, v in thumbs.items():
            files[SITE / "data" / "thumbs" / k] = v
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="models,fixtures,geometries")
    ap.add_argument("--check", action="store_true", help="compare with the committed files instead of writing")
    a = ap.parse_args(argv)
    files = build(set(a.only.split(",")))
    stale = []
    for p, b in files.items():
        if a.check:
            if not p.exists() or p.read_bytes() != b:
                stale.append(p.relative_to(ROOT).as_posix())
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b)
    if stale:
        print("out of date (run apps/playground/tools/build_assets.py):\n  " + "\n  ".join(stale))
        return 1
    print(f"{'checked' if a.check else 'wrote'} {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
