"""Model zoo and dataset hub: round trip load -> predict reproduces the stored metrics, checksums, cache, offline and
not-published behaviour, the fine-tune guard, validation and the publish script."""
from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pinneapple as pp  # noqa: E402
from pinneapple_hub import zoo  # noqa: E402
from pinneapple_hub.zoo import _artifacts as art  # noqa: E402

MODELS = ["airfoil_cst_rans_mlp", "heatsink_forced_mlp", "rover_contact_grc1_viper"]
DATASETS = ["airfoil_rans_cst_re4e6", "heatsink_forced_convection", "heatsink_natural_convection",
            "rover_contact_grc1_viper", "single_wheel_tests"]


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNEAPPLE_ZOO_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("PINNEAPPLE_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return tmp_path / "cache"


# ------------------------------------------------------------------ catalogue
def test_lazy_shortcuts_and_listing():
    assert pp.zoo is zoo and pp.datasets is zoo.datasets
    assert zoo.list() == MODELS
    assert pp.datasets.list() == DATASETS
    assert zoo.list(task="terramechanics") == ["rover_contact_grc1_viper"]


def test_manifest_is_small_hashed_and_versioned():
    man = art.manifest()
    for kind in ("models", "datasets"):
        for name, e in man[kind].items():
            assert e["hf_repo"].startswith("PINNeAPPle-Labs/") and e["revision"] == f"v{e['version']}"
            for fn, spec in e["files"].items():
                p = art.ASSETS / spec["bundled"]
                assert p.stat().st_size == spec["size"] < 2_000_000, (name, fn)
                assert art.sha256_file(p) == spec["sha256"], (name, fn)


@pytest.mark.parametrize("name", MODELS)
def test_model_card_is_complete(name):
    card = zoo.info(name)
    assert card.validate() == []
    assert card.task and card.version and card.license and card.citation and card.reference_source
    assert all(d.get("unit") for d in card.inputs + card.outputs)
    assert card.validity_range and card.data_lineage.get("dataset") in DATASETS
    assert card.trust_decision == "APPROVED"
    md = card.to_markdown()
    assert "Validity range" in md and "Inputs" in md and "Training data" in md


@pytest.mark.parametrize("name", DATASETS)
def test_dataset_schema_and_splits(name):
    info = pp.datasets.info(name)
    splits = pp.datasets.load(name)
    assert set(splits) == set(info.splits)
    for s, d in splits.items():
        assert d.n_rows == info.splits[s]["n"] > 0
        assert set(d.columns) <= set(info.columns)
    assert info.license and info.source and info.verification
    assert "## Splits" in info.to_markdown()


# ------------------------------------------------------------------ round trip
@pytest.mark.parametrize("name", MODELS)
def test_load_predict_reproduces_card_metrics(name):
    res = zoo.evaluate(name)
    assert res["matches_card"], res                     # within the card tolerance (float32 differs across CPUs)
    assert res["max_rel_diff_vs_card"] <= res["tolerance"] <= 1e-3


def test_airfoil_predicts_known_case_and_compares_with_xfoil():
    m = zoo.load("airfoil_cst_rans_mlp")
    xf = pp.datasets.load("airfoil_rans_cst_re4e6", split="xfoil_naca2412")
    out = m.predict({k: xf[k] for k in m.input_names})
    assert np.all(np.abs(out["Cl"] - xf["Cl"]) < 0.03)                 # independent solver, ~2 % of Cl
    assert np.all(np.abs(out["Cd"] / xf["Cd"] - 1) < 0.10)
    assert set(out) >= {"Cl", "Cd", "Cm", "Cl_std"}


def test_named_inputs_and_out_of_range_warning():
    m = zoo.load("rover_contact_grc1_viper")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = m.predict(slip=0.2, sinkage_m=0.02)
    with pytest.warns(zoo.OutOfRangeWarning):
        b = m.predict(np.array([[0.2, 0.02], [1.5, 0.02]]))
    assert np.allclose(a["Fx_N"], b["Fx_N"][0])
    assert m.in_range([[0.2, 0.02], [1.5, 0.02]]).tolist() == [True, False]
    with pytest.raises(KeyError):
        m.predict(slip=0.2)


def test_evaluate_on_given_cases_reports_out_of_range_fraction():
    m = zoo.load("heatsink_forced_mlp")
    d = pp.datasets.load("heatsink_forced_convection", split="test")
    r = zoo.evaluate(m, d)
    assert r["fraction_out_of_range"] == 0.0
    assert r["metrics"]["R_mean_abs_pct"] == pytest.approx(zoo.info("heatsink_forced_mlp")
                                                            .validation_metrics["test/R_mean_abs_pct"], rel=1e-4)


# ------------------------------------------------------------------ validation
def test_validate_approves_and_flags_out_of_range_queries():
    rep = zoo.validate("rover_contact_grc1_viper")
    assert rep.decisao == "APPROVED"
    with pytest.warns(zoo.OutOfRangeWarning):
        rep2 = zoo.validate("rover_contact_grc1_viper", inputs=[[0.1, 0.01], [0.1, 0.5]])
    assert rep2.details["fraction_out_of_range"] == 0.5
    assert rep2.decisao == "REVIEW"


# ------------------------------------------------------------------ checksums, cache, offline, not published
def _copy_assets(tmp_path, monkeypatch):
    dst = tmp_path / "assets"
    shutil.copytree(art.ASSETS, dst)
    monkeypatch.setattr(art, "ASSETS", dst)
    return dst


def test_bundled_checksum_mismatch_is_detected(tmp_path, monkeypatch, isolated_cache):
    dst = _copy_assets(tmp_path, monkeypatch)
    p = dst / "models" / "rover_contact_grc1_viper" / "weights.pt"
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0xFF
    p.write_bytes(bytes(b))
    with pytest.raises(zoo.ChecksumMismatchError):
        zoo.load("rover_contact_grc1_viper")


def test_offline_without_bundle_or_cache_says_so(monkeypatch, isolated_cache):
    monkeypatch.setenv("PINNEAPPLE_ZOO_IGNORE_BUNDLED", "1")
    monkeypatch.setenv("PINNEAPPLE_OFFLINE", "1")
    with pytest.raises(zoo.ArtifactUnavailableError, match="offline"):
        pp.datasets.load("single_wheel_tests", split="all")


def _fake_hub(monkeypatch, behaviour):
    """Replace huggingface_hub.hf_hub_download: 'serve' copies the bundled file, 'corrupt' serves a damaged copy,
    'missing' raises RepositoryNotFoundError."""
    hub = pytest.importorskip("huggingface_hub")
    calls = []

    def fake(repo_id, filename, *, revision, repo_type, local_dir):
        calls.append((repo_id, filename, revision, repo_type))
        if behaviour == "missing":
            err = RuntimeError("404 Client Error: Repository Not Found")
            err.response = _Resp()
            raise err
        kind = "datasets" if repo_type == "dataset" else "models"
        name = next(n for n, e in art.manifest()[kind].items() if e["hf_repo"] == repo_id)
        src = Path(__file__).resolve().parents[1] / "pinneapple_hub/zoo/assets" / art.manifest()[kind][name][
            "files"][filename]["bundled"]
        out = Path(local_dir) / filename
        data = bytearray(src.read_bytes())
        if behaviour == "corrupt":
            data[-1] ^= 0xFF
        out.write_bytes(bytes(data))
        return str(out)

    monkeypatch.setattr(hub, "hf_hub_download", fake)
    return calls


class _Resp:
    status_code = 404


def test_not_published_via_huggingface_hub(monkeypatch, isolated_cache):
    monkeypatch.setenv("PINNEAPPLE_ZOO_IGNORE_BUNDLED", "1")
    _fake_hub(monkeypatch, "missing")
    with pytest.raises(zoo.ArtifactNotPublishedError, match="publish_to_hub"):
        zoo.load("airfoil_cst_rans_mlp")


def test_lazy_download_then_cache_then_offline(monkeypatch, isolated_cache):
    monkeypatch.setenv("PINNEAPPLE_ZOO_IGNORE_BUNDLED", "1")
    calls = _fake_hub(monkeypatch, "serve")
    assert pp.datasets.status("single_wheel_tests") == {"data.npz": "remote"}
    d = pp.datasets.load("single_wheel_tests", split="all")
    assert d.n_rows > 20 and len(calls) == 1
    assert calls[0] == ("PINNeAPPle-Labs/single-wheel-tests", "data.npz", "v1.0.0", "dataset")
    assert pp.datasets.status("single_wheel_tests") == {"data.npz": "cached"}
    monkeypatch.setenv("PINNEAPPLE_OFFLINE", "1")
    pp.datasets.load("single_wheel_tests", split="all")                     # served from the cache
    assert len(calls) == 1


def test_downloaded_checksum_mismatch_is_rejected_and_not_cached(monkeypatch, isolated_cache):
    monkeypatch.setenv("PINNEAPPLE_ZOO_IGNORE_BUNDLED", "1")
    _fake_hub(monkeypatch, "corrupt")
    with pytest.raises(zoo.ChecksumMismatchError):
        pp.datasets.load("single_wheel_tests", split="all")
    assert not list(isolated_cache.rglob("data.npz"))


def test_corrupt_cache_entry_offline_is_an_error(monkeypatch, isolated_cache):
    monkeypatch.setenv("PINNEAPPLE_ZOO_IGNORE_BUNDLED", "1")
    _fake_hub(monkeypatch, "serve")
    pp.datasets.load("single_wheel_tests", split="all")
    cached = next(isolated_cache.rglob("data.npz"))
    cached.write_bytes(cached.read_bytes()[:-10])
    monkeypatch.setenv("PINNEAPPLE_OFFLINE", "1")
    with pytest.raises(zoo.ChecksumMismatchError):
        pp.datasets.load("single_wheel_tests", split="all")
    monkeypatch.delenv("PINNEAPPLE_OFFLINE")
    assert pp.datasets.load("single_wheel_tests", split="all").n_rows > 20     # re-fetched


def test_not_yet_published_gives_a_clear_message(monkeypatch, isolated_cache):
    monkeypatch.setenv("PINNEAPPLE_ZOO_IGNORE_BUNDLED", "1")
    import urllib.error
    import urllib.request

    def no_hub(*a, **k):
        raise urllib.error.HTTPError(a[0] if a else "", 404, "Not Found", {}, None)

    import builtins
    real_import = builtins.__import__

    def without_hf(name, *args, **kwargs):                       # exercise the plain-HTTPS fallback as well
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_hf)
    monkeypatch.setattr(urllib.request, "urlopen", no_hub)
    with pytest.raises(zoo.ArtifactNotPublishedError, match="not published on the Hugging Face Hub yet"):
        pp.datasets.load("single_wheel_tests", split="all")


# ------------------------------------------------------------------ fine-tuning guard
def test_finetune_guard_refuses_a_worse_model(tmp_path):
    d = pp.datasets.load("rover_contact_grc1_viper", split="test")
    X, Y = d.xy(["slip", "sinkage_m"], ["Fx_N", "Fz_N", "My_Nm"])
    res = zoo.finetune("rover_contact_grc1_viper", (X[:300], 1.3 * Y[:300]), epochs=10, lr=1e-3, replay=False)
    assert not res.accepted and res.after["test/Fx_rmse_N"] > res.before["test/Fx_rmse_N"]
    with pytest.raises(zoo.FinetuneRejectedError):
        res.save(tmp_path / "refused")
    assert not (tmp_path / "refused").exists()
    with pytest.raises(zoo.FinetuneRejectedError):
        zoo.finetune("rover_contact_grc1_viper", (X[:300], 1.3 * Y[:300]), epochs=10, lr=1e-3, replay=False,
                     save_to=tmp_path / "refused2")
    assert not (tmp_path / "refused2").exists()
    p = res.save(tmp_path / "forced", force=True)
    card = json.loads((p / "model_card.json").read_text())
    assert card["data_lineage"]["guard"]["accepted"] is False
    # the original is untouched
    assert zoo.evaluate("rover_contact_grc1_viper")["matches_card"]


def test_finetune_accepted_round_trip_and_registry(tmp_path):
    from pinneapple_registry import ArtifactRegistry
    d = pp.datasets.load("heatsink_forced_convection", split="val")
    res = zoo.finetune("heatsink_forced_mlp", d, epochs=1, lr=0.0, save_to=tmp_path / "ft",
                       registry=ArtifactRegistry(str(tmp_path / "reg")))
    assert res.accepted and res.saved_to
    m2 = zoo.load_dir(tmp_path / "ft")
    assert m2.card.version.startswith("1.0.0+ft.")
    assert zoo.evaluate(m2, "heatsink_forced_convection")["metrics"]["R_mean_abs_pct"] == pytest.approx(
        res.before["test/R_mean_abs_pct"], rel=1e-6)
    reg = ArtifactRegistry(str(tmp_path / "reg"))
    assert reg.models.latest("heatsink_forced_mlp-finetuned")


def test_finetune_replay_masks_nan_targets():
    d = pp.datasets.load("airfoil_rans_cst_re4e6", split="naca_reference")
    res = zoo.finetune("airfoil_cst_rans_mlp", d, epochs=1, lr=0.0, batch=64)
    assert res.accepted and np.isfinite(res.history).all()


# ------------------------------------------------------------------ benchmark and publishing
def test_benchmark_via_pp_benchmark(tmp_path):
    pytest.importorskip("pinneapple_analysis.benchmark")
    res = zoo.benchmark("rover_contact_grc1_viper", budget_s=2, out_dir=tmp_path / "b", log=lambda *a: None)
    md = res.to_markdown()
    assert "Zoo model (pre-trained)" in md and "trained from scratch" in md


def test_benchmark_without_pp_benchmark_is_explicit(monkeypatch):
    import importlib.util
    if importlib.util.find_spec("pinneapple_analysis.benchmark") is not None:
        pytest.skip("pp.benchmark is installed")
    with pytest.raises(ImportError, match="pp.benchmark"):
        zoo.benchmark("rover_contact_grc1_viper", budget_s=1)


def test_publish_script_dry_run_and_no_token(monkeypatch, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("publish_to_hub", Path(__file__).resolve().parents[1] /
                                                  "scripts/zoo/publish_to_hub.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--dry-run"]) == 0
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert mod.main([]) == 0
    assert "HF_TOKEN is not set" in capsys.readouterr().out
