"""Loading a model as a plain function ``(N, n_inputs) array -> {output name: (N,) array}``.

Supported ``path`` forms:

* ``models/surrogate.py:predict``: a Python function taking an ``(N, n_inputs)`` NumPy array and returning an
  ``(N, n_outputs)`` array (or a dict by output name). No torch needed: this is how to check any model, from any
  framework, behind a two-line wrapper.
* ``model.pt`` / ``model.pth`` / ``model.ckpt`` with ``architecture = "<zoo name>"`` (a name of
  ``pp.list_models()``) and ``config = {...}`` (its constructor arguments): a PINNeAPPle checkpoint
  (``BaseModel.save_checkpoint`` or a bare ``state_dict``). Its pickle stream is scanned first (rule PC803) and
  it is loaded with ``weights_only=True``. The training range is read from ``metadata.domain_bounds``.
* ``hub = "org/name"``: ``pinneapple_hub.from_pretrained``; the training range comes from the model card.
* ``zoo = "name"``: a pre-trained, verified model of ``pp.zoo`` (weights checked against the zoo manifest); inputs,
  outputs and the training range (the card's validity range) come from its model card.
"""
from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .numerics import components

__all__ = ["LoadedModel", "load_model", "ModelLoadError", "UnsafeCheckpoint"]


class ModelLoadError(RuntimeError):
    pass


class UnsafeCheckpoint(ModelLoadError):
    def __init__(self, report):
        self.report = report
        super().__init__("checkpoint imports callables outside the allow-list")


class LoadedModel:
    def __init__(self, fn: Callable[[np.ndarray], Any], inputs: list[str], outputs: list[str],
                 train_range: dict[str, list[float]] | None = None, file: Path | None = None):
        self.fn, self.inputs, self.outputs = fn, inputs, outputs
        self.train_range = train_range or {}
        self.file = file
        self.calls = 0

    def __call__(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predictions by output name (vector outputs split: ``U`` -> ``u, v, w``)."""
        self.calls += 1
        Y = self.fn(np.asarray(X, float))
        if isinstance(Y, dict):
            out = {k: np.asarray(v, float).reshape(len(X), -1).squeeze(-1) for k, v in Y.items()}
        else:
            Y = np.asarray(_to_numpy(Y), float).reshape(len(X), -1)
            if Y.shape[1] != len(self.outputs):
                raise ModelLoadError(f"the model returned {Y.shape[1]} columns for {len(self.outputs)} declared "
                                     f"outputs {self.outputs}")
            out = {n: Y[:, i] for i, n in enumerate(self.outputs)}
        alias = {"Ux": "u", "Uy": "v", "Uz": "w", "U_x": "u", "U_y": "v", "U_z": "w"}
        res = components(out, vector_ndim=2)
        for k, v in list(res.items()):
            if k in alias:
                res.setdefault(alias[k], v)
        return res


def _to_numpy(y):
    if hasattr(y, "detach"):
        return y.detach().cpu().numpy()
    return y


def load_model(target) -> LoadedModel:
    if target.zoo:
        return _from_zoo(target)
    if target.hub:
        return _from_hub(target)
    spec = str(target.path)
    if ":" in spec[2:]:
        file, fn = spec.rsplit(":", 1)
        return LoadedModel(_import_callable(Path(file), fn), target.inputs, target.outputs, file=Path(file))
    p = Path(spec)
    if not p.exists():
        raise ModelLoadError(f"{p} does not exist")
    if p.suffix.lower() in (".pt", ".pth", ".ckpt", ".bin"):
        rep = scan(p)                                     # never unpickle a file that can run code
        if not rep.safe:
            raise UnsafeCheckpoint(rep)
        return _from_checkpoint(target, p)
    raise ModelLoadError(f"unsupported model {p.name}: use `file.py:function`, a .pt checkpoint with "
                         "`architecture`, or `hub`")


def _import_callable(file: Path, fn: str) -> Callable:
    if not file.is_file():
        raise ModelLoadError(f"{file} does not exist")
    name = f"_pinneapple_check_model_{abs(hash(str(file.resolve())))}"
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    sys.path.insert(0, str(file.parent))
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        raise ModelLoadError(f"importing {file.name} failed: {e!r}") from e
    finally:
        sys.path.remove(str(file.parent))
    f = getattr(mod, fn, None)
    if not callable(f):
        raise ModelLoadError(f"{file.name} has no function {fn!r}")
    return f


def scan(p: Path):
    from pinneapple_security.model_security import scan_checkpoint
    return scan_checkpoint(str(p))


def _from_checkpoint(target, p: Path) -> LoadedModel:
    try:
        import torch
    except ImportError as e:
        raise ModelLoadError("loading a .pt checkpoint needs torch (or wrap the model as `file.py:function`)") from e
    ck = torch.load(p, map_location="cpu", weights_only=True)
    meta = ck.get("metadata", {}) if isinstance(ck, dict) else {}
    state = ck.get("state_dict", ck.get("model_state", ck)) if isinstance(ck, dict) else ck
    arch = target.architecture or (ck.get("class_name") if isinstance(ck, dict) else None)
    if not arch:
        raise ModelLoadError(f"{p.name}: give `architecture` (a zoo name from pp.list_models())")
    try:
        from pinneapple_neural.architectures.registry import ModelRegistry
        model = ModelRegistry.build(arch, **target.model_config)
    except Exception as e:
        raise ModelLoadError(f"building {arch!r} with {target.model_config}: {e}") from e
    try:
        model.load_state_dict(state)
    except Exception as e:
        raise ModelLoadError(f"{p.name} does not fit {arch!r}: {e}") from e
    model.eval()

    def fn(X):
        with torch.no_grad():
            y = model(torch.as_tensor(X, dtype=torch.float32))
        y = getattr(y, "y", y)
        if isinstance(y, dict):
            return {k: v.detach().cpu().numpy() for k, v in y.items()}
        return y.detach().cpu().numpy()

    rng = {k: [float(a) for a in v] for k, v in (meta.get("domain_bounds") or {}).items()}
    return LoadedModel(fn, target.inputs, target.outputs, rng, file=p)


def _from_hub(target) -> LoadedModel:
    try:
        import torch
        from pinneapple_hub import from_pretrained
    except ImportError as e:
        raise ModelLoadError("hub models need torch and huggingface_hub (pip install 'pinneapple[hub]')") from e
    model, card = from_pretrained(target.hub)
    model.eval()

    def fn(X):
        with torch.no_grad():
            y = model(torch.as_tensor(X, dtype=torch.float32))
        return getattr(y, "y", y).detach().cpu().numpy()

    rng = {k: [float(a) for a in v] for k, v in (getattr(card, "domain_bounds", None) or {}).items()}
    return LoadedModel(fn, target.inputs, target.outputs, rng)


def _from_zoo(target) -> LoadedModel:
    try:
        from pinneapple_hub import zoo
    except ImportError as e:
        raise ModelLoadError("zoo models need torch (pip install torch)") from e
    try:
        m = zoo.load(target.zoo)
    except Exception as e:
        raise ModelLoadError(f"pp.zoo model {target.zoo!r}: {e}") from e
    inputs = target.inputs or list(m.input_names)
    outputs = target.outputs or list(m.output_names)
    if inputs != list(m.input_names):
        raise ModelLoadError(f"pp.zoo model {target.zoo!r} takes inputs {m.input_names}, not {inputs}")

    def fn(X):
        return m.predict(np.asarray(X, float), warn=False)

    rng = {k: [float(a), float(b)] for k, (a, b) in m.bounds().items()}
    return LoadedModel(fn, inputs, outputs, rng)
