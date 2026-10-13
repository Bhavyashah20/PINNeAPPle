"""``pp.datasets``: ready-made, verified datasets in one line.

    import pinneapple as pp
    pp.datasets.list()                                   # names
    pp.datasets.info("airfoil_rans_cst_re4e6")           # schema, splits, provenance, verification, licence
    test = pp.datasets.load("airfoil_rans_cst_re4e6", split="test")
    test["Cl"], test.columns, test.xy(["T0", "alpha_deg"], ["Cl"])

Each dataset is stored as npz files (no pickles) whose arrays are named ``<split>__<column>``: either one ``data.npz``
holding every split, or one ``<split>.npz`` per split (large datasets; a split may be sharded into ``<split>-NN.npz``,
concatenated along the rows on load, and only the requested split is downloaded). Small files are bundled with the
package; the others are downloaded once from the Hugging Face Hub (``PINNeAPPle-Labs/<name>``), checked against the
SHA-256 in the manifest and cached (see ``pinneapple_hub.zoo._artifacts``).
"""
from __future__ import annotations

import builtins
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import _artifacts as art

__all__ = ["list", "info", "load", "DatasetInfo", "DatasetSplit", "status", "split_files"]


@dataclass
class DatasetInfo:
    name: str
    version: str
    title: str
    description: str
    license: str
    source: str
    verification: str
    splits: dict[str, dict[str, Any]]                # split -> {"n": rows, "description": ...}
    columns: dict[str, dict[str, str]]               # column -> {"unit", "description", "dtype"}
    citation: str = ""
    tags: builtins.list[str] = field(default_factory=builtins.list)
    hf_repo: str = ""
    revision: str = ""
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    split_rule: str = ""

    def to_markdown(self) -> str:
        size = sum(f.get("size", 0) for f in self.files.values())
        where = "bundled with the package" if all(f.get("bundled") for f in self.files.values()) else \
            "downloaded on first use"
        lines = [f"# {self.title}", "", self.description, "",
                 f"`{self.name}` v{self.version} · licence {self.license} · Hub `{self.hf_repo}` @ {self.revision} · "
                 f"{len(self.files)} file(s), {size / 1e6:.1f} MB, {where}", "",
                 "## Splits", "", "| Split | Rows | Content |", "|---|---|---|",
                 *(f"| `{s}` | {d['n']} | {d['description']} |" for s, d in self.splits.items()), "",
                 "## Columns", "", "| Column | Unit | Type | Description |", "|---|---|---|---|",
                 *(f"| `{c}` | {d['unit']} | {d['dtype']} | {d['description']} |" for c, d in self.columns.items()),
                 "", "## Provenance", "", self.source, "", "## Verification", "", self.verification, ""]
        if self.split_rule:
            lines += ["Split rule: " + self.split_rule, ""]
        lines += ["## Citation", "", self.citation or "(none)", ""]
        return "\n".join(lines)


class DatasetSplit(dict):
    """One split: ``{column: np.ndarray}`` (all columns have ``n_rows`` entries) plus ``name``, ``split``, ``info``."""

    def __init__(self, data: dict[str, np.ndarray], name: str, split: str, ds_info: DatasetInfo):
        super().__init__(data)
        self.name, self.split, self.info = name, split, ds_info

    @property
    def columns(self) -> builtins.list[str]:
        return builtins.list(self.keys())

    @property
    def n_rows(self) -> int:
        return len(next(iter(self.values()))) if self else 0

    def xy(self, inputs: Sequence[str], outputs: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """(n, len(inputs)) and (n, len(outputs)) float arrays; a missing output column is all NaN."""
        missing = [c for c in inputs if c not in self]
        if missing:
            raise KeyError(f"{self.name}/{self.split} has no columns {missing}; columns: {self.columns}")
        X = np.stack([np.asarray(self[c], float) for c in inputs], 1)
        Y = np.stack([np.asarray(self[c], float) if c in self else np.full(self.n_rows, np.nan) for c in outputs], 1)
        return X, Y

    def to_pandas(self):
        import pandas as pd
        return pd.DataFrame(dict(self))

    def __repr__(self) -> str:
        return f"<{self.name}[{self.split}]: {self.n_rows} rows x {len(self)} columns {self.columns}>"


def list() -> builtins.list[str]:  # noqa: A001 - mirrors pp.zoo.list()
    """Names of the shipped datasets."""
    return sorted(art.manifest()["datasets"])


def info(name: str) -> DatasetInfo:
    e = art.entry("datasets", name)
    return DatasetInfo(name=name, version=e["version"], title=e["title"], description=e["description"],
                       license=e["license"], source=e["source"], verification=e["verification"], splits=e["splits"],
                       columns=e["columns"], citation=e.get("citation", ""), tags=e.get("tags", []),
                       hf_repo=e["hf_repo"], revision=e["revision"], files=e["files"],
                       split_rule=e.get("split_rule", ""))


def split_files(name: str, split: str | None = None) -> builtins.list[str]:
    """Files holding ``split`` (all files when None): ``data.npz``, or ``<split>.npz`` / ``<split>-NN.npz``."""
    files = sorted(art.entry("datasets", name)["files"])
    if "data.npz" in files or split is None:
        return files
    return [f for f in files if re.fullmatch(rf"{re.escape(split)}(-\d+)?\.npz", f)]


def _read(name: str, offline: bool | None, split: str | None = None) -> dict[str, dict[str, np.ndarray]]:
    parts: dict[str, dict[str, builtins.list[np.ndarray]]] = {}
    for fn in split_files(name, split):
        path = art.fetch("datasets", name, fn, offline=offline)
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                s, col = key.split("__", 1)
                parts.setdefault(s, {}).setdefault(col, []).append(z[key])
    return {s: {c: v[0] if len(v) == 1 else np.concatenate(v) for c, v in cols.items()} for s, cols in parts.items()}


def load(name: str, split: str | None = None, *, offline: bool | None = None):
    """A split (:class:`DatasetSplit`), or ``{split: DatasetSplit}`` for all of them when ``split`` is None."""
    ds_info = info(name)
    if split is not None and split not in ds_info.splits:
        raise KeyError(f"{name} has no split {split!r}; splits: {sorted(ds_info.splits)}")
    data = _read(name, offline, split)
    if split is None:
        return {s: DatasetSplit(d, name, s, ds_info) for s, d in data.items()}
    if split not in data:
        raise KeyError(f"{name} has no split {split!r}; splits: {sorted(data)}")
    return DatasetSplit(data[split], name, split, ds_info)


def status(name: str) -> dict[str, str]:
    """Where the files resolve from right now (bundled | cached | remote), without downloading."""
    return art.status("datasets", name)


def _register_with_data_registry() -> None:
    """Expose the zoo datasets through ``pinneapple_data.datasets.DatasetRegistry`` as ``zoo/<name>``."""
    try:
        from pinneapple_data.datasets.registry import DatasetInfo as _DI
        from pinneapple_data.datasets.registry import DatasetRegistry
    except Exception:  # pragma: no cover - pinneapple_data is optional here
        return
    for n in list():
        e = art.entry("datasets", n)
        DatasetRegistry.register(
            _DI(id=f"zoo/{n}", name=e["title"], category="zoo", description=e["description"],
                fields=sorted(e["columns"]), tags=e.get("tags", []), license=e["license"],
                reference=e.get("citation", "")),
            lambda split="test", _n=n, **kw: dict(load(_n, split=split, **kw)))


_register_with_data_registry()
