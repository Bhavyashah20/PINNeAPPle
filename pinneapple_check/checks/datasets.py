"""Checks of a training dataset (``[[dataset]]``): the physics audit of :mod:`pinneapple_data.audit` as rules PC451
(physics and declared constraints), PC452 (diversity and duplicates), PC453 (train/test leakage), with non-finite
values under PC403 and units under PC501."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import _reuse
from ..config import DatasetTarget
from .common import Ctx

__all__ = ["check_dataset"]

_RULE = {"physics": "PC451", "constraint": "PC451", "diversity": "PC452", "leakage": "PC453", "units": "PC501",
         "finite": "PC403"}


def _read(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """``<split>__<column>`` arrays of an .npz (the zoo layout); plain keys go to a single split ``train``."""
    out: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as z:
        for k in z.files:
            split, col = k.split("__", 1) if "__" in k else ("train", k)
            out.setdefault(split, {})[col] = z[k]
    return out


def check_dataset(ctx: Ctx, t: DatasetTarget) -> None:
    audit = _reuse.dataset_audit()
    try:
        if t.zoo:
            card = audit.audit_zoo_dataset(t.zoo, **({"tolerances": t.audit} if t.audit else {}))
        else:
            splits = _read(t.path)
            card = audit.audit_dataset(splits, name=t.name, physics=t.physics, inputs=t.inputs or None,
                                       units=t.units or None, constraints=t.constraints, train=t.train,
                                       test=t.test, tolerances=t.audit or None)
    except Exception as e:
        ctx.emit("PC001", "read", False, f"cannot audit dataset {t.name}: {e}")
        return
    ctx.emit("PC001", "read", True, f"read dataset {t.name} ({', '.join(f'{s} {n}' for s, n in card.splits.items())})")
    loc = ctx.default_location()
    for c in card.checks:
        if c.status == "not-applicable":
            continue
        rule = _RULE[c.id.split(".")[0].split(":")[0]]
        ctx.emit(rule, c.id, c.status != "fail", f"{c.id}: {c.summary}", loc, value=c.value, threshold=c.threshold)
    substantive = ctx.ran - {"PC001"}
    ctx.emit("PC002", "checks", bool(substantive), f"checked by {', '.join(sorted(substantive))}" if substantive else
             f"no audit check applies to dataset {t.name}")
