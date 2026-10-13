"""JSON-safe copies of results: NumPy scalars and arrays to Python, NaN/Inf to ``None``, optional rounding.

Every service that returns a result as JSON (the apps, the MCP server, the lab, the data tools) needs the same
conversion; this is the one implementation::

    from pinneapple_core.jsonable import to_jsonable
    json.dumps(to_jsonable({"cl": np.float32(0.61), "field": np.zeros(3), "bad": float("nan")}))
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any

import numpy as np

__all__ = ["to_jsonable", "round_sig"]

_SCALARS = (str, int, bool, type(None))


def round_sig(x: float, digits: int = 6, small: float = 1e-3) -> float:
    """``round(x, digits)`` for ``|x| >= small``, else ``digits`` significant digits (keeps small values exact)."""
    return round(x, digits) if abs(x) >= small else float(f"{x:.{digits}g}")


def to_jsonable(obj: Any, *, digits: int | None = None, nonfinite: Any = None, drop_private: bool = False,
                unknown: str = "keep") -> Any:
    """A copy of ``obj`` that ``json.dumps`` accepts.

    - mappings become dicts with ``str`` keys, lists / tuples / arrays become lists;
    - NumPy scalars become ``int`` / ``float`` / ``bool``, paths become ``str``;
    - non-finite floats become ``nonfinite`` (``None`` by default; pass ``"str"`` for ``"nan"`` / ``"inf"``);
    - ``digits``: round floats with :func:`round_sig` (``round(x, digits)``, significant digits below 1e-3);
    - ``drop_private``: leave out mapping keys starting with ``"_"`` (internal arrays of a report);
    - ``unknown``: what to do with any other object: ``"keep"`` it, or replace it by its ``"repr"`` / ``"str"``.
    """
    kw = dict(digits=digits, nonfinite=nonfinite, drop_private=drop_private, unknown=unknown)
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v, **kw) for k, v in obj.items()
                if not (drop_private and str(k).startswith("_"))}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, **kw) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist(), **kw)
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return str(obj) if nonfinite == "str" else nonfinite
        return round_sig(obj, digits) if digits is not None else obj
    if isinstance(obj, _SCALARS):
        return obj
    if isinstance(obj, PurePath):
        return str(obj)
    if unknown == "repr":
        return repr(obj)
    if unknown == "str":
        return str(obj)
    return obj
