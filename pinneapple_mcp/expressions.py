"""Safe arithmetic expressions of the coordinates, e.g. ``"sin(pi*x)*sinh(pi*y)/sinh(pi)"``.

Agents describe boundary values, sources and closed-form references as text. The text is parsed with ``ast`` and
only numbers, the coordinate names, ``pi`` / ``e``, ``+ - * / ** %`` and a fixed list of NumPy functions are
accepted: no attribute access, no subscripts, no other names, nothing is passed to ``eval``.
"""
from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Sequence

import numpy as np

FUNCTIONS = {n: getattr(np, n) for n in ("sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
                                         "exp", "log", "log10", "sqrt", "abs", "sign", "floor", "ceil")}
FUNCTIONS.update({"arctan2": np.arctan2, "minimum": np.minimum, "maximum": np.maximum, "where": np.where,
                  "heaviside": lambda x: np.heaviside(x, 0.5)})
CONSTANTS = {"pi": np.pi, "e": np.e}
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
MAX_LENGTH = 1000


def _check(node: ast.AST, names: Sequence[str]) -> None:
    if isinstance(node, ast.Expression):
        return _check(node.body, names)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise ValueError(f"only numbers are allowed as constants, got {node.value!r}")
        return
    if isinstance(node, ast.Name):
        if node.id not in names and node.id not in CONSTANTS:
            raise ValueError(f"unknown name '{node.id}'; allowed: {list(names)} and {list(CONSTANTS)}")
        return
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        _check(node.left, names)
        return _check(node.right, names)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _check(node.operand, names)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMP:
        _check(node.left, names)
        return _check(node.comparators[0], names)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS \
            and not node.keywords:
        for a in node.args:
            _check(a, names)
        return
    raise ValueError(f"'{ast.dump(node)[:80]}' is not allowed; use numbers, {list(names)}, pi, e, + - * / ** % "
                     f"and the functions {sorted(FUNCTIONS)}")


def _eval(node: ast.AST, env: dict):
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return env[node.id] if node.id in env else CONSTANTS[node.id]
    if isinstance(node, ast.BinOp):
        return _BIN[type(node.op)](_eval(node.left, env), _eval(node.right, env))
    if isinstance(node, ast.UnaryOp):
        return _UN[type(node.op)](_eval(node.operand, env))
    if isinstance(node, ast.Compare):
        res = _CMP[type(node.ops[0])](_eval(node.left, env), _eval(node.comparators[0], env))
        return np.asarray(res, dtype=np.float64)
    if isinstance(node, ast.Call):
        return FUNCTIONS[node.func.id](*[_eval(a, env) for a in node.args])
    raise ValueError("unreachable")  # pragma: no cover - _check rejects everything else


def compile_expression(text: str, coords: Sequence[str]) -> Callable[[np.ndarray], np.ndarray]:
    """``f(X) -> (N,)`` for points ``X`` of shape ``(N, len(coords))``. Raises ``ValueError`` on anything that
    is not plain arithmetic of the coordinates."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("expression must be a non-empty string")
    if len(text) > MAX_LENGTH:
        raise ValueError(f"expression longer than {MAX_LENGTH} characters")
    tree = ast.parse(text.strip(), mode="eval")
    _check(tree, list(coords))
    names = list(coords)

    def fn(X, *_args) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        env = {n: X[:, i] for i, n in enumerate(names)}
        out = _eval(tree, env)
        return np.broadcast_to(np.asarray(out, dtype=np.float64), (X.shape[0],)).copy()

    fn.expression = text  # type: ignore[attr-defined]
    return fn


def value(v, coords: Sequence[str]):
    """A number stays a number; a string becomes ``compile_expression``."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return compile_expression(v, coords)
    raise ValueError(f"expected a number or an expression string, got {type(v).__name__}")
