"""Proof graphs: a mathematical argument as a graph of assumptions, definitions, lemmas, theorems and conjectures,
each with machine checks, and a status that only rises as far as its evidence and its dependencies allow.

The same discipline as the physics audit, applied to mathematics: every claim says what supports it.

=================  ==========================================================================================
status             meaning
=================  ==========================================================================================
``ASSUMED``        an assumption or definition: taken as given, stated explicitly
``PROVED``         every check is exact (symbolic identity, exhaustive enumeration over a finite set, a
                   symbolic degree or sign argument) and passed, and every dependency is PROVED or ASSUMED
``ARGUED``         the exact checks passed but part of the argument is a written proof step the machine did not
                   check (``argument=...`` without a check covering it), or a dependency is only ARGUED
``SUPPORTED``      only numerical evidence (random tests, counterexample searches that found nothing): evidence,
                   not proof
``REFUTED``        a check found a counterexample: the claim is false as stated
``OPEN``           no check, or a symbolic check that could not be decided
=================  ==========================================================================================

Checks (each returns a :class:`CheckOutcome`):

* :func:`symbolic_identity`: ``simplify(lhs - rhs) == 0`` (exact);
* :func:`symbolic_true`: a sympy boolean / relation that simplifies to True (exact);
* :func:`exhaustive`: a predicate over every element of a finite set (exact);
* :func:`numeric`: a predicate on random samples (evidence; a failure is a counterexample);
* :func:`counterexample_search`: minimise a violation measure with differential evolution; a value below
  ``-tol`` refutes, otherwise evidence;
* :func:`custom`: anything returning (passed, exact, detail).

A node with no check of its own is the composition of its dependencies (a theorem assembled from lemmas). Nodes
listed in ``supported_by`` corroborate it (numerical searches, independent checks) without lowering its status.

:meth:`ProofGraph.run` evaluates nodes in dependency order; :meth:`ProofGraph.report` writes the argument with its
evidence, :meth:`ProofGraph.to_mermaid` / :meth:`ProofGraph.plot` draw it coloured by status.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["ASSUMED", "PROVED", "ARGUED", "SUPPORTED", "REFUTED", "OPEN", "CheckOutcome", "Check", "Node",
           "ProofGraph", "symbolic_identity", "symbolic_true", "exhaustive", "numeric", "counterexample_search",
           "custom"]

ASSUMED, PROVED, ARGUED, SUPPORTED, REFUTED, OPEN = "ASSUMED", "PROVED", "ARGUED", "SUPPORTED", "REFUTED", "OPEN"
KINDS = ("assumption", "definition", "lemma", "theorem", "corollary", "conjecture")
_RANK = {REFUTED: 0, OPEN: 1, SUPPORTED: 2, ARGUED: 3, PROVED: 4, ASSUMED: 5}
COLORS = {ASSUMED: "#9AA5B1", PROVED: "#2E9E5B", ARGUED: "#7CC47F", SUPPORTED: "#F2B233", REFUTED: "#D9534F",
          OPEN: "#B0B0B0"}


@dataclass
class CheckOutcome:
    name: str
    passed: bool | None          # None: undecided
    exact: bool                  # an exact (proof-grade) check, or numerical evidence
    detail: str = ""
    counterexample: Any = None
    seconds: float = 0.0


@dataclass
class Check:
    name: str
    fn: Callable[[], CheckOutcome]

    def __call__(self) -> CheckOutcome:
        t0 = time.perf_counter()
        try:
            out = self.fn()
        except Exception as exc:  # noqa: BLE001 - a check that crashes is undecided, not passed
            out = CheckOutcome(self.name, None, False, f"check raised {type(exc).__name__}: {exc}")
        out.name = self.name
        out.seconds = time.perf_counter() - t0
        return out


def symbolic_identity(name: str, lhs: Any, rhs: Any = 0, simplify: Callable | None = None) -> Check:
    def fn():
        import sympy as sp

        d = (simplify or sp.simplify)(sp.expand(sp.sympify(lhs) - sp.sympify(rhs)))
        if d == 0:
            return CheckOutcome(name, True, True, "simplifies to 0")
        if d.is_number and d != 0:
            return CheckOutcome(name, False, True, f"differs by {d}", counterexample=d)
        return CheckOutcome(name, None, True, f"could not reduce to 0: {d}")
    return Check(name, fn)


def symbolic_true(name: str, statement: Any) -> Check:
    def fn():
        import sympy as sp

        s = sp.simplify(statement) if not isinstance(statement, bool) else statement
        if s is True or s == sp.true:
            return CheckOutcome(name, True, True, "holds")
        if s is False or s == sp.false:
            return CheckOutcome(name, False, True, "is false")
        return CheckOutcome(name, None, True, f"undecided: {s}")
    return Check(name, fn)


def exhaustive(name: str, predicate: Callable[[Any], bool], domain: Iterable[Any], describe: str = "") -> Check:
    def fn():
        n = 0
        for x in domain:
            n += 1
            if not predicate(x):
                return CheckOutcome(name, False, True, f"fails at {x}", counterexample=x)
        return CheckOutcome(name, True, True, f"holds for all {n} cases{(' (' + describe + ')') if describe else ''}")
    return Check(name, fn)


def numeric(name: str, predicate: Callable[[np.ndarray], bool], sampler: Callable[[np.random.Generator], np.ndarray],
            n: int = 2000, seed: int = 0) -> Check:
    def fn():
        rng = np.random.default_rng(seed)
        for _ in range(n):
            x = sampler(rng)
            if not predicate(x):
                return CheckOutcome(name, False, False, f"counterexample {x}", counterexample=x)
        return CheckOutcome(name, True, False, f"holds on {n} random samples")
    return Check(name, fn)


def counterexample_search(name: str, violation: Callable[[np.ndarray], float], bounds: Sequence[tuple[float, float]],
                          tol: float = 1e-8, seed: int = 0, maxiter: int = 300, restarts: int = 3) -> Check:
    """``violation(x) < -tol`` means x is a counterexample. Differential evolution minimises it from ``restarts``
    seeds; the smallest value found is reported."""
    def fn():
        from scipy.optimize import differential_evolution

        best = None
        for r in range(restarts):
            res = differential_evolution(violation, bounds, seed=seed + r, maxiter=maxiter, tol=1e-12, polish=True)
            if best is None or res.fun < best.fun:
                best = res
        if best.fun < -tol:
            return CheckOutcome(name, False, False, f"counterexample with violation {best.fun:.3g}",
                                counterexample=best.x)
        return CheckOutcome(name, True, False, f"no counterexample; smallest margin found {best.fun:.4g} "
                                                f"({restarts} searches)")
    return Check(name, fn)


def custom(name: str, fn: Callable[[], tuple[bool | None, bool, str]]) -> Check:
    def run():
        passed, exact, detail = fn()
        return CheckOutcome(name, passed, exact, detail)
    return Check(name, run)


@dataclass
class Node:
    id: str
    kind: str
    statement: str
    depends_on: list[str] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    argument: str = ""                # written proof step not covered by a check (makes PROVED -> ARGUED)
    covered: bool = True              # False: ``argument`` contains steps no check covers
    supported_by: list[str] = field(default_factory=list)   # corroborating nodes: shown, never lower the status
    status: str = OPEN
    outcomes: list[CheckOutcome] = field(default_factory=list)
    reason: str = ""


class ProofGraph:
    def __init__(self, title: str = ""):
        self.title = title
        self.nodes: dict[str, Node] = {}

    def add(self, id: str, kind: str, statement: str, depends_on: Sequence[str] = (), checks: Sequence[Check] = (),
            argument: str = "", covered: bool = True, supported_by: Sequence[str] = ()) -> Node:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if id in self.nodes:
            raise ValueError(f"duplicate node {id!r}")
        n = Node(id, kind, statement, list(depends_on), list(checks), argument, covered, list(supported_by))
        self.nodes[id] = n
        return n

    def assume(self, id: str, statement: str) -> Node:
        return self.add(id, "assumption", statement)

    def define(self, id: str, statement: str) -> Node:
        return self.add(id, "definition", statement)

    def _order(self) -> list[str]:
        seen, order, stack = set(), [], set()

        def visit(i):
            if i in stack:
                raise ValueError(f"circular dependency through {i!r}")
            if i in seen:
                return
            if i not in self.nodes:
                raise KeyError(f"unknown dependency {i!r}")
            stack.add(i)
            for d in self.nodes[i].depends_on + self.nodes[i].supported_by:
                visit(d)
            stack.discard(i)
            seen.add(i)
            order.append(i)
        for i in self.nodes:
            visit(i)
        return order

    def run(self) -> dict[str, str]:
        for i in self._order():
            n = self.nodes[i]
            if n.kind in ("assumption", "definition"):
                n.status, n.reason = ASSUMED, "stated"
                continue
            n.outcomes = [c() for c in n.checks]
            deps = [self.nodes[d].status for d in n.depends_on]
            if any(o.passed is False for o in n.outcomes):
                n.status, n.reason = REFUTED, "a check failed: " + "; ".join(
                    f"{o.name}: {o.detail}" for o in n.outcomes if o.passed is False)
                continue
            if not n.outcomes and not n.argument and not n.depends_on:
                n.status, n.reason = OPEN, "no check, no argument and no dependencies"
                continue
            exact = [o for o in n.outcomes if o.exact]
            undecided = [o for o in n.outcomes if o.passed is None]
            evidence_only = n.outcomes and not exact
            if undecided and not [o for o in exact if o.passed]:
                n.status, n.reason = OPEN, "undecided checks: " + "; ".join(o.detail for o in undecided)
                continue
            if not n.outcomes and not n.argument:
                own = PROVED if n.kind != "conjecture" else SUPPORTED   # pure composition of its dependencies
            elif evidence_only and not n.argument:
                own = SUPPORTED
            elif n.argument and (not n.covered or not exact):
                own = ARGUED
            else:
                own = PROVED
            weakest = min(deps + [own], key=_RANK.get) if deps else own
            if weakest in (REFUTED, OPEN):
                n.status, n.reason = OPEN, "depends on a claim that is " + weakest.lower()
            else:
                n.status = weakest
                n.reason = {PROVED: "all checks exact and passed, dependencies proved",
                            ARGUED: "checks passed; part of the argument is a written step not machine-checked"
                                    if own == ARGUED else "a dependency is only argued",
                            SUPPORTED: "numerical evidence only" if own == SUPPORTED else
                                       "a dependency rests on numerical evidence only"}[n.status]
        return {i: n.status for i, n in self.nodes.items()}

    # ------------------------------------------------------------------------------------------ output
    def report(self) -> str:
        out = [f"# {self.title}" if self.title else "# Proof graph", ""]
        counts = {}
        for n in self.nodes.values():
            counts[n.status] = counts.get(n.status, 0) + 1
        out.append("Status: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -_RANK[kv[0]])))
        out.append("")
        for i in self._order():
            n = self.nodes[i]
            out.append(f"## [{n.status}] {n.kind} {n.id}")
            out.append(n.statement)
            if n.depends_on:
                out.append(f"- depends on: {', '.join(n.depends_on)}")
            if n.supported_by:
                out.append(f"- corroborated by: {', '.join(f'{s} [{self.nodes[s].status}]' for s in n.supported_by)}")
            for o in n.outcomes:
                tag = "exact" if o.exact else "numerical"
                res = {True: "passed", False: "FAILED", None: "undecided"}[o.passed]
                out.append(f"- check ({tag}) {o.name}: {res}, {o.detail} [{o.seconds:.2f} s]")
            if n.argument:
                out.append(f"- argument{' (not machine-checked)' if not n.covered else ''}: {n.argument}")
            if n.reason and n.kind not in ("assumption", "definition"):
                out.append(f"- status because: {n.reason}")
            out.append("")
        return "\n".join(out)

    def to_mermaid(self) -> str:
        ids = {i: f"n{k}" for k, i in enumerate(self.nodes)}
        lines = ["graph TD"]
        for i, n in self.nodes.items():
            lines.append(f'  {ids[i]}["{n.kind} {i}<br/>{n.status}"]')
            lines.append(f"  style {ids[i]} fill:{COLORS[n.status]}")
        for i, n in self.nodes.items():
            for d in n.depends_on:
                lines.append(f"  {ids[d]} --> {ids[i]}")
            for d in n.supported_by:
                lines.append(f"  {ids[d]} -.-> {ids[i]}")
        return "\n".join(lines)

    def plot(self, path: str, figsize=(12, 7.5)) -> str:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx

        g = nx.DiGraph()
        for i, n in self.nodes.items():
            g.add_node(i)
            for d in n.depends_on:
                g.add_edge(d, i, style="solid")
            for d in n.supported_by:
                g.add_edge(d, i, style="dashed")
        layers: dict[str, int] = {}
        for i in self._order():
            layers[i] = 1 + max((layers[d] for d in self.nodes[i].depends_on + self.nodes[i].supported_by), default=-1)
        for i in g.nodes:
            g.nodes[i]["layer"] = layers[i]
        pos = nx.multipartite_layout(g, subset_key="layer", align="horizontal")
        pos = {k: (v[0], -v[1]) for k, v in pos.items()}
        fig, ax = plt.subplots(figsize=figsize)
        for style in ("solid", "dashed"):
            es = [(u, v) for u, v, d in g.edges(data=True) if d["style"] == style]
            nx.draw_networkx_edges(g, pos, edgelist=es, ax=ax, arrows=True, arrowsize=12, edge_color="#6B7280",
                                   width=1.0, node_size=2600, style=style)
        for i, (x, y) in pos.items():
            n = self.nodes[i]
            ax.text(x, y, f"{i}\n{n.status}", ha="center", va="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.45", fc=COLORS[n.status], ec="#1F2937", lw=0.8))
        handles = [plt.Line2D([0], [0], marker="s", ls="", ms=10, mfc=c, mec="#1F2937", label=s)
                   for s, c in COLORS.items() if any(n.status == s for n in self.nodes.values())]
        ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
        ax.set_title(self.title, fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
