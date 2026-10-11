"""Proof graphs: statuses from checks and dependencies, and the PT-tetramer no-go graph."""
import numpy as np
import pytest
import sympy as sp

from pinneapple_analysis.proof import (
    ARGUED,
    ASSUMED,
    OPEN,
    PROVED,
    REFUTED,
    SUPPORTED,
    ProofGraph,
    counterexample_search,
    custom,
    exhaustive,
    numeric,
    symbolic_identity,
    symbolic_true,
)

x = sp.Symbol("x", real=True)


def test_statuses_and_propagation(tmp_path):
    g = ProofGraph("toy")
    g.assume("A", "x is real")
    g.add("L1", "lemma", "(x+1)^2 = x^2 + 2x + 1", ["A"], [symbolic_identity("expand", (x + 1) ** 2, x ** 2 + 2 * x + 1)])
    g.add("L2", "lemma", "x^2 >= 0 numerically", ["A"],
          [numeric("random", lambda v: v[0] ** 2 >= 0, lambda r: r.normal(size=1), n=200)])
    g.add("L3", "lemma", "false claim: x^2 > x for all x", ["A"],
          [numeric("random", lambda v: v[0] ** 2 > v[0], lambda r: r.uniform(-2, 2, size=1), n=500)])
    g.add("L4", "lemma", "squares of 0..9 end in 0,1,4,5,6,9", [],
          [exhaustive("all digits", lambda k: (k * k) % 10 in {0, 1, 4, 5, 6, 9}, range(10))])
    g.add("L5", "lemma", "written step", ["L1"], [symbolic_true("2 > 1", sp.Integer(2) > 1)],
          argument="a step the machine did not check", covered=False)
    g.add("T1", "theorem", "composition of L1 and L4", ["L1", "L4"], supported_by=["L2"])
    g.add("T2", "theorem", "uses a refuted lemma", ["L3"])
    g.add("T3", "theorem", "uses an argued lemma", ["L5", "L1"])
    g.add("C1", "conjecture", "evidence only", ["L1"], [numeric("r", lambda v: True, lambda r: r.normal(size=1), n=5)])
    st = g.run()
    assert st["A"] == ASSUMED and st["L1"] == PROVED and st["L2"] == SUPPORTED and st["L3"] == REFUTED
    assert st["L4"] == PROVED and st["L5"] == ARGUED
    assert st["T1"] == PROVED                       # corroboration (L2, numerical) does not lower a proof
    assert st["T2"] == OPEN and st["T3"] == ARGUED and st["C1"] == SUPPORTED
    assert g.nodes["L3"].outcomes[0].counterexample is not None
    rep = g.report()
    assert "[REFUTED] lemma L3" in rep and "corroborated by: L2 [SUPPORTED]" in rep
    assert g.to_mermaid().startswith("graph TD") and "-.->" in g.to_mermaid()
    pytest.importorskip("matplotlib")
    pytest.importorskip("networkx")
    assert (tmp_path / "g.png").exists() is False
    g.plot(str(tmp_path / "g.png"))
    assert (tmp_path / "g.png").stat().st_size > 10_000


def test_undecided_crash_and_errors():
    g = ProofGraph()
    g.add("U", "lemma", "undecidable here", [], [symbolic_identity("opaque", sp.Function("f")(x), 0)])
    g.add("K", "lemma", "crashing check", [], [custom("boom", lambda: 1 / 0)])
    g.add("N", "lemma", "nothing", [])
    st = g.run()
    assert st["U"] == OPEN and st["K"] == OPEN and st["N"] == OPEN
    g.add("Z", "lemma", "cycle", ["Z"])
    with pytest.raises(ValueError):
        g.run()
    with pytest.raises(ValueError):
        ProofGraph().add("a", "axiom", "bad kind")


def test_counterexample_search_finds_and_misses():
    g = ProofGraph()
    g.add("F", "lemma", "x^2 - 2x >= 0 on [0, 3] (false at x = 1)", [],
          [counterexample_search("de", lambda v: v[0] ** 2 - 2 * v[0], [(0, 3)], maxiter=50, restarts=1)])
    g.add("T", "lemma", "x^2 + 1 > 0", [],
          [counterexample_search("de", lambda v: v[0] ** 2 + 1, [(-3, 3)], maxiter=30, restarts=1)])
    st = g.run()
    assert st["F"] == REFUTED and np.isclose(g.nodes["F"].outcomes[0].counterexample[0], 1.0, atol=1e-3)
    assert st["T"] == SUPPORTED


def test_pt_tetramer_graph_without_searches():
    from pinneapple_analysis.proof.pt_chains import (
        build_pt_chain_graph,
        tetramer_threshold_branches,
        threshold_numeric,
    )

    g = build_pt_chain_graph(max_deg=2, n_numeric=30, searches=False)
    st = g.run()
    assert st["L1"] == st["L2"] == PROVED
    assert st["L3"] == ARGUED and st["T1"] == ARGUED            # the continuity step of L3 is written, not checked
    for lem in ("L4", "L5"):                                   # own checks all exact and passed; capped by L3
        assert st[lem] == ARGUED and all(o.exact and o.passed for o in g.nodes[lem].outcomes)
        assert g.nodes[lem].reason == "a dependency is only argued"
    # the matrix threshold is one of the branch values
    for A, B in [(1.0, 0.3), (0.5, 1.2), (2.0, 0.9)]:
        num = threshold_numeric([np.sqrt(A), np.sqrt(B), np.sqrt(A)]) ** 2
        assert min(abs(num - c) for c in tetramer_threshold_branches(A, B)) < 1e-6
