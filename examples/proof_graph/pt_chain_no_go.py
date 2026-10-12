"""Proof-graph experiment: does the no-go theorem for linear exceptional-point tuning (PT dimers, Zenodo 23287622)
extend to PT tetramers whose bonds are modulated independently? An open question of the paper (Q2), attacked as a
graph of assumptions, lemmas and a theorem, each with exact (sympy) and numerical checks.

    python examples/proof_graph/pt_chain_no_go.py        # report + docs/assets/proof_graph/pt_chain_no_go.png

Takes a few minutes (the counterexample searches evaluate the matrix threshold thousands of times).
"""
import os

from pinneapple_analysis.proof.pt_chains import build_pt_chain_graph

g = build_pt_chain_graph(max_deg=4, n_numeric=300, search_iters=40)
status = g.run()
print(g.report())
print("\nMermaid:\n" + g.to_mermaid())
out = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "assets", "proof_graph")
os.makedirs(out, exist_ok=True)
g.plot(os.path.join(out, "pt_chain_no_go.png"))
with open(os.path.join(out, "pt_chain_no_go_report.md"), "w") as fh:
    fh.write(g.report())
print("\nstatuses:", status)
