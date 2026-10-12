"""Proof graphs: mathematical arguments as graphs of assumptions, lemmas and theorems with machine checks
(symbolic, exhaustive, numerical, counterexample search) and statuses that propagate along the dependencies.

:mod:`.pt_chains` applies it to an open question of the PT-symmetric exceptional-point literature: whether the no-go
theorem for linear EP tuning in dimers extends to longer chains.
"""
from .graph import (
    ARGUED,
    ASSUMED,
    OPEN,
    PROVED,
    REFUTED,
    SUPPORTED,
    Check,
    CheckOutcome,
    Node,
    ProofGraph,
    counterexample_search,
    custom,
    exhaustive,
    numeric,
    symbolic_identity,
    symbolic_true,
)

__all__ = ["ASSUMED", "PROVED", "ARGUED", "SUPPORTED", "REFUTED", "OPEN", "Check", "CheckOutcome", "Node",
           "ProofGraph", "symbolic_identity", "symbolic_true", "exhaustive", "numeric", "counterexample_search",
           "custom"]
