"""Reporting: figure style and PDF technical-report builder.

``paperkit``
    One matplotlib style for report figures (fixed categorical order, single-hue sequential and a
    blue/red diverging colormap) and :class:`~pinneapple_tools.reporting.paperkit.Paper`, a small PDF
    builder for numbered technical reports. The PDF part needs ``reportlab``
    (``pip install "pinneapple[veriphysics]"``); the figure helpers need only matplotlib.
"""
from __future__ import annotations

from .paperkit import (
    CAT,
    DIV,
    GRID,
    INK,
    INK2,
    MUTED,
    SEQ,
    Paper,
    fmt,
    sanitize,
    savefig,
    setup_mpl,
)

__all__ = ["CAT", "DIV", "GRID", "INK", "INK2", "MUTED", "SEQ", "Paper", "fmt", "sanitize", "savefig", "setup_mpl"]
