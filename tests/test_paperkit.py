"""Tests for pinneapple_tools.reporting.paperkit (figure style + PDF report builder)."""
from __future__ import annotations

import math
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from pinneapple_tools.reporting import paperkit as pk  # noqa: E402


@pytest.mark.parametrize("x, nd, expected", [
    (None, 3, "—"),
    (math.nan, 3, "n/a"),
    (0, 3, "0.000"),
    (0.5, 3, "0.500"),
    (1.23456, 2, "1.23"),
    (-2.5, 1, "-2.5"),
    (12345.0, 3, "1.23e+04"),
    (1e4, 3, "1.00e+04"),
    (9999.0, 1, "9999.0"),
    (0.001, 3, "0.001"),
    (0.000999, 3, "9.99e-04"),
    (-1.5e-7, 3, "-1.50e-07"),
    ("3.5", 2, "3.50"),
    ("FEM", 3, "FEM"),
    ([1, 2], 3, "[1, 2]"),
])
def test_fmt(x, nd, expected):
    assert pk.fmt(x, nd) == expected


@pytest.mark.parametrize("text, expected", [
    ("plain text", "plain text"),
    ("", ""),
    ("10⁻³", "10<super>-3</super>"),
    ("u₀ + u₁₂", "u<sub>0</sub> + u<sub>12</sub>"),
    ("xⁿ⁺¹ᵢ", "x<super>n+1</super>ᵢ"),          # U+1D62 (subscript i) has no mapping and passes through
    ("Re = 4·10⁶, k₀", "Re = 4·10<super>6</super>, k<sub>0</sub>"),
    ("<b>a²</b>", "<b>a<super>2</super></b>"),   # existing markup is left alone
    ("x²₃", "x<super>2</super><sub>3</sub>"),      # adjacent runs of each kind stay separate
])
def test_sanitize(text, expected):
    assert pk.sanitize(text) == expected


def test_palettes_and_style(tmp_path):
    assert len(pk.CAT) == 8 and len(set(pk.CAT)) == 8
    assert all(c.startswith("#") and len(c) == 7 for c in pk.CAT + [pk.INK, pk.INK2, pk.MUTED, pk.GRID])
    # sequential ramp gets darker from 0 to 1; diverging ramp has a near-neutral midpoint
    lum = [sum(pk.SEQ(v)[:3]) for v in (0.0, 0.5, 1.0)]
    assert lum[0] > lum[1] > lum[2]
    r, g, b, _ = pk.DIV(0.5)
    assert max(r, g, b) - min(r, g, b) < 0.05

    with plt.rc_context():
        pk.setup_mpl()
        pk.setup_mpl()                                    # idempotent: registering twice is fine
        assert "seq_blue" in matplotlib.colormaps and "div_blue_red" in matplotlib.colormaps
        assert plt.rcParams["image.cmap"] == "seq_blue"
        assert plt.rcParams["axes.prop_cycle"].by_key()["color"][:2] == pk.CAT[:2]

        fig, ax = plt.subplots(figsize=(2, 1.5))
        ax.plot([0, 1], [0, 1])
        png = pk.savefig(fig, tmp_path / "sub" / "fig.png")
        assert png == str(tmp_path / "sub" / "fig.png")
        assert (tmp_path / "sub" / "fig.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert not plt.fignum_exists(fig.number)          # savefig closes the figure

        fig, ax = plt.subplots(figsize=(2, 1.5))
        ax.imshow([[0, 1], [2, 3]], cmap="div_blue_red")
        pk.savefig(fig, tmp_path / "fig.pdf")
        assert (tmp_path / "fig.pdf").read_bytes()[:5] == b"%PDF-"


def test_paper_builds_pdf(tmp_path):
    pytest.importorskip("reportlab")
    with plt.rc_context():
        pk.setup_mpl()
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot([0, 1, 2], [1, 0, 1], label="u")
        ax.legend()
        png = pk.savefig(fig, tmp_path / "fig.png")

    paper = pk.Paper("A test report with u₀ and 10⁻³", subtitle="Unit test", date="January 1, 2026",
                     report_id="TR-0")
    paper.abstract("Short abstract.", keywords=["PINN", "test"])
    paper.box("What this shows", ["one", "two"])
    paper.section("Method")
    paper.subsection("Setup")
    paper.p("A paragraph with a<super>2</super> and k₀.")
    paper.bullets(["first", "second"])
    assert paper.eq("u<sub>t</sub> = ν u<sub>xx</sub>") == 1
    paper.code("for i in range(3):\n    print(i < 2)")
    assert paper.next_fig() == 1
    assert paper.figure(png, "A line.", width_cm=8) == 1
    assert paper.table([["Method", "rel. L2"], ["FEM", pk.fmt(1.2e-4)], ["PINN", pk.fmt(0.031)]],
                       "Errors.", col_widths_cm=[4, 3], highlight_rows=[1]) == 1
    assert paper.next_tab() == 2
    paper.references(["A. Author, A paper, 2020."])

    out = paper.build(str(tmp_path / "out" / "report.pdf"))
    data = (tmp_path / "out" / "report.pdf").read_bytes()
    assert out.endswith("report.pdf") and data[:5] == b"%PDF-" and len(data) > 2000
    assert re.findall(rb"/Count (\d+)", data) == [b"1"]           # everything fits on one A4 page


def test_missing_reportlab_names_the_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("reportlab"):
            raise ImportError("No module named 'reportlab'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(pk, "_RL", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"pinneapple\[veriphysics\]"):
        pk.Paper("x")
