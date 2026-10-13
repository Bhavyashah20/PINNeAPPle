"""Figure style and PDF paper builder for PINNeAPPle technical reports.

Figures: one fixed categorical colour order (never cycled past its 8 entries in a single plot), a
single-hue sequential blue ramp (``SEQ``, registered as ``"seq_blue"``), a blue/red diverging ramp with
a neutral grey midpoint (``DIV``, registered as ``"div_blue_red"``), recessive axes and no top/right
spines. Call :func:`setup_mpl` once, then save each figure with :func:`savefig`.

PDF: :class:`Paper` builds an A4 report with reportlab/platypus in the STIX serif shipped with
matplotlib (full Greek and math glyphs): title block, abstract, numbered sections, equations,
figures, tables with captions, references and page numbers. reportlab is optional and is imported
only when a :class:`Paper` (or :func:`para`) is created::

    pip install "pinneapple[veriphysics]"

Example::

    from pinneapple_tools.reporting.paperkit import CAT, Paper, fmt, savefig, setup_mpl

    setup_mpl()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(x, y, color=CAT[0])
    png = savefig(fig, "out/fig1.png")

    paper = Paper("Heat conduction in a slab", subtitle="PINN vs. FEM")
    paper.abstract("We compare ...", keywords=["PINN", "FEM"])
    paper.section("Results")
    paper.figure(png, "Temperature along the slab.")
    paper.table([["Method", "rel. L2"], ["FEM", fmt(1.2e-4)]], "Errors against the exact solution.")
    paper.build("out/report.pdf")

Matplotlib's backend is left alone; scripts that run headless and want to be explicit can call
``matplotlib.use("Agg")`` before importing pyplot.
"""
from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "CAT", "INK", "INK2", "MUTED", "GRID", "SEQ", "DIV",
    "setup_mpl", "savefig", "sanitize", "fmt", "para", "Paper",
]

# ----------------------------------------------------------------------------- palette
#: Categorical colours, in the fixed order series are assigned (blue, orange, green, amber, pink,
#: dark green, violet, red).
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
#: Text and axis greys: primary ink, secondary ink, muted (axes, footer) and grid lines.
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
#: Single-hue sequential blue colormap (light to dark), registered as ``"seq_blue"`` by :func:`setup_mpl`.
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
#: Blue/red diverging colormap with a neutral grey midpoint, registered as ``"div_blue_red"``.
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f3a8a7", "#e34948", "#9c1f1f"])

_MPL_FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf"

_REPORTLAB_HINT = ('the PDF builder needs reportlab: pip install "pinneapple[veriphysics]" '
                   "(or pip install reportlab)")


def setup_mpl() -> None:
    """Apply the report style to ``matplotlib.rcParams`` and register ``SEQ`` / ``DIV``.

    STIX serif text and math, 9 pt base size, muted axes without top/right spines, light grid,
    ``CAT`` as the colour cycle, ``seq_blue`` as the default image colormap, 220 dpi tight saves.
    Safe to call more than once.
    """
    for cmap in (SEQ, DIV):
        if cmap.name not in matplotlib.colormaps:
            matplotlib.colormaps.register(cmap)
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlesize": 9.5, "axes.titlecolor": INK,
        "axes.titleweight": "bold", "axes.linewidth": 0.7, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 4.5,
        "axes.prop_cycle": matplotlib.cycler(color=CAT), "figure.dpi": 150, "savefig.dpi": 220,
        "savefig.bbox": "tight", "figure.facecolor": "white", "image.cmap": "seq_blue",
    })


def savefig(fig, path, **kwargs) -> str:
    """Save ``fig`` to ``path`` (creating parent folders), close it and return the path as a string.

    The format follows the file extension (``.png``, ``.pdf``, ``.svg``, ...); ``kwargs`` go to
    :meth:`matplotlib.figure.Figure.savefig`.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **kwargs)
    plt.close(fig)
    return str(path)


def fmt(x: Any, nd: int = 3) -> str:
    """Compact number for a table cell.

    ``None`` gives an em dash, NaN gives ``"n/a"``, ``|x| >= 1e4`` or ``0 < |x| < 1e-3`` use
    scientific notation with 2 decimals, anything else fixed-point with ``nd`` decimals. Values that
    are not numbers are returned as ``str(x)``.
    """
    if x is None:
        return "—"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if x != x:
        return "n/a"
    if abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


# ----------------------------------------------------------------------------- PDF
_SUP = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ", "0123456789+-=()ni", strict=True))
_SUB = dict(zip("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₚₜ", "0123456789+-=()aeoxpt", strict=True))


def sanitize(text: str) -> str:
    """Turn runs of Unicode super/subscript characters into reportlab ``<super>``/``<sub>`` markup.

    Most PDF fonts lack those glyphs, so ``"10⁻³"`` becomes ``"10<super>-3</super>"`` and
    ``"u₀"`` becomes ``"u<sub>0</sub>"``. Other characters (and existing markup) pass through.
    """
    out, i = [], 0
    while i < len(text):
        for table, tag in ((_SUP, "super"), (_SUB, "sub")):
            if text[i] in table:
                j = i
                while j < len(text) and text[j] in table:
                    j += 1
                out.append(f"<{tag}>" + "".join(table[c] for c in text[i:j]) + f"</{tag}>")
                i = j
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


_RL: SimpleNamespace | None = None


def _reportlab() -> SimpleNamespace:
    """Import reportlab on first use, register the STIX fonts and return the names the builder needs."""
    global _RL
    if _RL is not None:
        return _RL
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
        from reportlab.lib.fonts import addMapping
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (
            Image,
            KeepTogether,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc

    if "STIX" not in pdfmetrics.getRegisteredFontNames():
        for name, ttf in (("STIX", "STIXGeneral.ttf"), ("STIX-B", "STIXGeneralBol.ttf"),
                          ("STIX-I", "STIXGeneralItalic.ttf"), ("STIX-BI", "STIXGeneralBolIta.ttf"),
                          ("Mono", "DejaVuSansMono.ttf")):
            pdfmetrics.registerFont(TTFont(name, str(_MPL_FONT / ttf)))
        addMapping("STIX", 0, 0, "STIX")
        addMapping("STIX", 1, 0, "STIX-B")
        addMapping("STIX", 0, 1, "STIX-I")
        addMapping("STIX", 1, 1, "STIX-BI")

    _RL = SimpleNamespace(colors=colors, TA_CENTER=TA_CENTER, TA_JUSTIFY=TA_JUSTIFY, A4=A4,
                          ParagraphStyle=ParagraphStyle, cm=cm, ImageReader=ImageReader, Image=Image,
                          KeepTogether=KeepTogether, PageBreak=PageBreak, Paragraph=Paragraph,
                          SimpleDocTemplate=SimpleDocTemplate, Spacer=Spacer, Table=Table, TableStyle=TableStyle)
    return _RL


def para(text: Any, style, **kw):
    """A reportlab ``Paragraph`` of ``str(text)`` after :func:`sanitize` (needs reportlab)."""
    return _reportlab().Paragraph(sanitize(str(text)), style, **kw)


class Paper:
    """A4 technical report built with reportlab/platypus (needs reportlab).

    Append content in reading order with :meth:`abstract`, :meth:`box`, :meth:`section`,
    :meth:`subsection`, :meth:`p`, :meth:`bullets`, :meth:`eq`, :meth:`code`, :meth:`figure`,
    :meth:`table`, :meth:`references` and :meth:`pagebreak`, then write the file with :meth:`build`.
    Sections, equations, figures and tables are numbered automatically; text accepts reportlab
    paragraph markup (``<b>``, ``<i>``, ``<super>``, ...) and Unicode super/subscripts.

    Args:
        title: report title (also the running header from page 2 and the PDF title).
        subtitle: optional line under the title (PDF subject).
        authors: author line (PDF author).
        affiliation: line under the authors.
        date: date string; today's date (``"October 13, 2026"`` style) when ``None``.
        report_id: optional identifier appended to the date line.
        footer: text at the bottom left of every page.

    Attributes:
        story: the list of platypus flowables; append your own for anything not covered above.
        S: the paragraph styles by name (``"p"``, ``"h1"``, ``"cap"``, ``"cell"``, ``"code"``, ...).
    """

    def __init__(self, title: str, subtitle: str = "", authors: str = "PINNeAPPle Labs",
                 affiliation: str = "PINNeAPPle Labs — Physics AI research notes",
                 date: str | None = None, report_id: str = "",
                 footer: str = "PINNeAPPle Labs · technical report"):
        rl = self._rl = _reportlab()
        self.title, self.subtitle, self.authors, self.affiliation = title, subtitle, authors, affiliation
        self.date = date or _dt.date.today().strftime("%B %d, %Y")
        self.report_id = report_id
        self.footer = footer
        self.story: list = []
        self.n_sec = self.n_sub = self.n_fig = self.n_tab = self.n_eq = 0
        hx = rl.colors.HexColor
        PS = rl.ParagraphStyle
        base = dict(fontName="STIX", textColor=hx(INK))
        self.S = {
            "title": PS("t", fontSize=17.5, leading=21.5, alignment=rl.TA_CENTER, spaceAfter=4, fontName="STIX-B",
                        textColor=hx(INK)),
            "subtitle": PS("st", fontSize=11, leading=14, alignment=rl.TA_CENTER, spaceAfter=6, fontName="STIX-I",
                           textColor=hx(INK2)),
            "meta": PS("m", fontSize=9.5, leading=12, alignment=rl.TA_CENTER, textColor=hx(INK2), fontName="STIX"),
            "abs_h": PS("ah", fontSize=10, leading=13, fontName="STIX-B", spaceBefore=10, spaceAfter=3,
                        textColor=hx(INK)),
            "abs": PS("a", fontSize=9.6, leading=12.6, alignment=rl.TA_JUSTIFY, leftIndent=18, rightIndent=18,
                      **base),
            "h1": PS("h1", keepWithNext=1, fontSize=12.5, leading=15, fontName="STIX-B", spaceBefore=12,
                     spaceAfter=5, textColor=hx(INK)),
            "h2": PS("h2", keepWithNext=1, fontSize=10.8, leading=13.5, fontName="STIX-B", spaceBefore=8,
                     spaceAfter=3, textColor=hx(INK)),
            "p": PS("p", fontSize=10.2, leading=13.6, alignment=rl.TA_JUSTIFY, spaceAfter=5, **base),
            "bullet": PS("b", fontSize=10.2, leading=13.4, leftIndent=14, bulletIndent=4, spaceAfter=2,
                         alignment=rl.TA_JUSTIFY, **base),
            "cap": PS("c", fontSize=8.9, leading=11.2, alignment=rl.TA_JUSTIFY, spaceBefore=3, spaceAfter=9,
                      textColor=hx(INK2), fontName="STIX"),
            "eq": PS("e", fontSize=10.4, leading=14, alignment=rl.TA_CENTER, spaceBefore=3, spaceAfter=5,
                     fontName="STIX-I", textColor=hx(INK)),
            "cell": PS("cell", fontSize=8.4, leading=10.2, fontName="STIX", textColor=hx(INK)),
            "cellb": PS("cellb", fontSize=8.4, leading=10.2, fontName="STIX-B", textColor=hx(INK)),
            "ref": PS("r", fontSize=8.9, leading=11.2, leftIndent=16, firstLineIndent=-16, spaceAfter=2.5, **base),
            "code": PS("code", fontSize=7.8, leading=9.6, fontName="Mono", leftIndent=10, textColor=hx(INK2),
                       spaceAfter=4),
            "box": PS("box", fontSize=9.6, leading=12.6, alignment=rl.TA_JUSTIFY, **base),
        }
        self._front()

    # -- front matter
    def _front(self) -> None:
        rl = self._rl
        self.story += [rl.Spacer(1, 0.3 * rl.cm), para(self.title, self.S["title"])]
        if self.subtitle:
            self.story.append(para(self.subtitle, self.S["subtitle"]))
        self.story += [para(self.authors, self.S["meta"]), para(self.affiliation, self.S["meta"]),
                       para(self.date + (f" · {self.report_id}" if self.report_id else ""), self.S["meta"]),
                       rl.Spacer(1, 0.2 * rl.cm)]

    def abstract(self, text: str, keywords: Sequence[str] = ()) -> None:
        """Add the abstract, with an optional ``Keywords:`` line."""
        Spacer = self._rl.Spacer
        self.story.append(para("Abstract", self.S["abs_h"]))
        self.story.append(para(text, self.S["abs"]))
        if keywords:
            self.story.append(Spacer(1, 3))
            self.story.append(para("<b>Keywords:</b> " + "; ".join(keywords), self.S["abs"]))
        self.story.append(Spacer(1, 6))

    def box(self, title: str, lines: Sequence[str]) -> None:
        """Add a shaded summary box: a bold title and one bullet per line (e.g. "What this study shows")."""
        rl = self._rl
        rows = [[para(f"<b>{title}</b>", self.S["box"])]] + [[para("• " + s, self.S["box"])] for s in lines]
        t = rl.Table(rows, colWidths=[16.2 * rl.cm])
        t.setStyle(rl.TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), rl.colors.HexColor("#f4f3f0")),
            ("BOX", (0, 0), (-1, -1), 0.6, rl.colors.HexColor("#c9c7c0")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
        self.story += [rl.Spacer(1, 4), t, rl.Spacer(1, 8)]

    # -- body
    def section(self, title: str) -> None:
        """Start numbered section ``n`` (resets the subsection counter)."""
        self.n_sec += 1
        self.n_sub = 0
        self.story.append(para(f"{self.n_sec}&nbsp;&nbsp;{title}", self.S["h1"]))

    def subsection(self, title: str) -> None:
        """Start subsection ``n.m`` of the current section."""
        self.n_sub += 1
        self.story.append(para(f"{self.n_sec}.{self.n_sub}&nbsp;&nbsp;{title}", self.S["h2"]))

    def p(self, *texts: str) -> None:
        """Add one justified paragraph per argument."""
        for t in texts:
            self.story.append(para(t, self.S["p"]))

    def bullets(self, items: Sequence[str]) -> None:
        """Add a bulleted list."""
        for it in items:
            self.story.append(para(it, self.S["bullet"], bulletText="•"))
        self.story.append(self._rl.Spacer(1, 3))

    def eq(self, text: str) -> int:
        """Add a centred display equation (reportlab markup) with its number on the right; return the number."""
        rl = self._rl
        self.n_eq += 1
        t = rl.Table([[para(text, self.S["eq"]), para(f"({self.n_eq})", self.S["p"])]],
                     colWidths=[14.6 * rl.cm, 1.6 * rl.cm])
        t.setStyle(rl.TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        self.story.append(t)
        return self.n_eq

    def code(self, text: str) -> None:
        """Add a monospaced block; spaces and ``<`` are kept literally."""
        for line in text.strip("\n").split("\n"):
            self.story.append(para(line.replace(" ", "&nbsp;").replace("<", "&lt;") or "&nbsp;", self.S["code"]))

    def figure(self, path: str, caption: str, width_cm: float = 15.5) -> int:
        """Add an image file with a numbered caption; return the figure number.

        The image is scaled to ``width_cm`` wide at its aspect ratio, capped at 19 cm tall.
        """
        rl = self._rl
        self.n_fig += 1
        iw, ih = rl.ImageReader(path).getSize()
        w = width_cm * rl.cm
        h = w * ih / iw
        if h > 19 * rl.cm:
            h = 19 * rl.cm
            w = h * iw / ih
        self.story.append(rl.KeepTogether([rl.Image(path, width=w, height=h),
                                           para(f"<b>Figure {self.n_fig}.</b> {caption}", self.S["cap"])]))
        return self.n_fig

    def next_fig(self) -> int:
        """Number the next :meth:`figure` call will get (for forward references in the text)."""
        return self.n_fig + 1

    def next_tab(self) -> int:
        """Number the next :meth:`table` call will get."""
        return self.n_tab + 1

    def table(self, rows: Sequence[Sequence], caption: str, col_widths_cm: Sequence[float] | None = None,
              header_rows: int = 1, highlight_rows: Sequence[int] = ()) -> int:
        """Add a booktabs-style table with its caption above; return the table number.

        String cells become paragraphs (bold in the first ``header_rows`` rows); other cells (e.g.
        flowables) are used as they are. ``highlight_rows`` are row indices shaded light blue.
        """
        rl = self._rl
        hx = rl.colors.HexColor
        self.n_tab += 1
        data = [[c if not isinstance(c, str) else para(c, self.S["cellb"] if r < header_rows else self.S["cell"])
                 for c in row] for r, row in enumerate(rows)]
        cw = [w * rl.cm for w in col_widths_cm] if col_widths_cm else None
        t = rl.Table(data, colWidths=cw, repeatRows=header_rows, hAlign="CENTER")
        st = [("LINEABOVE", (0, 0), (-1, 0), 0.9, hx(INK)),
              ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.6, hx(INK)),
              ("LINEBELOW", (0, -1), (-1, -1), 0.9, hx(INK)),
              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
              ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5)]
        for r in highlight_rows:
            st.append(("BACKGROUND", (0, r), (-1, r), hx("#eef4fc")))
        t.setStyle(rl.TableStyle(st))
        self.story.append(rl.KeepTogether([para(f"<b>Table {self.n_tab}.</b> {caption}", self.S["cap"]), t,
                                           rl.Spacer(1, 8)]))
        return self.n_tab

    def references(self, refs: Sequence[str]) -> None:
        """Add an unnumbered "References" heading and the list as ``[1]``, ``[2]``, ..."""
        self.story.append(para("References", self.S["h1"]))
        for i, r in enumerate(refs, 1):
            self.story.append(para(f"[{i}]&nbsp;&nbsp;{r}", self.S["ref"]))

    def pagebreak(self) -> None:
        """Start a new page."""
        self.story.append(self._rl.PageBreak())

    def build(self, path: str) -> str:
        """Write the PDF to ``path`` (creating parent folders) and return ``path``.

        Every page carries the footer and page number; pages after the first carry the title as a
        running header (truncated to 96 characters).
        """
        rl = self._rl
        cm, A4 = rl.cm, rl.A4
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        title, footer = self.title, self.footer

        def deco(canvas, doc):
            canvas.saveState()
            canvas.setFont("STIX", 8)
            canvas.setFillColor(rl.colors.HexColor(MUTED))
            if doc.page > 1:
                canvas.drawString(2.2 * cm, A4[1] - 1.3 * cm, (title[:95] + "…") if len(title) > 96 else title)
            canvas.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, f"{doc.page}")
            canvas.drawString(2.2 * cm, 1.2 * cm, footer)
            canvas.restoreState()

        doc = rl.SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                                   topMargin=2.0 * cm, bottomMargin=2.0 * cm, title=self.title,
                                   author=self.authors, subject=self.subtitle)
        doc.build(self.story, onFirstPage=deco, onLaterPages=deco)
        return path
