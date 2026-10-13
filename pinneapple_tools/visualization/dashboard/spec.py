"""What an animation-centred dashboard shows: an animation (frames), charts synced to its time cursor, KPI tiles, a
parameter panel and a colour bar. Pure data (numpy arrays and strings); the exporters are in :mod:`.html` and
:mod:`.static`.

Charts come in two kinds:

* ``"series"`` -- values along the frames (``x`` has one value per frame: time, distance ...). The cursor is a
  vertical line at the current frame; scrubbing the chart moves the animation.
* ``"phase"`` -- fixed curves (e.g. a testbed curve of drawbar pull versus slip) plus markers that move with the
  frames (e.g. each wheel's current slip and pull).

Units belong in every label: ``Chart(ylabel="Drawbar pull [N]")``, ``Tile(unit="m")``, ``Param(unit="kg")``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# categorical slots 1-8 of the reference palette, dark-surface steps (validated on #161b22)
PALETTE_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767")
THEME = {"bg": "#0e1217", "panel": "#161b22", "panel2": "#1c232c", "grid": "#2a323d", "axis": "#5b6673",
         "text": "#e8e8e8", "text2": "#aab3bf", "muted": "#7d8794", "accent": "#e8e8e8", "cursor": "#f0efec"}


@dataclass
class Series:
    """A line. ``y`` has one value per frame (series charts) or per ``x`` point (phase-chart curves)."""
    label: str
    y: np.ndarray
    color: str | None = None
    dash: bool = False
    x: np.ndarray | None = None        # phase charts: the curve's own x
    width: float = 1.8

    def __post_init__(self):
        self.y = np.asarray(self.y, float)
        if self.x is not None:
            self.x = np.asarray(self.x, float)


@dataclass
class Marker:
    """A point that moves with the frames (phase charts): ``x[k], y[k]`` at frame k."""
    label: str
    x: np.ndarray
    y: np.ndarray
    color: str | None = None

    def __post_init__(self):
        self.x, self.y = np.asarray(self.x, float), np.asarray(self.y, float)


@dataclass
class Chart:
    title: str
    ylabel: str
    xlabel: str
    series: list[Series] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    x: np.ndarray | None = None        # series charts: one x per frame
    kind: str = "series"
    slot: str = "right"                # "left", "right" or "bottom"
    ylim: tuple | None = None
    note: str = ""

    def __post_init__(self):
        if self.kind not in ("series", "phase"):
            raise ValueError("Chart.kind must be 'series' or 'phase'")
        if self.x is not None:
            self.x = np.asarray(self.x, float)


@dataclass
class Tile:
    """A KPI: a scalar, or one value per frame (the tile follows the cursor)."""
    label: str
    value: object
    unit: str = ""
    fmt: str = "{:.2f}"
    note: str = ""

    def at(self, k: int) -> float:
        v = np.asarray(self.value, float)
        return float(v) if v.ndim == 0 else float(v[min(k, len(v) - 1)])

    def text(self, k: int) -> str:
        return self.fmt.format(self.at(k))


@dataclass
class Param:
    label: str
    value: object
    unit: str = ""

    def text(self) -> str:
        v = self.value
        if isinstance(v, float):
            v = f"{v:.4g}"
        return f"{v} {self.unit}".strip()


@dataclass
class ColorBar:
    """Colour bar beside the animation; ``saturation`` < 1 shows the colour map muted the way a semi-transparent,
    desaturated field overlay draws it (``studio.planetary.SurfaceView.field_saturation``)."""
    title: str
    cmap: str
    lo: float
    hi: float
    saturation: float = 1.0

    def colors(self, n: int = 256) -> np.ndarray:
        """(n, 3) RGB in [0, 1] from ``lo`` to ``hi``."""
        import matplotlib
        c = matplotlib.colormaps[self.cmap](np.linspace(0, 1, n))[:, :3]
        if self.saturation != 1.0:
            lum = (c @ np.array([0.299, 0.587, 0.114]))[:, None]
            c = np.clip(lum + self.saturation * (c - lum), 0.0, 1.0)
        return c


@dataclass
class Dashboard:
    """``frames``: a list of RGB arrays (H, W, 3) or image paths, one per entry of ``t`` [s]."""
    title: str
    t: np.ndarray
    frames: list
    charts: list[Chart] = field(default_factory=list)
    tiles: list[Tile] = field(default_factory=list)
    params: list[Param] = field(default_factory=list)
    subtitle: str = ""
    colorbar: ColorBar | None = None
    footer: str = ""
    time_label: str = "t"
    time_unit: str = "s"
    scene3d: dict | None = None        # optional live three.js view, see html.scene3d_payload

    def __post_init__(self):
        self.t = np.asarray(self.t, float)
        if len(self.frames) != len(self.t):
            raise ValueError(f"{len(self.frames)} frames for {len(self.t)} times")
        for c in self.charts:
            if c.kind == "series":
                n = len(c.x) if c.x is not None else len(self.t)
                for s in c.series:
                    if len(s.y) != n:
                        raise ValueError(f"chart {c.title!r}: series {s.label!r} has {len(s.y)} values, not {n}")
            for m in c.markers:
                if len(m.x) != len(self.t):
                    raise ValueError(f"chart {c.title!r}: marker {m.label!r} needs one point per frame")
        self._colour()

    def _colour(self):
        """Fixed categorical order per chart (colour follows the entity: same label, same colour everywhere)."""
        seen: dict[str, str] = {}
        for c in self.charts:
            for item in [*c.series, *c.markers]:
                if item.color is None:
                    key = item.label.split(" ")[0]
                    if key not in seen:
                        seen[key] = PALETTE_DARK[len(seen) % len(PALETTE_DARK)]
                    item.color = seen[key]

    def frame(self, k: int) -> np.ndarray:
        f = self.frames[k]
        if isinstance(f, str):
            from PIL import Image
            return np.asarray(Image.open(f).convert("RGB"))
        return np.asarray(f)[..., :3]

    def charts_in(self, slot: str) -> list[Chart]:
        return [c for c in self.charts if c.slot == slot]

    def to_html(self, path: str, **kw) -> str:
        from .html import write_html
        return write_html(self, path, **kw)

    def to_png(self, path: str, k: int | None = None, **kw) -> str:
        from .static import write_png
        return write_png(self, path, k, **kw)

    def to_video(self, gif: str | None = None, mp4: str | None = None, **kw) -> dict:
        from .static import write_video
        return write_video(self, gif, mp4, **kw)


__all__ = ["Dashboard", "Chart", "Series", "Marker", "Tile", "Param", "ColorBar", "PALETTE_DARK", "THEME"]
