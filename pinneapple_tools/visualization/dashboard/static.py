"""PNG and GIF/MP4 exports of a :class:`~.spec.Dashboard`, drawn with matplotlib in the studio's dark style.

Layout (1920 x 1080 by default)::

    title, subtitle                                  KPI tiles
    left charts  |  animation (+ colour bar, time cursor)  |  right charts
    bottom charts ...                                       |  parameters

The figure is drawn once; for every frame only the image, the cursors, the moving markers and the tiles change, so
a video costs a fraction of a second per frame.
"""
from __future__ import annotations

import shutil

import numpy as np

from .spec import THEME, Chart, Dashboard


def _style(ax, fs: float = 1.0):
    ax.set_facecolor(THEME["panel"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(THEME["axis"])
    ax.tick_params(colors=THEME["text2"], labelsize=7.5 * fs, length=2.5)
    ax.grid(True, color=THEME["grid"], lw=0.6)
    ax.set_axisbelow(True)


class _Layout:
    def __init__(self, db: Dashboard, size=(1920, 1080), dpi: int = 100, font_scale: float = 1.0):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.db = db
        self.fs = float(font_scale)
        W, H = size
        # text size relative to the figure, against the 1920 x 1080 / 100 dpi reference layout: gaps that hold
        # text (titles, axis labels, legends) grow with it, so a small or large-font layout does not overlap
        self.kh = self.fs * dpi / 100.0 * 1080.0 / H
        self.kw = self.fs * dpi / 100.0 * 1920.0 / W
        self.fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=THEME["bg"])
        self.dyn = []                                   # (update(k) callables)
        self._header()
        self._centre()
        self._columns()
        self._bottom()

    # ------------------------------------------------------------------ pieces
    def _header(self):
        f, db = self.fig, self.db
        n = len(db.tiles)
        x_tiles = 0.47 if n else 0.988
        f.text(0.012, 0.962, db.title, color=THEME["text"], fontsize=self.fs * 17, fontweight="bold", va="center")
        if db.subtitle:
            import textwrap
            W = f.get_size_inches()[0] * f.dpi
            chars = max(20, int((x_tiles - 0.025) * W / (0.52 * self.fs * 9.5 * f.dpi / 72)))
            lines = textwrap.wrap(db.subtitle, chars)[:2]
            f.text(0.012, 0.928 - 0.012 * self.kh * (len(lines) - 1), "\n".join(lines), color=THEME["text2"],
                   fontsize=self.fs * (9.5 if len(lines) == 1 else 8.5), va="center", linespacing=1.15)
        if not n:
            return
        x0, x1, w_gap = 0.47, 0.988, 0.006
        w = (x1 - x0 - (n - 1) * w_gap) / n
        for i, tl in enumerate(db.tiles):
            ax = f.add_axes([x0 + i * (w + w_gap), 0.905, w, 0.08])
            ax.set_facecolor(THEME["panel"])
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(THEME["grid"])
            ax.text(0.06, 0.78, tl.label, color=THEME["text2"], fontsize=self.fs * 8.5, transform=ax.transAxes,
                    va="center")
            val = ax.text(0.06, 0.36, "", color=THEME["text"], fontsize=self.fs * 17, fontweight="bold", transform=ax.transAxes,
                          va="center")
            ax.text(0.94, 0.36, tl.unit, color=THEME["muted"], fontsize=self.fs * 9, transform=ax.transAxes, va="center",
                    ha="right")
            self.dyn.append(lambda k, v=val, tl=tl: v.set_text(tl.text(k)))

    def _centre(self):
        f, db = self.fig, self.db
        img0 = db.frame(0)
        h, w = img0.shape[:2]
        box = [0.255, 0.335, 0.49, 0.555]               # left, bottom, width, height (figure fractions)
        W, H = f.get_size_inches()
        aspect_fig = box[2] * W / (box[3] * H)
        if w / h > aspect_fig:                          # fit the frame in the box, keep its aspect ratio
            hh = box[2] * W / (w / h) / H
            box = [box[0], box[1] + (box[3] - hh) / 2, box[2], hh]
        else:
            ww = box[3] * H * (w / h) / W
            box = [box[0] + (box[2] - ww) / 2, box[1], ww, box[3]]
        ax = f.add_axes(box)
        ax.set_axis_off()
        im = ax.imshow(img0, interpolation="bilinear")
        self.dyn.append(lambda k, im=im: im.set_data(db.frame(k)))
        cb = db.colorbar
        if cb is not None:
            cax = f.add_axes([box[0] + box[2] - 0.012, box[1] + 0.06, 0.008, box[3] * 0.5])
            cax.imshow(cb.colors(256)[:, None, :], aspect="auto", origin="lower", extent=[0, 1, cb.lo, cb.hi])
            cax.set_xticks([])
            cax.yaxis.tick_left()
            cax.tick_params(colors=THEME["text"], labelsize=7.5 * self.fs)
            for s in cax.spines.values():
                s.set_color(THEME["axis"])
            f.text(box[0] + box[2] - 0.008, box[1] + 0.06 + box[3] * 0.5 + 0.012, cb.title, color=THEME["text"],
                   fontsize=self.fs * 8.5, ha="right")
        # time cursor bar under the animation
        tax = f.add_axes([box[0], box[1] - 0.03, box[2], 0.012])
        tax.set_xlim(db.t[0], db.t[-1])
        tax.set_ylim(0, 1)
        tax.set_axis_off()
        tax.add_patch(__import__("matplotlib").patches.Rectangle((db.t[0], 0.35), db.t[-1] - db.t[0], 0.3,
                                                                 color=THEME["grid"]))
        prog = tax.barh(0.5, 0.0, left=db.t[0], height=0.3, color=THEME["accent"])[0]
        dot, = tax.plot([db.t[0]], [0.5], "o", color=THEME["cursor"], ms=7)
        lab = f.text(box[0], box[1] - 0.052, "", color=THEME["text2"], fontsize=self.fs * 9)

        def upd(k, prog=prog, dot=dot, lab=lab):
            prog.set_width(db.t[k] - db.t[0])
            dot.set_data([db.t[k]], [0.5])
            lab.set_text(f"{db.time_label} = {db.t[k]:6.2f} {db.time_unit}   ({k + 1}/{len(db.t)})")
        self.dyn.append(upd)

    def _chart(self, rect, c: Chart):
        ax = self.fig.add_axes(rect)
        _style(ax, self.fs)
        db = self.db
        ax.set_title(c.title, color=THEME["text"], fontsize=self.fs * 9, loc="left", pad=4)
        ax.set_xlabel(c.xlabel, color=THEME["text2"], fontsize=self.fs * 7.5, labelpad=1)
        ax.set_ylabel(c.ylabel, color=THEME["text2"], fontsize=self.fs * 7.5, labelpad=2)
        if c.kind == "series":
            x = c.x if c.x is not None else db.t
            for s in c.series:
                ax.plot(x, s.y, color=s.color, lw=s.width * 0.75, ls="--" if s.dash else "-", label=s.label)
            cur = ax.axvline(x[0], color=THEME["cursor"], lw=1.0, alpha=0.85)
            dots = [ax.plot([x[0]], [s.y[0]], "o", color=s.color, ms=4, mec=THEME["panel"], mew=1)[0]
                    for s in c.series]

            def upd(k, cur=cur, dots=dots, x=x, c=c):
                kk = min(k, len(x) - 1)
                cur.set_xdata([x[kk], x[kk]])
                for d, s in zip(dots, c.series, strict=True):
                    d.set_data([x[kk]], [s.y[kk]])
            self.dyn.append(upd)
        else:
            for s in c.series:
                ax.plot(s.x, s.y, color=s.color, lw=s.width * 0.75, ls="--" if s.dash else "-", label=s.label)
            ms = [ax.plot([m.x[0]], [m.y[0]], "o", color=m.color, ms=6 * self.fs, mec=THEME["text"], mew=0.8,
                          label=m.label)[0] for m in c.markers]
            xs = np.concatenate([s.x for s in c.series] + [m.x for m in c.markers])
            ys = np.concatenate([s.y for s in c.series] + [m.y for m in c.markers])
            xs, ys = xs[np.isfinite(xs)], ys[np.isfinite(ys)]
            if len(xs) and len(ys):                      # room for every position the markers take
                px, py = 0.04 * (np.ptp(xs) or 1.0), 0.06 * (np.ptp(ys) or 1.0)
                ax.set_xlim(xs.min() - px, xs.max() + px)
                ax.set_ylim(ys.min() - py, ys.max() + py)

            def upd(k, ms=ms, c=c):
                for p, m in zip(ms, c.markers, strict=True):
                    p.set_data([m.x[k]], [m.y[k]])
            self.dyn.append(upd)
        if c.ylim:
            ax.set_ylim(*c.ylim)
        n = len(c.series) + len(c.markers)
        if n >= 2:
            leg = ax.legend(fontsize=self.fs * 6.5, frameon=True, ncol=min(4, n), loc="upper left", handlelength=1.4,
                            columnspacing=0.8, borderaxespad=0.2, facecolor=THEME["panel"], edgecolor="none",
                            framealpha=0.85)                 # the curves pass under the legend, not through it
            for t in leg.get_texts():
                t.set_color(THEME["text2"])
        return ax

    def _columns(self):
        for slot, x0 in (("left", 0.035), ("right", 0.787)):
            cs = self.db.charts_in(slot)
            if not cs:
                continue
            top, bottom = 0.875, 0.335
            h = (top - bottom) / len(cs)
            dx = 0.012 * (self.kw - 1) if slot == "left" else 0.0
            for i, c in enumerate(cs):
                self._chart([x0 + dx, top - (i + 1) * h + 0.035 * self.kh, 0.19 - abs(dx) - 0.006 * (self.kw - 1),
                             h - 0.062 * self.kh], c)

    def _bottom(self):
        cs = self.db.charts_in("bottom")
        params = self.db.params
        x1 = 0.80 if params else 0.988
        if cs:
            w = (x1 - 0.035) / len(cs)
            x0 = 0.035 + 0.012 * (self.kw - 1)
            for i, c in enumerate(cs):
                self._chart([x0 + i * w, 0.06 * self.kh, w - 0.045 * self.kw, 0.255 - 0.06 * self.kh], c)
        if params:
            ax = self.fig.add_axes([0.815, 0.03, 0.173, 0.25])
            ax.set_facecolor(THEME["panel"])
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(THEME["grid"])
            ax.text(0.05, 0.95, "Parameters", color=THEME["text"], fontsize=self.fs * 9.5, fontweight="bold",
                    transform=ax.transAxes, va="top")
            n = len(params)
            top = 0.95 - 0.10 * self.kh
            step = min(0.075, top / max(n, 1))
            box_px = 0.25 * self.fig.get_size_inches()[1] * 72           # box height in points
            fsz = min(self.fs * 7.6, 0.92 * step * box_px)
            for i, p in enumerate(params):
                y = top - i * step
                ax.text(0.05, y, p.label, color=THEME["text2"], fontsize=fsz, transform=ax.transAxes, va="top")
                ax.text(0.95, y, p.text(), color=THEME["text"], fontsize=fsz, transform=ax.transAxes, va="top",
                        ha="right")
        if self.db.footer:
            self.fig.text(0.012, 0.008, self.db.footer, color=THEME["muted"], fontsize=self.fs * 7)

    # ------------------------------------------------------------------ output
    def draw(self, k: int) -> np.ndarray:
        for u in self.dyn:
            u(k)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        import matplotlib.pyplot as plt
        plt.close(self.fig)


def write_png(db: Dashboard, path: str, k: int | None = None, size=(1920, 1080), dpi: int = 100,
              font_scale: float = 1.0) -> str:
    """The dashboard at frame ``k`` (default: the last) as a PNG."""
    from PIL import Image
    L = _Layout(db, size, dpi, font_scale)
    img = L.draw(len(db.t) - 1 if k is None else k)
    L.close()
    Image.fromarray(img).save(path)
    return path


def write_gif(path: str, frames, fps: float, colors: int = 255, loop: int = 0) -> str:
    """Animated GIF of RGB ``frames`` with one shared adaptive palette (octree quantisation of a few frames spread
    over the clip, which keeps small areas of distinct colour -- chart lines, a colour bar -- that a median cut
    merges away) and frame differencing: after the first frame, pixels whose palette index did not change are written as
    one transparent index, so the static parts of a dashboard (and the still areas of the picture) cost almost
    nothing."""
    from PIL import Image
    frames = [np.asarray(f)[..., :3] for f in frames]
    colors = int(np.clip(colors, 2, 255))                 # one index stays free for transparency
    pick = frames[:: max(1, len(frames) // 6)][:6]
    mosaic = Image.fromarray(np.concatenate(pick, 0))
    pal = mosaic.quantize(colors=colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    palette = pal.getpalette()[: 3 * colors] + [0, 0, 0] * (256 - colors)
    clear = colors                                        # the transparent index
    ims, prev = [], None
    for f in frames:
        idx = np.asarray(Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE), np.uint8)
        out = idx if prev is None else np.where(idx == prev, np.uint8(clear), idx)
        prev = idx
        im = Image.fromarray(out, "P")
        im.putpalette(palette)
        ims.append(im)
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(round(1000 / fps)), loop=loop,
                optimize=False, disposal=1, transparency=clear)
    return path


def _gif_frames(src) -> list:
    from PIL import Image, ImageSequence
    if isinstance(src, (str, bytes)) or hasattr(src, "__fspath__"):
        with Image.open(src) as im:
            return [f.convert("RGB") for f in ImageSequence.Iterator(im)]
    return [f.convert("RGB") if isinstance(f, Image.Image) else Image.fromarray(np.asarray(f, np.uint8)[..., :3])
            for f in src]


def hstack_gifs(paths_or_frames, out, fps: float = 10, colors: int = 128, background: str = "white",
                mismatch: str = "error", loop: int = 0):
    """Several animations side by side in one GIF, frame by frame (e.g. two runs of the same case compared).

    ``paths_or_frames``: GIF paths, or sequences of RGB frames (arrays or PIL images), left to right. Each output
    frame is as tall as the tallest input; shorter inputs are pasted at the top and the rest is filled with
    ``background``. ``mismatch`` sets what happens when the inputs have different frame counts: ``"error"`` (default,
    raise ``ValueError``), ``"hold"`` (the shorter ones stay on their last frame until the longest ends) or
    ``"loop"`` (the shorter ones start again). Frames are quantised to ``colors`` colours each (median cut) and
    written at ``fps``. Returns ``out``."""
    from PIL import Image
    if mismatch not in ("error", "hold", "loop"):
        raise ValueError("mismatch must be 'error', 'hold' or 'loop'")
    clips = [_gif_frames(s) for s in paths_or_frames]
    if not clips or any(len(c) == 0 for c in clips):
        raise ValueError("give at least one non-empty animation")
    counts = [len(c) for c in clips]
    if len(set(counts)) > 1 and mismatch == "error":
        raise ValueError(f"the animations have different frame counts {counts}; use mismatch='hold' or 'loop'")
    frames = []
    for k in range(max(counts)):
        row = [c[k] if k < len(c) else (c[-1] if mismatch == "hold" else c[k % len(c)]) for c in clips]
        im = Image.new("RGB", (sum(f.width for f in row), max(f.height for f in row)), background)
        x = 0
        for f in row:
            im.paste(f, (x, 0))
            x += f.width
        frames.append(im.quantize(colors=int(colors), method=Image.Quantize.MEDIANCUT))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=loop, optimize=True)
    return out


def write_video(db: Dashboard, gif: str | None = None, mp4: str | None = None, *, fps: int = 12, size=(1920, 1080),
                gif_size=(1280, 720), gif_every: int = 1, dpi: int = 100, frames=None, progress=None,
                gif_font_scale: float | None = None, gif_colors: int = 255, mp4_quality: int = 8) -> dict:
    """GIF and/or MP4 (H.264, ``size``) of the dashboard over ``frames`` (default all). The GIF keeps every
    ``gif_every``-th frame at the same playback speed. With ``gif_font_scale`` the GIF is laid out natively at
    ``gif_size`` with the text scaled by that factor (readable on small screens and social feeds); otherwise its
    frames are the video frames downscaled to ``gif_size``."""
    import imageio.v2 as imageio
    from PIL import Image
    ks = list(range(len(db.t))) if frames is None else list(frames)
    L = _Layout(db, size, dpi)
    Lg = _Layout(db, gif_size, dpi, gif_font_scale) if gif and gif_font_scale else None
    out = {}
    writer = None
    if mp4 and shutil.which("ffmpeg"):
        writer = imageio.get_writer(mp4, fps=fps, codec="libx264", quality=mp4_quality, macro_block_size=8,
                                    pixelformat="yuv420p")
    gif_frames = []
    for j, k in enumerate(ks):
        img = L.draw(k) if (writer is not None or Lg is None) else None
        if writer is not None:
            writer.append_data(img)
        if gif and j % gif_every == 0:
            gif_frames.append(Lg.draw(k) if Lg is not None else
                              np.asarray(Image.fromarray(img).resize(gif_size, Image.LANCZOS)))
        if progress:
            progress(j + 1, len(ks))
    L.close()
    if Lg is not None:
        Lg.close()
    if writer is not None:
        writer.close()
        out["mp4"] = mp4
    if gif:
        write_gif(gif, gif_frames, fps / gif_every, colors=gif_colors)
        out["gif"] = gif
    out["frames"] = len(ks)
    return out


__all__ = ["write_png", "write_video", "write_gif", "hstack_gifs"]
