"""Interactive HTML export of a :class:`~.spec.Dashboard`: one self-contained file (no server, no internet).

* The animation is embedded as an H.264 MP4 (short keyframe interval, so seeking is exact) or, without ffmpeg, as
  JPEG frames. Charts are drawn on canvases; hover shows a crosshair and the values, and dragging on a time chart
  scrubs the animation. Play/pause, a time slider and the arrow keys step through the frames; the KPI tiles follow.
* The colour bar, the tick and number formatting come from the studio viewer's ``studio-core.js`` (inlined).
* Optional live 3D view (``scene3d``): the vehicle parts moving with the frames over the terrain, drawn with the
  studio's vendored three.js (inlined through an import map), lit with a :mod:`..studio.lighting` preset.
"""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np

from .spec import THEME, Dashboard

STUDIO_WEB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "studio", "web")


def _b64(a, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(np.asarray(a, dtype)).tobytes()).decode()


def _r(a, nd=5):
    a = np.asarray(a, float)
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a.ravel()]


def _cmap_stops(name: str, n: int = 9):
    import matplotlib
    cm = matplotlib.colormaps[name]
    return [[i / (n - 1), [int(255 * c) for c in cm(i / (n - 1))[:3]]] for i in range(n)]


def _media(db: Dashboard, fps: int, max_width: int, quality: int) -> dict:
    """The frames as an MP4 data URI (ffmpeg) or a list of JPEG data URIs."""
    from PIL import Image
    imgs = []
    for k in range(len(db.t)):
        im = Image.fromarray(db.frame(k))
        if im.width > max_width:
            im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
        imgs.append(im)
    w, h = imgs[0].size
    if shutil.which("ffmpeg"):
        d = tempfile.mkdtemp()
        for k, im in enumerate(imgs):
            im.save(os.path.join(d, f"f{k:05d}.png"))
        common = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(d, "f%05d.png"),
                  "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-g", "4"]
        sources = []
        webm, mp4 = os.path.join(d, "a.webm"), os.path.join(d, "a.mp4")
        try:                                             # VP9 plays in every browser engine, H.264 where licensed
            subprocess.run(common + ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(quality + 8), "-row-mt", "1",
                                     webm], check=True)
            sources.append(("video/webm", open(webm, "rb").read()))
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        try:
            subprocess.run(common + ["-c:v", "libx264", "-crf", str(quality), "-movflags", "+faststart", mp4],
                           check=True)
            sources.append(("video/mp4", open(mp4, "rb").read()))
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        shutil.rmtree(d, ignore_errors=True)
        if sources:
            return {"kind": "video", "src": [f"data:{t};base64," + base64.b64encode(b).decode() for t, b in sources],
                    "types": [t for t, _ in sources], "fps": fps, "w": w, "h": h}
    srcs = []
    for im in imgs:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        srcs.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
    return {"kind": "frames", "src": srcs, "fps": fps, "w": w, "h": h}


def payload(db: Dashboard) -> dict:
    """Everything the page draws, as JSON-able data."""
    charts = []
    for c in db.charts:
        x = c.x if c.x is not None else db.t
        item = {"title": c.title, "xlabel": c.xlabel, "ylabel": c.ylabel, "kind": c.kind, "slot": c.slot,
                "ylim": list(c.ylim) if c.ylim else None, "note": c.note,
                "series": [{"label": s.label, "color": s.color, "dash": s.dash, "width": s.width,
                            "x": _r(s.x) if s.x is not None else None, "y": _r(s.y)} for s in c.series],
                "markers": [{"label": m.label, "color": m.color, "x": _r(m.x), "y": _r(m.y)} for m in c.markers]}
        if c.kind == "series":
            item["x"] = _r(x)
        charts.append(item)
    tiles = []
    for tl in db.tiles:
        v = np.asarray(tl.value, float)
        tiles.append({"label": tl.label, "unit": tl.unit, "note": tl.note,
                      "text": [tl.text(k) for k in range(len(db.t))] if v.ndim else [tl.text(0)]})
    cb = db.colorbar
    return {"title": db.title, "subtitle": db.subtitle, "t": _r(db.t, 4), "time_label": db.time_label,
            "time_unit": db.time_unit, "charts": charts, "tiles": tiles,
            "params": [{"label": p.label, "text": p.text()} for p in db.params], "footer": db.footer,
            "colorbar": None if cb is None else {"title": cb.title, "lo": cb.lo, "hi": cb.hi,
                                                 "stops": _cmap_stops(cb.cmap)},
            "theme": THEME}


def scene3d_payload(bodies: dict, poses: dict, terrain=None, lighting=None, axes="z_up") -> dict:
    """Data for the live 3D view.

    ``bodies[name] = [(vertices in the body frame (V, 3), faces (F, 3), rgb (3,) in 0..1), ...]``;
    ``poses[name] = (positions (T, 3), quaternions (T, 4) w x y z)`` (world, z up); ``terrain = (X, Y, Z, rgb)``
    with rgb (ny, nx, 3) in 0..1; ``lighting``: a :func:`..studio.lighting.threejs_config` dict.
    """
    out = {"axes": axes, "bodies": [], "lighting": lighting}
    for name, parts in bodies.items():
        P, Q = poses[name]
        out["bodies"].append({
            "name": name, "pos": _b64(P, np.float32), "quat": _b64(Q, np.float32),
            "parts": [{"v": _b64(V, np.float32), "f": _b64(np.asarray(F), np.uint32),
                       "rgb": [round(float(c), 4) for c in rgb]} for V, F, rgb in parts]})
    if terrain is not None:
        X, Y, Z, rgb = terrain
        out["terrain"] = {"x0": float(X[0, 0]), "y0": float(Y[0, 0]), "dx": float(X[0, 1] - X[0, 0]),
                          "dy": float(Y[1, 0] - Y[0, 0]), "nx": int(X.shape[1]), "ny": int(X.shape[0]),
                          "z": _b64(Z, np.float32), "rgb": _b64(np.clip(rgb, 0, 1) * 255, np.uint8)}
    return out


def write_html(db: Dashboard, path: str, *, fps: int = 12, max_width: int = 1280, quality: int = 26) -> str:
    """Write the self-contained page; returns ``path``."""
    data = payload(db)
    media = _media(db, fps, max_width, quality)
    core = open(os.path.join(STUDIO_WEB, "studio-core.js"), encoding="utf-8").read()
    three = ""
    if db.scene3d is not None:
        three_src = open(os.path.join(STUDIO_WEB, "vendor", "three", "three.module.min.js"), encoding="utf-8").read()
        orbit = open(os.path.join(STUDIO_WEB, "vendor", "three", "OrbitControls.js"), encoding="utf-8").read()
        imap = {"imports": {"three": "data:text/javascript;base64," + base64.b64encode(three_src.encode()).decode(),
                            "three-orbit": "data:text/javascript;base64," + base64.b64encode(orbit.encode()).decode()}}
        three = f'<script type="importmap">{json.dumps(imap)}</script>'
    html = (TEMPLATE.replace("/*THEME*/", _css_vars())
            .replace("<!--IMPORTMAP-->", three)
            .replace("/*STUDIO_CORE*/", core)
            .replace("/*DATA*/", json.dumps(data, separators=(",", ":")))
            .replace("/*MEDIA*/", json.dumps(media, separators=(",", ":")))
            .replace("/*SCENE3D*/", json.dumps(db.scene3d, separators=(",", ":")) if db.scene3d else "null")
            .replace("/*VIEW3D*/", VIEW3D if db.scene3d else "")
            .replace("<!--TITLE-->", db.title))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def _css_vars() -> str:
    return "".join(f"--{k}:{v};" for k, v in THEME.items())


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title><!--TITLE--></title>
<!--IMPORTMAP-->
<style>
:root{/*THEME*/}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:13px/1.35 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{display:flex;gap:16px;align-items:center;padding:10px 16px 6px;flex-wrap:wrap}
header .t{flex:1 1 320px;min-width:0}
h1{font-size:19px;margin:0 0 2px;font-weight:650}
.sub{color:var(--text2);font-size:12px}
.tiles{display:flex;gap:6px;flex-wrap:wrap}
.tile{background:var(--panel);border:1px solid var(--grid);border-radius:6px;padding:6px 10px;min-width:118px}
.tile .l{color:var(--text2);font-size:11px}.tile .v{font-size:20px;font-weight:650;font-variant-numeric:tabular-nums}
.tile .u{color:var(--muted);font-size:11px;margin-left:4px}
main{display:grid;grid-template-columns:minmax(220px,1fr) minmax(320px,2.1fr) minmax(220px,1fr);gap:10px;padding:4px 16px}
.col{display:flex;flex-direction:column;gap:8px;min-width:0}
.card{background:var(--panel);border:1px solid var(--grid);border-radius:6px;padding:6px 8px 4px;position:relative;min-width:0}
.card h3{font-size:12px;margin:0 0 2px;font-weight:600}
.card canvas{width:100%;height:150px;display:block;cursor:crosshair}
.legend{display:flex;flex-wrap:wrap;gap:4px 10px;font-size:11px;color:var(--text2);margin:0 0 2px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:4px;vertical-align:middle}
.legend i.m{width:9px;height:9px;border-radius:50%}
.stage{background:#000;border:1px solid var(--grid);border-radius:6px;position:relative;overflow:hidden}
.stage video,.stage img,.stage canvas.v3{width:100%;display:block}
#cbar{position:absolute;right:10px;top:10px;background:rgba(14,18,23,.72);padding:6px 8px;border-radius:6px;font-size:11px}
.ctrl{display:flex;gap:8px;align-items:center;margin-top:6px;flex-wrap:wrap}
.ctrl button,.ctrl select{background:var(--panel2);color:var(--text);border:1px solid var(--grid);border-radius:5px;padding:4px 10px;font:inherit}
.ctrl input[type=range]{flex:1;min-width:120px}
#tlab{color:var(--text2);font-variant-numeric:tabular-nums;min-width:170px}
.bottom{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr)) minmax(240px,300px);gap:10px;padding:8px 16px 4px}
.params table{width:100%;border-collapse:collapse;font-size:12px}
.params td{padding:2px 0;border-bottom:1px solid var(--grid)}.params td:last-child{text-align:right;font-variant-numeric:tabular-nums}
.tip{position:fixed;pointer-events:none;background:rgba(10,13,17,.94);border:1px solid var(--grid);border-radius:5px;padding:5px 8px;font-size:11px;display:none;z-index:9;white-space:nowrap}
.tip b{font-weight:600}
footer{color:var(--muted);font-size:11px;padding:4px 16px 12px}
.tabs{position:absolute;left:10px;top:10px;display:flex;gap:4px}
.tabs button{background:rgba(14,18,23,.72);color:var(--text);border:1px solid var(--grid);border-radius:5px;padding:3px 9px;font:inherit;font-size:11px}
.tabs button.on{background:var(--text);color:var(--bg)}
@media (max-width:900px){main{grid-template-columns:1fr}.col.side{order:2}}
</style></head>
<body>
<header><div class="t"><h1 id="title"></h1><div class="sub" id="subtitle"></div></div><div class="tiles" id="tiles"></div></header>
<main>
 <div class="col side" id="left"></div>
 <div class="col">
  <div class="stage" id="stage"><div id="cbar"></div></div>
  <div class="ctrl"><button id="play">Play</button><input type="range" id="slider" min="0" value="0">
   <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select>
   <span id="tlab"></span></div>
 </div>
 <div class="col side" id="right"></div>
</main>
<section class="bottom" id="bottom"></section>
<footer id="footer"></footer>
<div class="tip" id="tip"></div>
<script type="module">
/*STUDIO_CORE*/
const DATA = /*DATA*/;
const MEDIA = /*MEDIA*/;
const SCENE3D = /*SCENE3D*/;
const $ = (s) => document.querySelector(s);
const N = DATA.t.length;
let K = 0, playing = false, last = 0, acc = 0;
$("#title").textContent = DATA.title; $("#subtitle").textContent = DATA.subtitle; $("#footer").textContent = DATA.footer;
document.title = DATA.title;

// ---------------------------------------------------------------- animation
const stage = $("#stage");
let media;
if (MEDIA.kind === "video") {
  media = document.createElement("video"); media.muted = true; media.playsInline = true; media.preload = "auto";
  MEDIA.src.forEach((src, i) => { const so = document.createElement("source"); so.src = src; so.type = MEDIA.types[i]; media.appendChild(so); });
  media.style.aspectRatio = `${MEDIA.w} / ${MEDIA.h}`;
} else { media = document.createElement("img"); media.src = MEDIA.src[0]; media.style.aspectRatio = `${MEDIA.w} / ${MEDIA.h}`; }
stage.prepend(media);
function showFrame(k) {
  if (MEDIA.kind === "video") { const tt = (k + 0.5) / MEDIA.fps; if (Math.abs(media.currentTime - tt) > 0.25 / MEDIA.fps) media.currentTime = tt; }
  else media.src = MEDIA.src[k];
}
if (DATA.colorbar) {
  const c = DATA.colorbar;
  $("#cbar").innerHTML = colorbarHTML({ title: c.title, lo: c.lo, hi: c.hi, stops: c.stops, width: 190 });
} else $("#cbar").remove();

// ---------------------------------------------------------------- tiles and parameters
$("#tiles").innerHTML = DATA.tiles.map((t, i) => `<div class="tile" title="${t.note}"><div class="l">${t.label}</div><div><span class="v" id="tile${i}"></span><span class="u">${t.unit}</span></div></div>`).join("");
const params = document.createElement("div"); params.className = "card params";
params.innerHTML = `<h3>Parameters</h3><table>${DATA.params.map((p) => `<tr><td>${p.label}</td><td>${p.text}</td></tr>`).join("")}</table>`;

// ---------------------------------------------------------------- charts
const tip = $("#tip");
const charts = DATA.charts.map((c) => makeChart(c));
$("#bottom").appendChild(params);
function niceTicks(lo, hi, n = 4) {                        // 1-2-5 steps inside [lo, hi]
  const raw = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
  const step = (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p, d = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  const out = []; for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9 * Math.abs(hi); v += step) out.push([v, Math.abs(v) < step * 1e-6 ? "0" : fmt(v, d)]);
  return out;
}
function extent(c) {
  let xs = [], ys = [];
  if (c.kind === "series") { xs = c.x; c.series.forEach((s) => ys.push(...s.y)); }
  else { c.series.forEach((s) => { xs.push(...s.x); ys.push(...s.y); }); c.markers.forEach((m) => { xs.push(...m.x); ys.push(...m.y); }); }
  const f = (a) => a.filter((v) => v !== null && Number.isFinite(v));
  xs = f(xs); ys = f(ys);
  let [x0, x1] = [Math.min(...xs), Math.max(...xs)], [y0, y1] = c.ylim || [Math.min(...ys), Math.max(...ys)];
  if (y1 === y0) { y0 -= 1; y1 += 1; } const py = c.ylim ? 0 : 0.06 * (y1 - y0);
  return [x0, x1 === x0 ? x0 + 1 : x1, y0 - py, y1 + py];
}
function makeChart(c) {
  const card = document.createElement("div"); card.className = "card";
  const items = [...c.series.map((s) => ({ ...s, m: false })), ...c.markers.map((m) => ({ ...m, m: true }))];
  const leg = items.length > 1 ? `<div class="legend">${items.map((s) => `<span><i class="${s.m ? "m" : ""}" style="background:${s.color}${s.dash ? ";background:repeating-linear-gradient(90deg," + s.color + " 0 4px,transparent 4px 7px)" : ""}"></i>${s.label}</span>`).join("")}</div>` : "";
  card.innerHTML = `<h3>${c.title}</h3>${leg}<canvas></canvas>`;
  (c.slot === "left" ? $("#left") : c.slot === "bottom" ? $("#bottom") : $("#right")).appendChild(card);
  const cv = card.querySelector("canvas"), ctx = cv.getContext("2d");
  const ch = { c, cv, ctx, ext: extent(c), pad: { l: 46, r: 8, t: 6, b: 30 }, base: null };
  const drag = (e) => { if (c.kind !== "series") return; const k = nearestK(ch, e); setFrame(k); };
  cv.addEventListener("mousemove", (e) => { hover(ch, e); if (e.buttons === 1) drag(e); });
  cv.addEventListener("mousedown", (e) => { pause(); drag(e); });
  cv.addEventListener("mouseleave", () => { tip.style.display = "none"; ch.hoverK = null; draw(ch); });
  return ch;
}
function geom(ch) {
  const r = ch.cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  if (ch.cv.width !== Math.round(r.width * dpr) || ch.cv.height !== Math.round(r.height * dpr)) {
    ch.cv.width = Math.round(r.width * dpr); ch.cv.height = Math.round(r.height * dpr); ch.base = null;
  }
  const [x0, x1, y0, y1] = ch.ext, p = ch.pad, W = r.width, H = r.height;
  return { dpr, W, H, sx: (x) => p.l + (x - x0) / (x1 - x0) * (W - p.l - p.r), sy: (y) => H - p.b - (y - y0) / (y1 - y0) * (H - p.t - p.b),
           ix: (px) => x0 + (px - p.l) / (W - p.l - p.r) * (x1 - x0) };
}
function drawBase(ch, g) {
  const { ctx, c } = ch, T = DATA.theme, [x0, x1, y0, y1] = ch.ext;
  ctx.setTransform(g.dpr, 0, 0, g.dpr, 0, 0); ctx.clearRect(0, 0, g.W, g.H);
  ctx.font = "10px system-ui"; ctx.lineWidth = 1;
  const yt = niceTicks(y0, y1), xt = niceTicks(x0, x1);
  ctx.strokeStyle = T.grid; ctx.fillStyle = T.text2; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  yt.forEach(([v, s]) => { const y = g.sy(v); if (y < ch.pad.t - 1 || y > g.H - ch.pad.b + 1) return; ctx.beginPath(); ctx.moveTo(ch.pad.l, y); ctx.lineTo(g.W - ch.pad.r, y); ctx.stroke(); ctx.fillText(s, ch.pad.l - 4, y); });
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  xt.forEach(([v, s]) => { const x = g.sx(v); if (x < ch.pad.l - 1 || x > g.W - ch.pad.r + 1) return; ctx.fillText(s, x, g.H - ch.pad.b + 3); });
  ctx.fillStyle = T.muted; ctx.fillText(c.xlabel, (ch.pad.l + g.W - ch.pad.r) / 2, g.H - 13);
  ctx.save(); ctx.translate(10, (g.H - ch.pad.b + ch.pad.t) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "middle"; ctx.fillText(c.ylabel, 0, 0); ctx.restore();
  ctx.strokeStyle = T.axis; ctx.beginPath(); ctx.moveTo(ch.pad.l, ch.pad.t); ctx.lineTo(ch.pad.l, g.H - ch.pad.b); ctx.lineTo(g.W - ch.pad.r, g.H - ch.pad.b); ctx.stroke();
  ctx.lineJoin = "round"; ctx.lineCap = "round";
  for (const s of c.series) {
    const xs = c.kind === "series" ? c.x : s.x;
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.8; ctx.setLineDash(s.dash ? [5, 4] : []);
    ctx.beginPath(); let pen = false;
    s.y.forEach((y, i) => { if (y === null || xs[i] === null) { pen = false; return; } const X = g.sx(xs[i]), Y = g.sy(y); pen ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); pen = true; });
    ctx.stroke(); ctx.setLineDash([]);
  }
  ch.base = ctx.getImageData(0, 0, ch.cv.width, ch.cv.height);
}
function draw(ch) {
  const g = geom(ch), { ctx, c } = ch, T = DATA.theme;
  if (!ch.base) drawBase(ch, g); else ctx.putImageData(ch.base, 0, 0);
  ctx.setTransform(g.dpr, 0, 0, g.dpr, 0, 0);
  if (c.kind === "series") {
    const k = Math.min(K, c.x.length - 1), X = g.sx(c.x[k]);
    ctx.strokeStyle = T.cursor; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X, ch.pad.t); ctx.lineTo(X, g.H - ch.pad.b); ctx.stroke();
    for (const s of c.series) if (s.y[k] !== null) drawDot(ctx, X, g.sy(s.y[k]), 3.5, s.color, T.panel);
    if (ch.hoverK != null && ch.hoverK !== k) { const Xh = g.sx(c.x[ch.hoverK]); ctx.strokeStyle = T.muted; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(Xh, ch.pad.t); ctx.lineTo(Xh, g.H - ch.pad.b); ctx.stroke(); ctx.setLineDash([]); }
  } else {
    for (const m of c.markers) if (m.y[K] !== null) drawDot(ctx, g.sx(m.x[K]), g.sy(m.y[K]), 5, m.color, T.text);
  }
}
function drawDot(ctx, x, y, r, fill, ring) { ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fillStyle = fill; ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = ring; ctx.stroke(); }
function nearestK(ch, e) {
  const g = geom(ch), r = ch.cv.getBoundingClientRect(), xv = g.ix(e.clientX - r.left), xs = ch.c.x;
  let best = 0, bd = Infinity; xs.forEach((x, i) => { const d = Math.abs(x - xv); if (d < bd) { bd = d; best = i; } }); return Math.min(best, N - 1);
}
function hover(ch, e) {
  const c = ch.c, T = DATA.theme; let html = "";
  if (c.kind === "series") {
    const k = nearestK(ch, e); ch.hoverK = k; draw(ch);
    html = `<b>${c.xlabel}: ${fmt(c.x[k], 3)}</b><br>` + c.series.map((s) => `<span style="color:${s.color}">●</span> ${s.label}: ${s.y[k] === null ? "–" : fmt(s.y[k], 3)}`).join("<br>") + `<br><span style="color:${T.muted}">click or drag to move the animation</span>`;
  } else {
    html = `<b>${c.title}</b> at ${DATA.time_label} = ${fmt(DATA.t[K], 2)} ${DATA.time_unit}<br>` + c.markers.map((m) => `<span style="color:${m.color}">●</span> ${m.label}: ${fmt(m.x[K], 3)}, ${fmt(m.y[K], 3)}`).join("<br>");
  }
  tip.innerHTML = html; tip.style.display = "block";
  tip.style.left = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8) + "px"; tip.style.top = (e.clientY + 12) + "px";
}

// ---------------------------------------------------------------- time
const slider = $("#slider"); slider.max = N - 1;
slider.oninput = () => { pause(); setFrame(+slider.value); };
function setFrame(k) {
  K = Math.max(0, Math.min(N - 1, k)); slider.value = K; showFrame(K);
  $("#tlab").textContent = `${DATA.time_label} = ${fmt(DATA.t[K], 2)} ${DATA.time_unit}  (${K + 1}/${N})`;
  DATA.tiles.forEach((t, i) => { document.getElementById("tile" + i).textContent = t.text[Math.min(K, t.text.length - 1)]; });
  charts.forEach(draw); if (window.view3d) window.view3d(K);
}
function pause() { playing = false; $("#play").textContent = "Play"; }
$("#play").onclick = () => { playing = !playing; $("#play").textContent = playing ? "Pause" : "Play"; if (playing && K >= N - 1) setFrame(0); last = performance.now(); acc = 0; };
document.addEventListener("keydown", (e) => { if (e.key === "ArrowRight") { pause(); setFrame(K + 1); } if (e.key === "ArrowLeft") { pause(); setFrame(K - 1); } if (e.key === " ") { e.preventDefault(); $("#play").click(); } });
(function loop(now) {
  if (playing) { acc += (now - last) / 1000 * MEDIA.fps * +$("#speed").value; const step = Math.floor(acc); if (step > 0) { acc -= step; if (K + step >= N) { setFrame(N - 1); pause(); } else setFrame(K + step); } }
  last = now; requestAnimationFrame(loop);
})(performance.now());
new ResizeObserver(() => charts.forEach((c) => { c.base = null; draw(c); })).observe(document.body);
setFrame(0);
/*VIEW3D*/
</script>
</body></html>
"""

VIEW3D = r"""
// ---------------------------------------------------------------- live 3D view (studio three.js)
const THREE = await import("three");
const { OrbitControls } = await import("three-orbit");
const tabs = document.createElement("div"); tabs.className = "tabs";
tabs.innerHTML = `<button class="on" data-v="render">Rendered</button><button data-v="3d">Live 3D</button>`;
stage.appendChild(tabs);
const canvas = document.createElement("canvas"); canvas.className = "v3"; canvas.style.display = "none"; stage.appendChild(canvas);
const dec = (b64, T) => { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new T(u.buffer); };
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene(), L = SCENE3D.lighting || { background: ["#000000", "#000000"], sun: { direction: [-0.8, -0.4, 0.4], intensity: 3, color: [1, 1, 1] }, ambient: 0.1 };
scene.background = new THREE.Color(L.background[1]);
const Z = (p) => new THREE.Vector3(p[0], p[2], -p[1]);                     // z up -> three.js y up
const sun = new THREE.DirectionalLight(new THREE.Color(...L.sun.color), 1.4 * L.sun.intensity); scene.add(sun); scene.add(sun.target);
scene.add(new THREE.HemisphereLight(0xffffff, 0x222222, 0.6 * L.ambient + 0.15));
const camera = new THREE.PerspectiveCamera(35, 16 / 9, 0.02, 500);
const controls = new OrbitControls(camera, canvas); controls.enableDamping = true;
const groups = SCENE3D.bodies.map((b) => {
  const g = new THREE.Group(); g.userData = { pos: dec(b.pos, Float32Array), quat: dec(b.quat, Float32Array) };
  for (const p of b.parts) {
    const geo = new THREE.BufferGeometry(); const V = dec(p.v, Float32Array);
    const W = new Float32Array(V.length); for (let i = 0; i < V.length; i += 3) { W[i] = V[i]; W[i + 1] = V[i + 2]; W[i + 2] = -V[i + 1]; }
    geo.setAttribute("position", new THREE.BufferAttribute(W, 3)); geo.setIndex(new THREE.BufferAttribute(dec(p.f, Uint32Array), 1)); geo.computeVertexNormals();
    g.add(new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: new THREE.Color(...p.rgb.map(lin)), roughness: 0.55, metalness: 0.1, flatShading: true })));
  }
  scene.add(g); return g;
});
if (SCENE3D.terrain) {
  const t = SCENE3D.terrain, z = dec(t.z, Float32Array), rgb = dec(t.rgb, Uint8Array);
  const geo = new THREE.PlaneGeometry(1, 1, t.nx - 1, t.ny - 1), P = geo.attributes.position, C = new Float32Array(P.count * 3);
  for (let j = 0; j < t.ny; j++) for (let i = 0; i < t.nx; i++) {
    const q = j * t.nx + i, v = Z([t.x0 + i * t.dx, t.y0 + j * t.dy, z[q]]); P.setXYZ(q, v.x, v.y, v.z);
    for (let c = 0; c < 3; c++) C[3 * q + c] = lin(rgb[3 * q + c] / 255);
  }
  geo.setAttribute("color", new THREE.BufferAttribute(C, 3)); geo.computeVertexNormals();
  scene.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide })));
}
function place(k) {
  for (const g of groups) { const p = g.userData.pos, q = g.userData.quat;
    g.position.copy(Z([p[3 * k], p[3 * k + 1], p[3 * k + 2]]));
    // rotation z-up -> y-up: conjugate by the axis change (x, y, z) -> (x, z, -y)
    g.quaternion.set(q[4 * k + 1], q[4 * k + 3], -q[4 * k + 2], q[4 * k]); }
  const c = groups[0].position, d = Z(L.sun.direction);
  sun.position.copy(c).addScaledVector(d, 20); sun.target.position.copy(c);
}
place(0); const c0 = groups[0].position.clone(); camera.position.copy(c0).add(new THREE.Vector3(-4.5, 2.6, 4.5)); controls.target.copy(c0).add(new THREE.Vector3(0, 0.6, 0));
let follow = groups[0].position.clone();
window.view3d = (k) => { place(k); const c = groups[0].position; const d = c.clone().sub(follow); camera.position.add(d); controls.target.add(d); follow = c.clone(); };
function size3d() { const w = stage.clientWidth, h = Math.round(w * MEDIA.h / MEDIA.w); renderer.setSize(w, h, false); canvas.style.height = h + "px"; camera.aspect = w / h; camera.updateProjectionMatrix(); }
new ResizeObserver(size3d).observe(stage); size3d();
(function r3() { if (canvas.style.display !== "none") { controls.update(); renderer.render(scene, camera); } requestAnimationFrame(r3); })();
tabs.querySelectorAll("button").forEach((b) => b.onclick = () => {
  tabs.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
  const three = b.dataset.v === "3d"; canvas.style.display = three ? "block" : "none"; media.style.display = three ? "none" : "block"; size3d();
});
"""

__all__ = ["write_html", "payload", "scene3d_payload"]
