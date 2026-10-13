// Airfoil mode: drag the CST shape (or pick a NACA preset) and the angle of attack; the zoo ensemble predicts
// Cl, Cd, Cm with its spread, sweeps the polar, and the live checks say whether the answer can be trusted.
import * as cst from "../cst.js";
import { LineChart } from "../charts.js";
import { writeState } from "../main.js";
import { load } from "../models.js";
import { renderModelTrust } from "../trust.js";
import { clamp, fmt, h, rafThrottle, renderVerdict, segmented, slider, svg, tile } from "../ui.js";

export const title = "Airfoil";
const MODEL = "airfoil_cst_rans_mlp";
const A_VALID = [-2, 14], A_SLIDER = [-6, 18];
const SWEEP = Array.from({ length: 49 }, (_, i) => -6 + i * 0.5);
const EXTRA_PRESETS = { "NACA 0006": "0006", "NACA 0025": "0025", "NACA 8412": "8412" };
const DEFAULT = { preset: "NACA 2412", alpha: 4 };

// drawing frame: chord 0..1 -> 600 px, leading edge at x=110, chord line at y=190
const S = 600, OX = 110, OY = 190, PIV = 0.25;
const P = (x, y) => [OX + x * S, OY - y * S];

export async function mount(root, params) {
  const model = await load(MODEL);
  const live = model.meta.live;
  const presets = { ...live.presets };
  for (const [k, code] of Object.entries(EXTRA_PRESETS)) presets[k] = cst.naca4(code);
  const st = { shape: presets[DEFAULT.preset].slice(), alpha: DEFAULT.alpha, chart: "cl" };

  // ------------------------------------------------------------------ layout
  const chips = h("div", { class: "chips", role: "group", "aria-label": "Presets" });
  for (const name of Object.keys(presets)) {
    const out = cst.outOfBounds(presets[name]).length > 0;
    chips.append(h("button", { class: `chip${out ? " out" : ""}`, type: "button", "data-preset": name, title: out ? "outside the model's CST bounds" : "", onclick: () => { st.shape = presets[name].slice(); syncSliders(); update(); } }, name));
  }
  const alpha = slider({ label: "Angle of attack α", unit: "deg", min: A_SLIDER[0], max: A_SLIDER[1], step: 0.25, value: st.alpha, valid: A_VALID, digits: 2, testid: "alpha", onInput: (v) => { st.alpha = v; update(); } });
  const weightSliders = cst.PARAMS.map((p, i) => {
    const [lo, hi] = cst.BOUNDS[i], span = hi - lo;
    return slider({ label: p, min: +(lo - 0.6 * span).toFixed(3), max: +(hi + 0.6 * span).toFixed(3), step: 0.0005, value: st.shape[i], valid: [lo, hi], digits: 4, testid: `w-${p}`, onInput: (v) => { st.shape[i] = v; update(); } });
  });
  const syncSliders = () => { weightSliders.forEach((s, i) => s.set(st.shape[i])); alpha.set(st.alpha); };

  const controls = h("div", { class: "card" },
    h("h3", {}, "Shape"), chips,
    h("p", { class: "hint" }, "Drag the round handles on the airfoil, or pick a preset. Dashed presets sit outside the model's range on purpose."),
    alpha.el,
    h("details", { class: "more" }, h("summary", {}, "CST weights (thickness T, camber M)"), ...weightSliders.map((s) => s.el),
      h("p", { class: "hint" }, "Green: the bounds the model was trained on. upper = M + T, lower = M − T.")));

  const drawing = svg("svg", { class: "draw", viewBox: "0 0 820 380", "aria-label": "Airfoil shape; drag the handles to edit", "data-testid": "airfoil-svg" });
  const overlay = h("div", { class: "overlay" }, h("span", { "data-testid": "stage-alpha" }), h("span", {}, "Re 4·10⁶ · incompressible · k-ω SST"));
  const stage = h("div", { class: "card stage" }, overlay, drawing,
    h("div", { class: "note", "data-testid": "field-note" }, "Pressure and velocity fields: not available yet. This zoo model predicts integrated coefficients (Cl, Cd, Cm); a field model is planned for the next zoo release, and the field view will appear here when it ships."));

  const chartCard = h("div", { class: "card" });
  const chartHead = h("h3", {}, h("span", {}, "Polar sweep"), segmented([["cl", "Cl–α"], ["ld", "L/D–α"], ["polar", "Cl–Cd"], ["cm", "Cm–α"]], st.chart, (k) => { st.chart = k; drawChart(); }));
  chartCard.append(chartHead);
  const chart = new LineChart(chartCard, { xLabel: "α (deg)", yLabel: "Cl", testid: "polar-chart" });

  const tLD = tile("Lift-to-drag L/D", { hero: true, testid: "LD" }), tCl = tile("Lift Cl", { testid: "Cl" }), tCd = tile("Drag Cd", { testid: "Cd" }), tCm = tile("Moment Cm c/4", { testid: "Cm" }), tT = tile("Thickness t/c", { testid: "thickness" });
  const tCam = tile("Max camber", { testid: "camber" });
  const results = h("div", { class: "card" }, h("h3", {}, h("span", {}, "Prediction"), h("span", { class: "hint", style: "margin:0;text-transform:none;letter-spacing:0", "data-testid": "latency" })),
    h("div", { class: "tiles" }, tLD.el, tCl.el, tCd.el, tCm.el, tT.el, tCam.el),
    h("p", { class: "hint" }, "± is the spread of the 5-member ensemble (1σ)."));
  const verdict = h("div"), design = h("div");
  const trustCard = h("div", { class: "card" }, h("h3", {}, "Trust verdict"), verdict, h("h3", { style: "margin-top:14px" }, "Design constraints"), design,
    h("p", { class: "hint" }, h("a", { href: "#", onclick: (e) => { e.preventDefault(); document.getElementById("trust-open").click(); } }, "Why trust this? Model card and evidence →")));

  root.append(h("div", { class: "mode" },
    h("div", { class: "col-left stack" }, controls),
    h("div", { class: "col-center stack" }, stage, chartCard),
    h("div", { class: "col-right stack" }, results, trustCard)));
  stage.classList.add("o1"); results.classList.add("o2"); controls.classList.add("o3"); trustCard.classList.add("o4"); chartCard.classList.add("o5");

  // ------------------------------------------------------------------ drawing + drag
  let dragging = null, sweep = null, pred = null;
  const gShape = svg("g"), gFixed = svg("g");
  const gFlow = svg("g");
  drawing.append(svg("defs", {}, svg("marker", { id: "ah", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse" }, svg("path", { d: "M0,0 L10,5 L0,10 z", fill: "context-stroke" }))), gFlow, gShape, gFixed);
  for (let k = 0; k < 6; k++) gFlow.append(svg("path", { class: "flow-arrow", d: `M14,${50 + k * 58} L64,${50 + k * 58}`, "marker-end": "url(#ah)" }));
  gFlow.append(svg("text", { x: 14, y: 368, fill: "var(--muted)", "font-size": 12 }, "V∞"));

  function localPoint(evt) {
    const m = gShape.getScreenCTM().inverse(), pt = new DOMPoint(evt.clientX, evt.clientY).matrixTransform(m);
    return [(pt.x - OX) / S, (OY - pt.y) / S];
  }
  drawing.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const [, y] = localPoint(e), hd = cst.handles(st.shape);
    const yu = hd.upper.map((p) => p[1]), yl = hd.lower.map((p) => p[1]), i = dragging.i;
    if (dragging.side === "upper") yu[i] = clamp(y, yl[i] + 0.004, 0.3); else yl[i] = clamp(y, -0.3, yu[i] - 0.004);
    st.shape = cst.shapeFromHandles(yu, yl);
    syncSliders(); update();
  });
  const endDrag = () => { if (dragging) { dragging.el.classList.remove("drag"); dragging = null; } };
  drawing.addEventListener("pointerup", endDrag); drawing.addEventListener("pointercancel", endDrag);

  function drawShape() {
    gShape.innerHTML = "";
    const [px, py] = P(PIV, 0);
    gShape.setAttribute("transform", `rotate(${st.alpha} ${px} ${py})`);
    const pts = cst.outline(st.shape, 140).map(([x, y]) => P(x, y)), out = cst.outOfBounds(st.shape).length > 0 || !cst.valid(st.shape)[0];
    gShape.append(svg("path", { d: `M${pts.map((p) => p.map((v) => v.toFixed(1)).join(",")).join("L")}Z`, fill: out ? "var(--bad-bg)" : "var(--shape-fill)", stroke: out ? "var(--bad)" : "var(--shape-line)", "stroke-width": 2, "stroke-linejoin": "round", "data-testid": "airfoil-path" }));
    const xs = cst.cosine(60), [yu, yl] = cst.surfaces(st.shape, xs);
    gShape.append(svg("path", { d: `M${xs.map((x, i) => P(x, (yu[i] + yl[i]) / 2).map((v) => v.toFixed(1)).join(",")).join("L")}`, fill: "none", stroke: "var(--shape-line)", "stroke-width": 1, "stroke-dasharray": "4 4", opacity: 0.6 }));
    gShape.append(svg("line", { x1: P(0, 0)[0], y1: OY, x2: P(1, 0)[0], y2: OY, stroke: "var(--axis)", "stroke-width": 1, "stroke-dasharray": "2 5" }));
    const hd = cst.handles(st.shape);
    for (const side of ["upper", "lower"]) hd[side].forEach(([x, y], i) => {
      const [cx, cy] = P(x, y);
      const c = svg("circle", { class: "handle", cx, cy, r: 8, "data-handle": `${side}${i}`, tabindex: 0, role: "slider", "aria-label": `${side} surface handle ${i + 1}`, "aria-valuenow": y.toFixed(4) });
      const hit = svg("circle", { class: "handle-hit", cx, cy, r: 20 });
      const start = (e) => { e.preventDefault(); drawing.setPointerCapture(e.pointerId); dragging = { side, i, el: c }; c.classList.add("drag"); };
      hit.addEventListener("pointerdown", start); c.addEventListener("pointerdown", start);
      c.addEventListener("keydown", (e) => {
        const d = { ArrowUp: 0.002, ArrowDown: -0.002 }[e.key]; if (!d) return;
        e.preventDefault();
        const yu2 = hd.upper.map((p) => p[1]), yl2 = hd.lower.map((p) => p[1]);
        (side === "upper" ? yu2 : yl2)[i] += d;
        if (yu2[i] - yl2[i] > 0.004) { st.shape = cst.shapeFromHandles(yu2, yl2); syncSliders(); update(); requestAnimationFrame(() => drawing.querySelector(`[data-handle="${side}${i}"]`)?.focus()); }
      });
      gShape.append(hit, c);
    });
  }

  function drawForces() {
    gFixed.innerHTML = "";
    if (!pred) return;
    // forces at the quarter chord: lift normal to the free stream, drag along it (drawn 25x longer, it is small)
    const [px, py] = P(PIV, 0), L = clamp(pred.Cl, -1.2, 2.4) * 70, D = clamp(pred.Cd * 70 * 25, 0, 300);
    gFixed.append(svg("line", { class: "force lift", x1: px, y1: py, x2: px, y2: py - L, "marker-end": "url(#ah)" }));
    gFixed.append(svg("text", { x: px - 8, y: py - L + (L > 0 ? -8 : 18), "text-anchor": "end", fill: "var(--series-3)", "font-size": 14, "font-weight": 650 }, `Lift  Cl ${fmt(pred.Cl, 3)}`));
    gFixed.append(svg("line", { class: "force drag", x1: px, y1: py, x2: px + D, y2: py, "marker-end": "url(#ah)" }));
    gFixed.append(svg("text", { x: px + D + 8, y: py + 5, fill: "var(--series-2)", "font-size": 14, "font-weight": 650 }, `Drag ×25`));
    gFixed.append(svg("circle", { cx: px, cy: py, r: 4, fill: "var(--ink)" }));
  }

  // ------------------------------------------------------------------ model + checks
  function stats(row) {
    const K = row.length / 3, cl = [], cd = [], cm = [], ld = [];
    for (let k = 0; k < K; k++) { cl.push(row[3 * k]); cd.push(row[3 * k + 1]); cm.push(row[3 * k + 2]); ld.push(row[3 * k] / row[3 * k + 1]); }
    const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length, std = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((v) => (v - m) ** 2))); };
    return { Cl: mean(cl), Cd: mean(cd), Cm: mean(cm), LD: mean(ld), Cl_std: std(cl), Cd_std: std(cd), Cm_std: std(cm), LD_std: std(ld) };
  }

  function checks(props, okGeo) {
    const out = [], bad = cst.outOfBounds(st.shape);
    out.push({ id: "cst-range", rule: "PC601", name: "Shape inside the CST bounds the model was trained on", status: bad.length ? "fail" : "pass",
      detail: bad.length ? bad.map((p) => { const i = cst.PARAMS.indexOf(p); const [lo, hi] = cst.BOUNDS[i], by = st.shape[i] > hi ? st.shape[i] - hi : lo - st.shape[i];
        return `${p} = ${fmt(st.shape[i], 4)}, ${(100 * by / (hi - lo)).toFixed(1)} % of its range ${st.shape[i] > hi ? "above" : "below"} the bound (${lo} … ${hi})`; }).join("; ") : "all 6 CST weights in range" });
    const aOut = st.alpha < A_VALID[0] || st.alpha > A_VALID[1];
    out.push({ id: "alpha-range", rule: "PC601", name: `Angle of attack inside ${A_VALID[0]}…${A_VALID[1]}°`, status: aOut ? "fail" : "pass", detail: aOut ? `α = ${st.alpha}°: ${st.alpha > A_VALID[1] ? "beyond the trained range, towards deep stall (excluded from training)" : "below the trained range"}` : "" });
    out.push({ id: "geometry", rule: "PC402", name: "Valid airfoil: surfaces do not cross, no wavy surface", status: okGeo[0] ? "pass" : "fail", detail: okGeo[1] });
    const finite = [pred.Cl, pred.Cd, pred.Cm].every(Number.isFinite) && pred.Cd > 0;
    out.push({ id: "finite", rule: "PC403", name: "Outputs finite, drag positive", status: finite ? "pass" : "fail" });
    const sp = live.spread, spreadHigh = pred.Cl_std > sp.Cl_std_p99 || pred.Cd_std / pred.Cd > sp.Cd_std_rel_p99;
    out.push({ id: "spread", rule: "PC602", name: "Ensemble agrees as well as on held-out shapes", status: spreadHigh ? "warn" : "pass",
      detail: `σ(Cl) ${fmt(pred.Cl_std, 4)} (held-out p99 ${fmt(sp.Cl_std_p99, 4)}), σ(Cd)/Cd ${(100 * pred.Cd_std / pred.Cd).toFixed(2)} % (p99 ${(100 * sp.Cd_std_rel_p99).toFixed(2)} %)` });
    if (sweep) {
      const lin = SWEEP.map((a, i) => [a, sweep[i].Cl]).filter(([a]) => a >= -2 && a <= 6);
      const n = lin.length, mx = lin.reduce((s, p) => s + p[0], 0) / n, my = lin.reduce((s, p) => s + p[1], 0) / n;
      const slope = lin.reduce((s, p) => s + (p[0] - mx) * (p[1] - my), 0) / lin.reduce((s, p) => s + (p[0] - mx) ** 2, 0);
      const dev = Math.abs(slope / live.thin_airfoil_slope_per_deg - 1);
      out.push({ id: "slope", rule: "PC302", name: "Lift slope within 15 % of thin-airfoil 2π/rad", status: dev <= live.slope_tolerance ? "pass" : "warn", detail: `dCl/dα = ${fmt(slope, 4)}/deg (${(100 * dev).toFixed(1)} % from 0.1097/deg)` });
      const mono = SWEEP.map((a, i) => [a, sweep[i].Cl]).filter(([a]) => a >= -2 && a <= 10).every((p, i, arr) => i === 0 || p[1] > arr[i - 1][1]);
      out.push({ id: "monotonic", rule: "PC302", name: "Cl rises with α below stall (−2…10°)", status: mono ? "pass" : "warn" });
    }
    const dc = live.design_constraints;
    const design = [
      { id: "thickness", name: `Thickness t/c ≥ ${dc.min_thickness * 100} %`, status: props.thickness >= dc.min_thickness ? "pass" : "warn", detail: `t/c = ${(100 * props.thickness).toFixed(1)} % at x/c ${props.thickness_at.toFixed(2)}` },
      { id: "moment", name: `|Cm| ≤ ${dc.max_abs_cm}`, status: Math.abs(pred.Cm) <= dc.max_abs_cm ? "pass" : "warn", detail: `Cm = ${fmt(pred.Cm, 4)} at α ${st.alpha}°` },
    ];
    return [out, design];
  }

  function matchPreset() {
    for (const [k, v] of Object.entries(presets)) if (v.every((w, i) => Math.abs(w - st.shape[i]) < 2e-4)) return k;
    return null;
  }

  function drawChart() {
    if (!sweep) return;
    const preset = matchPreset(), y = (k) => sweep.map((p) => p[k]), sd = (k) => sweep.map((p) => p[`${k}_std`]);
    const band = (k) => ({ lo: y(k).map((v, i) => v - 2 * sd(k)[i]), hi: y(k).map((v, i) => v + 2 * sd(k)[i]) });
    const refs = (key) => live.references.filter((r) => r.preset === preset).map((r, j) => ({ name: r.source, color: j ? "var(--series-3)" : "var(--series-2)", x: key === "polar" ? r.Cd : r.alpha, y: key === "polar" ? r.Cl : key === "ld" ? r.Cl.map((c, i) => c / r.Cd[i]) : key === "cm" ? r.Cm : r.Cl })).map((r) => ({ ...r, x: r.x.filter((_, i) => Number.isFinite(r.y[i])), y: r.y.filter(Number.isFinite) }));
    const cfg = {
      cl: { yLabel: "Cl", key: "Cl" }, ld: { yLabel: "L/D", key: "LD" }, cm: { yLabel: "Cm c/4", key: "Cm" },
      polar: { yLabel: "Cl", key: "Cl" },
    }[st.chart];
    chart.yLabel = cfg.yLabel; chart.xLabel = st.chart === "polar" ? "Cd" : "α (deg)";
    const name = `${MODEL} (ensemble mean)`;
    if (st.chart === "polar") {
      const keep = SWEEP.map((a) => a >= A_VALID[0] && a <= A_VALID[1]);
      chart.update({ series: [{ name, x: y("Cd").filter((_, i) => keep[i]), y: y("Cl").filter((_, i) => keep[i]), color: "var(--series-1)" }], refs: refs("polar"), marker: { x: pred.Cd, y: pred.Cl }, xFmt: (v) => v.toFixed(4), ariaLabel: "Drag polar, Cl against Cd" });
    } else {
      chart.update({ series: [{ name, x: SWEEP, y: y(cfg.key), ...band(cfg.key), color: "var(--series-1)" }], refs: refs(st.chart), marker: { x: st.alpha, y: pred[cfg.key] }, xValid: A_VALID, ariaLabel: `${cfg.yLabel} against angle of attack` });
    }
  }

  let seq = 0;
  async function compute() {
    const my = ++seq;
    const rows = [[...st.shape, st.alpha], ...SWEEP.map((a) => [...st.shape, a])];
    const out = await model.run(rows);
    if (my !== seq) return;
    const all = out.members.map(stats);
    pred = all[0]; sweep = all.slice(1);
    pred.mean = out.mean[0];
  }

  async function update() {
    writeState("airfoil", { s: st.shape.map((v) => v.toFixed(5)).join(","), a: st.alpha, ...(st.chart !== "cl" ? { c: st.chart } : {}) });
    drawShape();
    const props = cst.properties(st.shape), okGeo = cst.valid(st.shape);
    await compute();
    if (!pred) return;
    drawForces();
    overlay.firstChild.textContent = `α = ${st.alpha.toFixed(2)}°`;
    const ld = pred.LD;
    tLD.set(`${fmt(ld, 1)} <small>± ${fmt(pred.LD_std, 1)}</small>`, "ensemble mean ± 1σ", ld);
    tCl.set(fmt(pred.Cl, 3), `± ${fmt(pred.Cl_std, 4)}`, pred.Cl);
    tCd.set(`${(pred.Cd * 1e4).toFixed(1)} <small>counts</small>`, `Cd ${fmt(pred.Cd, 5)} ± ${(pred.Cd_std * 1e4).toFixed(1)} ct`, pred.Cd);
    tCm.set(fmt(pred.Cm, 4), `± ${fmt(pred.Cm_std, 4)}`, pred.Cm);
    tT.set(`${(100 * props.thickness).toFixed(1)} <small>%</small>`, `at x/c ${props.thickness_at.toFixed(2)}`, props.thickness);
    tCam.set(`${(100 * props.camber).toFixed(2)} <small>%</small>`, `at x/c ${props.camber_at.toFixed(2)}`, props.camber);
    document.querySelector("[data-testid=latency]").textContent = `${SWEEP.length + 1} evaluations in ${model.lastMs >= 1 ? model.lastMs.toFixed(0) : "<1"} ms`;
    const [cks, dcs] = checks(props, okGeo);
    renderVerdict(verdict, cks);
    design.innerHTML = "";
    design.append(h("ul", { class: "checks" }, dcs.map((c) => h("li", { class: c.status, "data-check": c.id }, h("span", { class: "ico", html: c.status === "pass" ? '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>' : '<svg viewBox="0 0 24 24"><path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17.5v.01"/></svg>' }), h("span", {}, c.name, h("span", { class: "det" }, c.detail))))));
    chips.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.preset === matchPreset())));
    drawChart();
    renderModelTrust({ meta: model.meta, checks: cks, extra: h("div", {}, h("h4", {}, "Independent reference on the polar"), h("p", {}, "Pick NACA 2412, 0012, 4412 or 4415: the polar shows OpenFOAM runs the model never saw and, for NACA 2412, an XFOIL polar from an independent solver.")) });
  }
  const doUpdate = rafThrottle(update);

  function setState(p) {
    const s = (p.get("s") || "").split(",").map(Number);
    if (s.length === 6 && s.every(Number.isFinite)) st.shape = s;
    else if (p.get("preset") && presets[p.get("preset")]) st.shape = presets[p.get("preset")].slice();
    const a = Number(p.get("a"));
    if (p.has("a") && Number.isFinite(a)) st.alpha = clamp(a, A_SLIDER[0], A_SLIDER[1]);
    if (["cl", "ld", "polar", "cm"].includes(p.get("c"))) st.chart = p.get("c");
    chartHead.querySelectorAll(".seg button").forEach((b, i) => b.setAttribute("aria-pressed", String(["cl", "ld", "polar", "cm"][i] === st.chart)));
    syncSliders(); doUpdate();
  }
  setState(params);
  const onTheme = () => drawChart();
  window.addEventListener("themechange", onTheme);
  // expose for tests and the demo recorder
  window.__playground = { mode: "airfoil", state: st, get pred() { return pred; }, update: doUpdate };
  return { setState, unmount() { window.removeEventListener("themechange", onTheme); } };
}

