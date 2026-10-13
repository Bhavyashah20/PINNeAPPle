// CST geometry and network-input encoding in JS against pinneapple_design.aero.geometry (Python fixtures).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import * as cst from "../../site/js/cst.js";

const fx = JSON.parse(readFileSync(new URL("../fixtures/geometry.json", import.meta.url)));
const close = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (tol ${tol})`);

test("surfaces match Python at the fixture stations", () => {
  for (const c of fx.cases) {
    const [yu, yl] = cst.surfaces(c.shape, c.x);
    c.x.forEach((_, i) => { close(yu[i], c.upper[i], 1e-12, "upper"); close(yl[i], c.lower[i], 1e-12, "lower"); });
  }
});

test("properties and validity match Python", () => {
  for (const c of fx.cases) {
    const p = cst.properties(c.shape);
    for (const [k, v] of Object.entries(c.properties)) close(p[k], v, 1e-10, k);
    assert.deepEqual(cst.valid(c.shape), c.valid);
  }
  assert.ok(fx.cases.some((c) => !c.valid[0]), "the fixtures include an invalid shape");
});

test("NACA 4-digit CST fit matches numpy lstsq", () => {
  for (const [code, w] of Object.entries(fx.naca4)) cst.naca4(code).forEach((v, i) => close(v, w[i], 1e-9, `NACA ${code} w${i}`));
});

test("network input encoding matches MLPEnsemble.inputs", () => {
  for (const c of fx.cases) cst.encode(c.shape, c.encode_alpha).forEach((v, i) => close(v, c.encode[i], 1e-6, `x${i}`));
});

test("handles round-trip: the shape through its own handles is the same shape", () => {
  for (const c of fx.cases) {
    const hd = cst.handles(c.shape), s = cst.shapeFromHandles(hd.upper.map((p) => p[1]), hd.lower.map((p) => p[1]));
    s.forEach((v, i) => close(v, c.shape[i], 1e-12, `w${i}`));
  }
});

test("out-of-bounds names the offending weights", () => {
  assert.deepEqual(cst.outOfBounds([0.17, 0.15, 0.15, 0.02, 0.05, 0.03]), []);
  assert.deepEqual(cst.outOfBounds([0.30, 0.15, 0.15, 0.02, 0.20, 0.03]), ["T0", "M1"]);
});
