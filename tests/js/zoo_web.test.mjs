// The zoo's browser runtime (pinneapple_hub/zoo/web: onnx.js, zoo.js) against Python, on the bundle written by
// `python -m pinneapple_hub.zoo.web data/web/zoo` (its fixtures.json holds the Python answers).
//   node --test tests/js/zoo_web.test.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { Session } from "../../pinneapple_hub/zoo/web/onnx.js";
import { evaluate, formatTemplate, geometryResult, grouped } from "../../pinneapple_hub/zoo/web/zoo.js";

const B = (p) => new URL(`../../data/web/zoo/${p}`, import.meta.url);
const json = (p) => JSON.parse(readFileSync(B(p)));
const fx = json("fixtures.json");
const session = (f) => new Session(new Uint8Array(readFileSync(B(f))));
const run = (s, feeds) => {
  const f = {};
  for (const [k, rows] of Object.entries(feeds)) f[k] = { dims: [rows.length, rows[0].length], data: rows.flat() };
  return s.run(f);
};
const row = (t, i) => { const w = t.dims[t.dims.length - 1]; return Array.from(t.data.subarray(i * w, (i + 1) * w)); };
// scale of the 1e-5 parity tolerance per predict key (as pinneapple_hub.zoo.export.output_scale)
const SCALE = { Cl: 0.4838, Cm: 0.0356, Cl_std: 0.4838, Cm_std: 0.0356, Fx_N: 833.7, Fz_N: 699.8, My_Nm: 236.8 };

for (const name of ["airfoil_cst_rans_mlp", "heatsink_forced_mlp", "rover_contact_grc1_viper"]) {
  test(`${name}: predictions and query checks match Python`, () => {
    const meta = json(`${name}.json`), f = fx.models[name], X = f.inputs;
    const out = run(session(meta.onnx.file), { x: X, query: X });
    for (const [o, cols] of Object.entries(meta.onnx.outputs)) {
      if (!Array.isArray(cols)) continue;
      cols.forEach((k, j) => X.forEach((_, i) => {
        const ref = f.predict[k][i], got = row(out[o], i)[j];
        const sc = k.endsWith("_std") && k !== "Cl_std" && k !== "Cm_std" ? Math.max(Math.abs(f.predict[k.replace("_std", "")][i]), 1)
          : k.startsWith("LD") ? Math.max(Math.abs(f.predict.LD[i]), SCALE.Cl / f.predict.Cd[i]) // as export.output_scale
          : SCALE[k] ?? Math.abs(ref);
        assert.ok(Math.abs(got - ref) <= 1e-5 * sc, `${name} ${k}[${i}]: ${got} vs ${ref}`);
      }));
    }
    X.forEach((_, i) => {
      const v = row(out.checks, i), a = row(out.aux, i);
      v.forEach((x, j) => assert.ok(Math.abs(x - f.check_values[i][j]) <= 1e-3 * Math.max(Math.abs(f.check_values[i][j]), 1), `${name} check ${j}`));
      const ev = evaluate(meta.checks, v, a);
      assert.deepEqual(ev.checks.map((c) => c.status), f.evaluation[i].statuses, `${name} row ${i} statuses`);
      assert.equal(ev.verdict.status, f.evaluation[i].verdict);
      // the same values give the same text as Python
      const evPy = evaluate(meta.checks, f.check_values[i], f.check_aux[i]);
      assert.deepEqual(evPy.checks.map((c) => c.detail_text), f.evaluation[i].details);
    });
    // a range check fails exactly where ZooModel.in_range is False
    f.in_range.forEach((ok, i) => assert.equal(f.evaluation[i].statuses.some((s, j) => s === "fail" && meta.checks.checks[j].rule === "PC601"), !ok));
  });
}

test("fixtures cover pass and fail verdicts in every model", () => {
  for (const f of Object.values(fx.models)) {
    const v = f.evaluation.map((e) => e.verdict);
    assert.ok(v.includes("pass") && v.includes("fail"));
  }
});

test("CST geometry graphs match pinneapple_design.aero.geometry", () => {
  const meta = json("cst_geometry.json"), g = fx.geometry, S = g.shapes;
  const out = run(session(meta.files.geometry), { x: S });
  for (const k of ["props", "validity", "upper", "lower", "handles"]) {
    S.forEach((_, i) => row(out[k], i).forEach((v, j) => assert.ok(Math.abs(v - g[k][i][j]) <= 1e-9 * Math.max(1, Math.abs(g[k][i][j])), `${k}[${i}][${j}]`)));
  }
  S.forEach((_, i) => {
    const r = geometryResult(meta, Object.fromEntries(Object.keys(out).map((k) => [k, row(out[k], i)])));
    assert.deepEqual(r.valid, g.valid[i]);
  });
  const back = run(session(meta.files.from_handles), { x: g.handles });
  S.forEach((s, i) => row(back.shape, i).forEach((v, j) => assert.ok(Math.abs(v - s[j]) < 1e-12)));
});

test("templates format like Python str.format", () => {
  assert.equal(formatTemplate("{v:.4g}|{w:.2%}|{x:.1f}|{y:.3g}|{z:.2g}", { v: 0.23551, w: 0.0314, x: 12.345, y: 1.234e-5, z: 1234 }), "0.2355|3.14%|12.3|1.23e-05|1.2e+03");
});

test("grouped display: one line per group, worst status", () => {
  const meta = json("airfoil_cst_rans_mlp.json"), f = fx.models.airfoil_cst_rans_mlp;
  const ev = evaluate(meta.checks, f.check_values[4], f.check_aux[4]);
  const g = grouped(meta.checks, ev.checks).find((c) => c.id === "cst-range");
  assert.equal(g.status, "fail");
  assert.match(g.detail_text, /T0 = 0.3/);
});
