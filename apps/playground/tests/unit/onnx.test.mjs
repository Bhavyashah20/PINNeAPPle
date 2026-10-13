// The built-in ONNX interpreter against onnxruntime (Python) on the zoo inputs stored as fixtures, and the
// heat-sink validity mask against HeatsinkForced.in_range.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { Session } from "../../site/js/onnx.js";
import { heatsinkInRange, tractionLimit } from "../../site/js/physics.js";

const site = (p) => new URL(`../../site/${p}`, import.meta.url);
const fixture = (p) => JSON.parse(readFileSync(new URL(`../fixtures/${p}`, import.meta.url)));

for (const name of ["airfoil_cst_rans_mlp", "heatsink_forced_mlp", "rover_contact_grc1_viper"]) {
  test(`${name}: JS interpreter matches onnxruntime to 1e-5`, () => {
    const s = new Session(new Uint8Array(readFileSync(site(`models/${name}.onnx`))));
    const fx = fixture(`onnx_${name}.json`), X = fx.inputs, nIn = X[0].length;
    const out = s.run({ x: { dims: [X.length, nIn], data: X.flat() } });
    for (const [k, ref] of Object.entries(fx.outputs)) {
      const t = out[k], w = ref[0].length;
      assert.deepEqual(t.dims, [X.length, w], `${k} shape`);
      let worst = 0;
      ref.forEach((row, i) => row.forEach((r, j) => {
        let sc = fx.scale[k][j];
        if (sc === null) sc = Math.abs(k === "std" ? fx.outputs.mean[i][j] : r);
        worst = Math.max(worst, Math.abs(t.data[i * w + j] - r) / Math.max(sc, 1e-12));
      }));
      assert.ok(worst <= 1e-5, `${name}/${k}: max scaled error ${worst}`);
    }
  });
}

test("heat-sink validity mask matches HeatsinkForced.in_range", () => {
  const fx = fixture("heatsink_range.json"), meta = JSON.parse(readFileSync(site("models/heatsink_forced_mlp.json")));
  const got = fx.X.map((r) => heatsinkInRange(r, fx.input_names, meta.card.validity_range));
  assert.deepEqual(got, fx.in_range);
  assert.ok(fx.in_range.includes(true) && fx.in_range.includes(false));
});

test("traction limit: cohesion term at zero load, friction slope tan(phi)", () => {
  const c = { R_m: 0.25, b_m: 0.2, c_Pa: 250, phi_deg: 38 };
  assert.ok(Math.abs(tractionLimit(0, 0, c)) < 1e-12);
  const d = tractionLimit(1000, 0.02, c) - tractionLimit(0, 0.02, c);
  assert.ok(Math.abs(d - 1000 * Math.tan((38 * Math.PI) / 180)) < 1e-9);
});
