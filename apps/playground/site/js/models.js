// Model loading and inference. Default engine: the built-in ONNX interpreter (js/onnx.js). With engine=ort in the
// URL hash the same .onnx files run in onnxruntime-web instead (loaded on demand from a CDN, exact version pinned).
import { Session } from "./onnx.js";

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.mjs";
const cache = new Map();
let engine = "builtin";

export function setEngine(e) { if (e !== engine) { engine = e === "ort" ? "ort" : "builtin"; cache.clear(); } }
export const getEngine = () => engine;

async function ortSession(bytes) {
  const ort = await import(ORT_URL);
  ort.env.wasm.wasmPaths = ORT_URL.replace(/[^/]+$/, "");
  ort.env.wasm.numThreads = 1;
  const s = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  return {
    outputNames: s.outputNames,
    async run(feeds) {
      const f = {};
      for (const [k, v] of Object.entries(feeds)) f[k] = new ort.Tensor("float32", Float32Array.from(v.data), v.dims);
      const r = await s.run(f), out = {};
      for (const k of s.outputNames) out[k] = { dims: [...r[k].dims], data: Float64Array.from(r[k].data) };
      return out;
    },
  };
}

/** {meta, run(rows) -> {output: Float64Array rows x k}} for a zoo model. */
export function load(name) {
  if (!cache.has(name)) {
    cache.set(name, (async () => {
      const [meta, bytes] = await Promise.all([
        fetch(`models/${name}.json`).then((r) => r.json()),
        fetch(`models/${name}.onnx`).then((r) => { if (!r.ok) throw new Error(`models/${name}.onnx: HTTP ${r.status}`); return r.arrayBuffer(); }),
      ]);
      const sess = engine === "ort" ? await ortSession(new Uint8Array(bytes)) : new Session(new Uint8Array(bytes));
      const nIn = meta.onnx.inputs.length;
      return {
        meta, engine, bytes: bytes.byteLength,
        /** rows: array of input rows in the card's input order. Returns {name: array of rows}. */
        async run(rows) {
          const data = new Float64Array(rows.length * nIn);
          rows.forEach((r, i) => data.set(r, i * nIn));
          const t0 = performance.now();
          const out = await sess.run({ x: { dims: [rows.length, nIn], data } });
          this.lastMs = performance.now() - t0;
          const res = {};
          for (const [k, t] of Object.entries(out)) {
            const w = t.dims[t.dims.length - 1];
            res[k] = rows.map((_, i) => Array.from(t.data.subarray(i * w, (i + 1) * w)));
          }
          return res;
        },
      };
    })());
  }
  return cache.get(name);
}
