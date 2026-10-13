# PINNeAPPle Playground

Draw an airfoil, size a heat sink or drive a lunar-rover wheel: a PINNeAPPle zoo model answers in the browser, and
the physics checks and the trust verdict sit next to every number. It is a static site, so nothing is installed
and no server runs any code. The state lives in the URL.

![demo](../../docs/assets/playground/demo.gif)

```bash
python apps/playground/build.py --serve          # build into apps/playground/dist and serve on :8765
```

The app is UI code only. The models, query checks, geometry and their runtime are library features:

| Library | What the Playground uses |
|---|---|
| `pp.zoo.export_web`, `python -m pinneapple_hub.zoo.web data/web/zoo` | Each zoo model as one ONNX graph (predictions + query checks, parity with PyTorch ≤ 1e-5), its card and check spec |
| `pp.zoo.query_checks`, `pinneapple_hub.zoo.checks` | The per-query trust checks and the verdict (rule IDs of `pinneapple check`) |
| `pinneapple_design.aero.geometry_onnx` | The CST airfoil geometry and the drag handles as ONNX graphs |
| `pp.geometries.export_web`, `python -m pinneapple_design.geometry_library.web data/web/geometries` | Gallery index and thumbnails |
| `pinneapple_hub/zoo/web/onnx.js`, `zoo.js` | The browser runtime: ONNX interpreter, `loadModel`, `loadAirfoilGeometry`, `evaluate` |

| Path | What |
|---|---|
| `site/` | The page: HTML, CSS and the modes' JS (plain ES modules, no bundler). |
| `build.py` | Assembles the static site from the UI and the library's artefacts, adds three.js and the light glTF files, and writes the offline service worker. |
| `tools/record_demo.mjs` | Records the demo video. |
| `tests/e2e/` | Playwright smoke test of every mode against Python. Library tests: `tests/test_zoo_web.py`, `tests/js/zoo_web.test.mjs`. |
| `Dockerfile`, `Caddyfile` | Static image for `apps/deploy` (service `playground`, port 8095). |

Tests:

```bash
pytest tests/test_zoo_web.py tests/test_playground.py            # ONNX parity, query checks, bundle current
cd apps/playground && npm ci && npm test                         # JS runtime tests + browser smoke test
```

Third-party code: three.js r169 (MIT; vendored in `pinneapple_tools`). The tests also use onnxruntime-web 1.22.0
(MIT; loaded from jsDelivr only with `engine=ort`) and @playwright/test 1.56.1 (Apache-2.0; tests only).

How the page works, what the checks mean and how to deploy it: [docs/guides/playground.md](../../docs/guides/playground.md).
