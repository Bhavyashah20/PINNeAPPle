# PINNeAPPle Playground

Draw an airfoil, size a heat sink or drive a lunar-rover wheel: a PINNeAPPle zoo model answers in the browser, and
the physics checks and the trust verdict sit next to every number. It is a static site, so nothing is installed
and no server runs any code. The state lives in the URL.

![demo](../../docs/assets/playground/demo.gif)

```bash
python apps/playground/build.py --serve          # build into apps/playground/dist and serve on :8765
```

| Path | What |
|---|---|
| `site/` | The page: plain ES modules, no bundler. It holds the ONNX models and their cards (`models/`) and the gallery index (`data/`). |
| `site/js/onnx.js` | The built-in ONNX interpreter. `engine=ort` in the hash switches to onnxruntime-web. |
| `site/js/cst.js` | The CST airfoil geometry, ported from `pinneapple_design.aero.geometry`. |
| `tools/build_assets.py` | Exports the zoo models to ONNX, checks parity with PyTorch (≤ 1e-5), and writes the cards, fixtures and gallery index. |
| `tools/record_demo.mjs` | Records the demo video. |
| `build.py` | Assembles the static site. It adds three.js and the light glTF files, and writes the offline service worker. |
| `tests/` | `unit/` (node --test, against Python fixtures) and `e2e/` (Playwright smoke test). The Python parity test is `tests/test_playground.py`. |
| `Dockerfile`, `Caddyfile` | Static image for `apps/deploy` (service `playground`, port 8094). |

Tests:

```bash
pytest tests/test_playground.py                                  # ONNX vs PyTorch, stale assets
cd apps/playground && npm ci && npm test                         # JS unit tests + browser smoke test
```

Third-party code: three.js r169 (MIT; vendored in `pinneapple_tools`). The tests also use onnxruntime-web 1.22.0
(MIT; loaded from jsDelivr only with `engine=ort`) and @playwright/test 1.56.1 (Apache-2.0; tests only).

How the page works, what the checks mean and how to deploy it: [docs/guides/playground.md](../../docs/guides/playground.md).
