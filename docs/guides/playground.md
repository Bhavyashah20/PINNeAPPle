# Playground: physics surrogates in the browser

The Playground is a static web page. You draw an airfoil, size a heat sink or drive a lunar-rover wheel, and a
PINNeAPPle zoo model answers immediately. Next to each answer the page shows the physics checks and the model's
trust verdict. Everything runs in the browser: nothing is installed and nothing is sent to a server. The URL holds
the whole state, so a link opens exactly what you were looking at.

![Reshaping an airfoil: L/D and the trust verdict follow, and turn red when the shape leaves the trained range](../assets/playground/demo.gif)

| Mode | Model (zoo) | What you change | What you get |
|---|---|---|---|
| Airfoil | `airfoil_cst_rans_mlp` | Drag the CST handles on the airfoil, pick a NACA preset, set the angle of attack | Cl, Cd, Cm and L/D with the spread of the 5-member ensemble; the polar sweep, with the OpenFOAM runs the model never saw and an XFOIL polar overlaid for the NACA presets |
| Heat sink | `heatsink_forced_mlp` | Fin count, height and thickness, air speed (plus base, material, power) | Thermal resistance with its split-conformal 95 % bound, base temperature, pressure drop, and R and ΔP against air speed |
| Rover wheel | `rover_contact_grc1_viper` | Slip and sinkage of the VIPER wheel on GRC-1 | Drawbar pull, normal load and torque, with held-out points of the analytic Wong–Reece model near the current sinkage |
| Geometry gallery | `pp.geometries` | Pick one of the 17 geometries | A 3-D view of the light glTF file, its dimensions and sources, and the code that regenerates it |

## The trust verdict

Every change re-runs the live checks. Their rule IDs are the ones [`pinneapple check`](physics_checks_in_ci.md)
uses:

- **PC601, out-of-range query.** Is every input inside the model card's validity range? This includes the derived
  limits: the fin gap for the heat sink, and the six CST weights and the angle of attack for the airfoil. The page
  uses the same rule as `ZooModel.in_range`. A query that fails it is labelled an extrapolation, in red.
- **PC403 / PC402, non-finite or unphysical output.** For example, a negative drag, or airfoil surfaces that cross.
- **PC602, sparse training region.** The ensemble spread is compared with the 99th percentile of the spread on the
  held-out shapes. A larger spread means the members disagree more than they ever did on the test set.
- **PC302, model residual.** The physics checks of the zoo gate, evaluated at the current query:
  - the lift slope against thin-airfoil theory, and Cl rising with α;
  - more airflow, or a more conductive material, never raising the resistance;
  - the Mohr–Coulomb traction limit, My ≥ R·Fx, and the monotonic trends of the wheel.

The **Why trust this?** panel shows the full model card:

- the validity range;
- the held-out metrics against the independent reference, next to the zoo's acceptance limits;
- the physics checks of the stored trust report;
- data provenance and known limitations;
- the SHA-256 of the ONNX file the page runs.

The airfoil design constraints (t/c ≥ 11 %, |Cm| ≤ 0.10, the defaults of `pinneapple_design.aero.aircraft`) are
shown separately. They say whether a design is acceptable, not whether the model can be trusted.

Note that the NACA 4412 and 4415 presets sit just outside the model's CST box: M1 0.142 > 0.14 and M2 0.123 > 0.12.
`pp.zoo` raises an `OutOfRangeWarning` for them, and the page marks them red. Their OpenFOAM reference runs, which
the model never saw, are overlaid on the polar, so you can judge the extrapolation yourself.

There is no pressure or velocity field yet. The zoo's airfoil model predicts integrated coefficients only, and the
field model (MeshGraphNet) is waiting for its training fields to be regenerated and published. The page says so in
place of a field view.

## How it runs in the browser

Each zoo model is exported to one ONNX graph, from the card's physical inputs to its physical outputs. The graph
includes the CST and log-feature pre-processing, the normalisation, the ensemble mean and spread, and the
conformal bound. The three files total 0.95 MB, and the airfoil mode loads 0.52 MB before compression.

The page evaluates the graph with a small built-in ONNX interpreter (`js/onnx.js`, about 350 lines). It reads the
protobuf directly and supports the ~20 operators these MLPs use. A 50-point polar sweep takes 10–30 ms on a laptop.

onnxruntime-web runs the same files, but it is about 10 MB of WebAssembly, which would dominate the page load. It
is therefore opt-in: add `&engine=ort` to the URL hash. It loads `onnxruntime-web@1.22.0` from jsDelivr.

The small geometry pieces (CST surfaces, NACA 4-digit fits, thickness and camber) are ported to JavaScript
(`js/cst.js`) instead of loading Pyodide. Pyodide would add about 10 MB for a few dozen lines of NumPy.

What is tested, in CI (`.github/workflows/playground.yml`):

- **Python** (`tests/test_playground.py`):
  - The committed ONNX files and a fresh export both reproduce the zoo's PyTorch models within 1e-5. The inputs
    are every held-out and independent-reference input of the zoo plus 500 random in-range points. The error is
    measured relative to the output scale; measured: 7e-7 airfoil, 1.6e-6 heat sink, 6e-7 rover.
  - The page's model cards, fixtures and gallery index are current.
- **JavaScript** (`node --test`):
  - The geometry, the network-input encoding and the heat-sink validity mask match the Python fixtures.
  - The ONNX interpreter matches onnxruntime within 1e-5.
- **Browser** (Playwright, Chromium):
  - Every mode loads.
  - The numbers on the page match the Python model for fixed inputs.
  - Leaving the range turns the verdict red, and dragging a handle changes the answer and the URL.
  - A shared link reproduces the state.
  - The trust panel shows the evidence.
  - The page works offline after one visit.
  - The mobile layout has no horizontal scroll.
  - onnxruntime-web gives the same numbers as the built-in interpreter.

## Run it locally

```bash
python apps/playground/build.py --serve          # http://localhost:8765/
```

The build uses only the standard library. It assembles `apps/playground/site/` together with three.js, which is
vendored in `pinneapple_tools`, and the light glTF files from `data/geometries/`. The output is self-contained and
uses only relative paths, so it works under any URL prefix.

The ONNX files, model cards and fixtures are generated from the zoo. Regenerate them when a zoo model changes:

```bash
python apps/playground/tools/build_assets.py     # needs torch, onnx, onnxruntime
cd apps/playground && npm ci && npm test         # JS unit tests + Playwright smoke test
```

## Deploying

The page is static, so there are two options, both prepared and neither switched on:

- **Docker + Caddy, next to the other apps.** `apps/playground/Dockerfile` builds the site and serves it with Caddy
  on port 8094. `apps/deploy` has the `playground` service and the `PLG_DOMAIN` site block.
- **GitHub Pages, under the docs site.** Set the repository variable `PLAYGROUND_PAGES` to `true`. The docs workflow
  then publishes the build at `https://pinneapple-labs.github.io/PINNeAPPle/playground/`.
