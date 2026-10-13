## Summary

Integrates **REA** (Reverse Engineer Anything, MIT) as the third source of the research pipeline, with a license gate and a provenance ledger, and adds components extracted from 50+ reference projects. Every new component is tested against an independent reference — closed form, exact solution, identity or an independent implementation (rule D1).

### Integration
- `pinneapple_tools/rea_bridge/`
  - CLI client: `PINNEAPPLE_REA_BIN` > `rea` > `npx rea-agents@6.4.0`.
  - JS Evidence → feature map (functions, call graph, exact lines), plus `locate(keywords)`.
  - ELF layout → exported symbols.
  - License gate (`adapt` / `clean_room` / `reference_only`).
  - `HarvestLedger`, which refuses adapted code from copyleft projects.
- `PhysicsToolRegistry.register_all()` → `reference_license_policy`, `rea_analyze_javascript`, `rea_locate_feature`.
- `.mcp.json`: REA MCP server for coding agents.
- `docs/dev/REA_INTEGRATION.md`, `docs/dev/ANALISE_FEATURES_COMERCIAIS.md` (feature-gap analysis vs Ansys, Siemens, Luminary, PhysicsX, Neural Concept, Godela, Omniverse and 20+ CFD codes).
- `docs/dev/rea_harvest/ledger.json` (53 projects).
- `docs/dev/issues/` (35 issue drafts) + `scripts/publish_issues.sh`.

### New components
| File | Origin (license, method) | Validation |
|---|---|---|
| `numerical_solvers/etdrk4.py` (S27) | Cox–Matthews / Kassam–Trefethen (clean-room vs Dedalus GPL); blow-up guard from VisualPDE via REA | exact exp(Lt), measured 4th order, linear KdV, Burgers vs Cole–Hopf (< 1e-9), **matches Exponax to 4.5e-14 on chaotic KS** |
| `numerical_solvers/stable_fluids.py` (S28) | WebGL-Fluid-Simulation via REA (MIT) | div = 0 to round-off, Taylor–Green 1st order, exact translation |
| `analysis/validation/field_metrics.py` | PDEBench (MIT, fRMSE fixed), The Well (BSD-2: VRMSE, Pearson) | Parseval 1D/2D/3D, single-mode bands, closed-form VRMSE |
| `analysis/uncertainty/posterior_metrics.py` (CRPS) | Gneiting et al.; Earth-2 Studio usage | integral definition, pairwise identity, deterministic limit |
| `design/geometry/ops/curvature.py` | PhysicsNeMo (Apache, independent NumPy) | exact Gauss–Bonnet, sphere/cylinder/plane |
| `external_solvers/modelica/fmu_inspect.py` | FMI headers (BSD-2), REA on Modelica Reference-FMUs | parser = `nm -D` = REA/pwntools on 15 FMUs |
| `pinneapple_twin3d` UsdPhysics export | OpenUSD schemas | OpenUSD's own physics parser recognises scene, bodies, colliders, material |

### Fixes relative to the references
- WebGL-Fluid: a 5-point Jacobi solve with central div/grad is inconsistent, so the divergence never reaches zero. Replaced by a consistent spectral projection.
- PDEBench: the 2-D/3-D fRMSE drops negative frequencies. Now the full spectrum is binned.

### Tests
62 new tests pass (several also end to end with the real REA, Exponax and usd-core). Integration tests skip cleanly when REA, Exponax, usd-core or the built Reference-FMUs are absent. **The full suite was not run** (no torch in the sandbox), so run `pytest` before merging. In `method_status.json` only S27/S28 were inserted: a full regeneration would drop the PINNeAPPle-CFD references, which aren't available there.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01TTWeeDL5NNNcvZGQPVBuW6
