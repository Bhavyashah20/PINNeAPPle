# From a problem description to checked CAD, fit-for-purpose meshes and one report

`pinneapple_design.fitmesh` turns a problem written in words (English or Portuguese) into reference geometries,
checked CAD variants, meshes sized for what the result will be used for, and a single report that ties every
decision to its evidence. The goal is the mesh and CAD quality the problem needs: not so fine that it wastes days,
not so coarse that nothing in the result can be trusted.

```python
from pinneapple_design.fitmesh import FitMeshSession

s = FitMeshSession("Erosion and pressure drop in a 3 inch pneumatic pipe bend, air at 20 m/s", "study/")
s.step()                         # the open questions and the geometry options, with references
s.answer(accuracy="design")      # screening / design / verification
s.choose_geometry("pipe_bend_90")
s.choose_parameters(n=4)         # the suggested parameters and ranges, or vary={"bend_ratio": [1, 3]}
s.generate_cad()                 # STEP + STL per variant, each with its checks
s.plan_meshes()                  # mesh plan per variant, with the rules and sources used
s.generate_meshes(procs=4)       # meshes + quality (and the coarser family for design / verification)
s.add_results("pressure drop", [f_fine, f_medium, f_coarse])   # optional: grid-convergence index
s.report()                       # study/report.html and study/report.json
```

`FitMeshSession.run(text, out_dir, answers=..., geometry=..., vary=...)` does it all in one call, and
`pinneapple_design.fitmesh.api.create_app()` exposes the same steps over HTTP (one call per step, files and report
downloadable).

## The steps

| Step | What happens | What is checked |
|---|---|---|
| Problem | physics, fluid, speed, size, Reynolds number, purpose, quantities, material and load are read from the text; anything missing becomes a question; every fact keeps its source (text, answer, default, model) | unanswered facts are listed in the report |
| Geometry | templates ranked by physics and keywords: NACA 4-digit wing, Ahmed body, cylinder, sphere, 90° pipe bend, backward-facing step, plate with a hole, cantilever beam, plate-fin heat sink | each template cites the published references it can be compared with |
| Parameters | high-influence parameters proposed for variation over their usual range (sizes within a factor 2 of the base), the rest kept; dependent sizes follow (development lengths in diameters); Latin hypercube, reference design first | validity limits of every parameter |
| CAD | STEP and STL per variant | valid solid (OpenCascade), volume against the exact formula, watertight STL, STL/solid volume within the tessellation tolerance for the purpose, dimensions against each reference object (same / inside the studied range / extrapolated) |
| Mesh plan | method, sizes, prism layers, domain, cell and cost estimate, and the mesh family | every rule is written into the rationale with its source |
| Meshes | Gmsh (solids: quadratic tetrahedra, `.msh` and CalculiX `.inp`; 2-D sections with a boundary layer), snappyHexMesh (3-D external with far field and ground, internal with inlet and outlet from the CAD faces) | `pinneapple_data.cae` quality (non-orthogonality, skewness, aspect ratio, scaled Jacobian, disconnected regions), checkMesh, prism-layer coverage |
| Report | one JSON and one HTML page, a verdict per stage | grid-convergence index when the results of the three meshes are given |

## The rules behind the mesh plan

| Purpose | Surface cells | Wall treatment | Prism growth | Meshes |
|---|---|---|---|---|
| screening (~10 %) | L/40, 4 across the smallest feature | wall functions, y+ ≈ 50 (30-300) | 1.3 | one |
| design (a few %) | L/80, 8 across the smallest feature | y+ ≈ 1 where separation or wall quantities matter (k-ω SST without wall functions) | 1.2 | the mesh and one 1.5× coarser |
| verification | L/120, 12 across the smallest feature | y+ ≈ 1 | 1.15 | three, ratio 1.5, for the GCI |

- First-cell height from the skin friction at the reference length: Cf = 0.026 Re^-1/7 (White) turbulent,
  0.664 Re^-1/2 laminar, and the pipe friction factor 0.316 Re^-1/4 for internal flows. y+ between 5 and 30 (the
  buffer layer) is never used.
- External domains: 5 L upstream, 10-20 L downstream, 5 body sizes to the sides; the blockage is reported.
  Extruded sections (wing, cylinder, step) are meshed as 2-D unless a 3-D wake is asked for.
- Solids: quadratic tetrahedra; 2-4 elements through the thinnest wall; 16-48 elements around holes, where the
  stress peaks; the closed-form check (Kt from Peterson's, beam formulas) is printed next to the plan.
- Grid convergence: observed order, Richardson extrapolation and GCI with a safety factor of 1.25 for three grids
  (Roache; Celik et al., J. Fluids Eng. 130:078001, 2008). Oscillatory convergence is reported as a range, not
  forced through the formula, and the asymptotic range is checked.

## Language models, local or hosted

```python
from pinneapple_design.fitmesh import FitMeshSession, LLMAssist
llm = LLMAssist(provider="ollama", model="qwen2.5:7b")          # or "anthropic" / "openai"
s = FitMeshSession("a bracket that holds a 2 kg motor on a wall", "study/", llm=llm)
```

The model can read the problem, rank the geometries, propose parameters and ranges, write a CadQuery part when no
template fits, and review the mesh plan. It proposes; PINNeAPPle checks:

- unknown template or parameter names are dropped and ranges are clipped to the validity limits (the report says
  so);
- model-written CadQuery is parsed first (imports limited to cadquery, math and numpy; no file, system or dunder
  access), then runs in a separate isolated Python process with a timeout, and the part goes through the same CAD
  checks, marked "no published reference";
- the mesh review may change only the y+ target, the growth ratio and the layer count, inside the rule limits;
  refused changes are written into the rationale.

Calls can be logged with `pinneapple_llm.ConversationStore`. Without a model everything runs on the rules.

## How this compares with the tools in use today

Commercial meshers (Fluent watertight and fault-tolerant workflows, STAR-CCM+ automated meshing, Fidelity
Pointwise, ANSA batch meshing) and open ones (Gmsh, snappyHexMesh, cfMesh, Netgen) size meshes from curvature,
proximity and gaps, but the first-layer height still comes from a y+ target the user supplies, and none checks the
resolution against the quantity of interest before solving. Design-space tools (CAESES, optiSLang, HEEDS, Dakota,
SALib) sample the ranges the engineer sets; none says where a range came from. Text-to-CAD (Zoo, Text2CAD,
CAD-Coder, CADCodeVerify) now produces running CadQuery for simple parts, but dimensional accuracy is weak and
failures can be silent. LLM simulation agents (MetaOpenFOAM, Foam-Agent, ChatCFD) report how often a case runs,
not whether it is fit for purpose, and none runs a grid-convergence study.

This module's aim is the missing piece: references and ranges with their source, CAD checked against exact
formulas and published objects, a mesh sized from the physics and the purpose, and one report with a verdict per
stage. Template parts are not free-form CAD and the rules are not adjoint-based adaptation; when the stakes are
high, the verification purpose and the GCI are what make a number quotable.

## Limits

- Nine templates today; any other part goes through the language model (no published reference) or a STEP file.
- snappyHexMesh prism layers collapse at sharp edges and thin gaps; the coverage is measured and reported, and a
  low coverage is a warning, not hidden.
- Cell, cost and degree-of-freedom estimates are order-of-magnitude.
- Gmsh (GPL) and OpenFOAM (GPL) are optional external tools; install them separately (`pip install gmsh`).
