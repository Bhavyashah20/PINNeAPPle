---
title: "In-browser PDE preview (SymbolicPDE → GLSL), VisualPDE-style"
labels: [enhancement, frontend]
---

**Who has it:** VisualPDE (MIT) — located with REA: `setRDEquations`, `url_codec`, auto steps-per-frame.

**What PINNeAPPle already has:** `pinneapple_app/frontend` (React/Vite); `SymbolicPDE`.

**Proposal:** Emit GLSL from a 2-D reaction–diffusion `ProblemSpec`, run it in the frontend, keep the state shareable in the URL.

**Acceptance (rule D1 — independent reference):** Gray–Scott: steady pattern matches the NumPy solver (correlation > 0.99) on 5 presets.

**Origin:** ledger: VisualPDE
