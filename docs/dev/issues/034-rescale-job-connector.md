---
title: "Rescale REST connector: run a DOE campaign on Rescale and pull results"
labels: [enhancement, orchestration, hpc]
---

**Who has it:** Rescale API v2 (jobs, submit, statuses, files), AI Physics dataset format (PyVista-compatible).

**What PINNeAPPle already has:** SLURM script generation, Prefect flows, file-lock sharded OpenFOAM campaigns; no managed cloud backend.

**Proposal:** A `campaign` backend that submits each DOE point as a Rescale job, polls, downloads outputs and registers them in `pinneapple_registry`.

**Acceptance (rule D1 — independent reference):** Contract tests against recorded API responses; resumable after interruption (no duplicate submissions).

**Origin:** mapping 2026-10-12: Rescale
