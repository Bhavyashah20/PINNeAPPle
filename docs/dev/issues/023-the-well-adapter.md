---
title: "The Well adapter: WellDataset → PINNeAPPle dense volumes + leaderboard-comparable metrics"
labels: [enhancement, data, benchmark]
---

**Who has it:** PolymathicAI The Well (BSD-2-Clause): 16 physics datasets (fluids, MHD, supernovae, acoustics, biology), `WellDataset` loader, VRMSE / binned spectral MSE / Wasserstein-1 / windowed DTW metrics.

**What PINNeAPPle already has:** Planned in ROADMAP §4 (`pinneapple_data.adapters.well_adapter`), not implemented. `field_metrics` now reports `vrmse` and `pearson` with The Well's definitions.

**Proposal:** Adapter to the `load_dense_volumes` format; streaming from the HF Hub; binned spectral MSE with The Well's default log-spaced bins; register the datasets in `pinneapple_catalog.resources` with their license.

**Acceptance (rule D1 — independent reference):** Round trip on the `MHD_64` (or smallest) split: shapes/channels/time match the HDF5 metadata; VRMSE of the persistence baseline (u(t+1) = u(t)) reproduced within 5% of the value in The Well paper.

**Origin:** mapping 2026-10-12: The Well
