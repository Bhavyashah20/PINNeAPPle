# A latent surrogate that says when not to trust it

A latent reduced-order model encodes the solver's fields in a few coordinates (POD), regresses those coordinates on
the design parameters, and decodes. It answers in a fraction of a millisecond, and answers just as fast where it is
wrong. `TrustedLatentSurrogate` (in `pinneapple_neural.architectures.rom`) adds three checks to every answer,
calibrated on designs the latent model never saw, and hands the result to the library's `TrustReport`:

| check | what it measures | calibrated how | in the report |
|---|---|---|---|
| error bound | latent predictive spread (Gaussian-process std, or a bootstrap ensemble of latent regressors) scaled to the relative L2 error of the field | normalised split-conformal: covers the true error with probability `1 - alpha` for queries drawn like the calibration set | not critical: above the tolerance → REVIEW |
| design envelope | mean distance to the `k` nearest training designs | conformal p-value against the calibration designs | critical: p < `alpha_envelope` → REJECT |
| physics residual | the discrete governing equations evaluated on the decoded field (a sparse mat-vec, no solve) | `1 - alpha_residual` conformal quantile of calibration residuals | critical → REJECT |

The envelope check exists because conformal coverage only holds while queries look like the calibration data; out
there the spread itself can be overconfident. The physics residual is independent of the data: it sees errors that
every member of an ensemble agrees on.

## Usage

```python
from pinneapple_neural.architectures.rom import ParametricPOD, TrustedLatentSurrogate

sur = TrustedLatentSurrogate(
    ParametricPOD(r=64, regressor="gpr", normalize=True),   # latent model; "rbf" uses a bootstrap ensemble
    alpha=0.1,                                              # 90 % error bound
    residual=lambda p, u: rel_residual(p, u),               # optional: ||L(p) u - f(p)|| / ||f(p)||
).fit(P_train, X_train, P_cal, X_cal)                       # calibration designs: never used for fitting

a = sur.assess(P_new, tol=0.02)        # X, error_bound, envelope_p, residual, decision per query
report = sur.trust_report(P_new[0], tol=0.02)
report.decisao, report.to_markdown()   # APPROVED / REVIEW / REJECT with the reasons
```

`assess` returns the same decision as the report: REJECT on an envelope or residual failure, REVIEW when the error
bound exceeds the tolerance, APPROVED otherwise. A REVIEW or REJECT means "run the solver for this one", which is
what a design loop or a digital twin should do with the answer.

## Experiment: every answer against the full solver

`python examples/latent/trusted_latent_surrogate.py --seed 0` (also seeds 1 and 2). Full model: steady 2-D
advection-diffusion-reaction of a Gaussian source on a 96 × 96 grid (upwind finite differences, sparse direct solve),
four parameters (wind speed, direction, decay rate, source position). 150 training and 100 calibration designs from
an envelope with a hole (a band of wind speeds and directions never simulated). Three test groups of 150: inside the
envelope, inside the hole (inside the parameter box, so a box check passes them), and 5–30 % beyond the box in one
parameter. Tolerance: 2 % relative L2 error. Mean of three seeds, [min, max] where it varies:

| | envelope | hole | beyond the box |
|---|---|---|---|
| median / p95 relative error | 0.84 % / 3.8 % | 0.61 % / 1.4 % | 4.9 % / 17 % |
| nearest training design (baseline), median | 17 % | 20 % | 31 % |
| fraction with error > 2 % | 15 % | 2 % | 90 % |
| coverage of the 90 % error bound | 0.92 [0.87, 0.96] | 0.99 | 0.65 |
| APPROVED / REVIEW / REJECT | 58 / 39 / 3 % | 71 / 28 / 0 % | 6 / 18 / 76 % |
| approved with error > 2 % | 2.4 % | 0.4 % | 1.6 % |

Detecting a wrong answer (error > 2 %) over all 450 queries, ROC AUC: error bound **0.980**, physics residual
**0.976**, envelope p-value 0.943, box check 0.891. Spearman correlation with the true error: bound 0.95, residual
0.93.

As a gate, over the three seeds: the trust decision approved 607 answers, of which 20 (3.3 %) were wrong, the
largest approved error being 1.2 %, 3.5 % and 5.3 % per seed. A box check approved the 900 in-box answers, of which
79 (8.8 %) were wrong, with largest approved errors of 7.4 % to 14.5 %.

Speed: full solve 45.6 ms; surrogate 0.71 ms per query in a batch (65 times faster), 2.5 ms per query with the three
checks (18 times faster). A single query costs 11 ms (only 4 times faster) because scikit-learn's GP prediction has a
fixed per-call overhead for each latent coordinate; batch the queries.

What the numbers say:

* The latent model is genuinely useful here (0.8 % median error, 20 times better than the nearest design), and
  most of its failures are recognisable before anyone looks at the reference: the bound and the residual each rank
  wrong answers with AUC ≈ 0.98.
* The hole turned out to be interpolable (0.6 % error), and the gate approved most of it: the envelope check flags
  distance, but the decision only rejects when the p-value is extreme, so the gate is not needlessly conservative.
* Beyond the box the coverage of the bound drops to 0.65, which is why the envelope check is critical: it rejects
  three quarters of those queries.
* The error bound is marginal over the calibration distribution, not conditional per query. With 100 calibration
  designs the coverage fluctuates by a few percent between seeds, and the approval rate inside the envelope varied
  from 33 % to 71 %: more calibration designs make both steadier.

## References

* J. Lei, M. G'Sell, A. Rinaldo, R. Tibshirani, L. Wasserman, "Distribution-free predictive inference for
  regression", JASA 113 (2018).
* A. N. Angelopoulos, S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty
  quantification", arXiv:2107.07511.
* V. Vovk, A. Gammerman, G. Shafer, Algorithmic Learning in a Random World, Springer (2005).
* B. Peherstorfer, K. Willcox, "Data-driven operator inference for nonintrusive projection-based model reduction",
  CMAME 306 (2016).
* J. S. Hesthaven, G. Rozza, B. Stamm, Certified Reduced Basis Methods for Parametrized PDEs, Springer (2016)
  (residual-based a-posteriori error estimators).
