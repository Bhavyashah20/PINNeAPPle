# Multiscale ensembles: scales, filters and closures

An adaptive ensemble ([Adaptive ensembles of physics models](physics_ensembles.md)) picks a model from the errors
it made and the physics of the current case. In turbulence that is not enough. Every model answers at a *scale*:

- a surrogate is trained on fields at one resolution, with one implicit or explicit filter;
- an LES resolves the field down to its filter width and models the rest with a closure;
- a DNS resolves everything down to the dissipation scale.

A question asks for a scale too: the field filtered at some width, a spectrum up to some wavenumber, a time scale. A
model that does not resolve those scales cannot answer, however small its validation error. And when it can answer,
its output is filtered in a particular way, which the question may not share.

`pinneapple_physics.multiscale` deals with filtering and closure first and explicitly, then uses the scales to
choose models. It is plain numpy, except `multiscale.learned` (PyTorch).

| Module | Contents |
|---|---|
| `filters` | box, Gaussian and sharp spectral filters with exact transfer functions; restriction and prolongation between grids; post-filtering from one width to another and how much of the target filter it can reach; filters whose width varies in space, and their commutation error |
| `ns2d` | pseudo-spectral 2-D vorticity DNS and LES (ETDRK4, 2/3 dealiasing), energy spectrum, residual of the vorticity equation |
| `closures` | Smagorinsky, dynamic Smagorinsky, gradient (Clark) model; a-priori and a-posteriori tests; label-free closure health |
| `residuals` | residuals of the *filtered* equations at several widths, split into subfilter, commutation and model parts |
| `consistency` | cross-scale consistency: the fine prediction, filtered, against the coarse one |
| `scales` | `ScaleProfile`, `ScaleQuery`, `scale_compatibility`, `ScaleAwareRouter` with a fidelity ladder |
| `learned` | scale-conditioned FNO with a filtering-consistency loss; differentiable LES; closure trained a posteriori |
| `viz` | animation of a run: requested scale, chosen model, spectra, weights |

## Filters

A filter of width Δ acts on each Fourier mode through its transfer function. On a periodic grid it is applied
mode by mode, so the transfer function is exact to round-off (tested on single modes). Filters with a uniform width
commute with derivatives. In 2-D the filter is the tensor product of the 1-D filters.

| kind | transfer function Ĝ(k) | remarks |
|---|---|---|
| box | sin(kΔ/2) / (kΔ/2) | top-hat average; changes sign, with zeros at k = 2πn/Δ |
| Gaussian | exp(−k²Δ²/24) | same second moment as the box; positive; two Gaussians compose (Δ² = Δ₁² + Δ₂²) |
| sharp | 1 for \|k\| ≤ π/Δ, else 0 | spectral cut-off; idempotent |

```python
from pinneapple_physics.multiscale import Filter, apply_filter, refilter, realizability_mismatch, restrict

g = Filter("gaussian", 0.2)
w_bar = apply_filter(w, g)                              # periodic, last two axes
w_coarse = restrict(w_bar, 32)                          # Fourier restriction: integrals preserved to round-off
w_wide = refilter(w_bar, g, Filter("gaussian", 0.5))    # exact: a wider Gaussian from a narrower one
realizability_mismatch(Filter("sharp", 0.3), Filter("gaussian", 0.3), kmax=32)   # > 0: content beyond the cut-off is lost
```

`refilter(u, source, target)` applies `Ĝ_target / Ĝ_source` where that ratio is at most 2. Beyond that ratio the
scales are treated as lost: recovering them would only amplify the source's errors.
`realizability_mismatch` measures, weighted by a model spectrum, how much of the target filter cannot be reached.
The router uses it.

`filter_variable_width(u, kind, delta_field)` and `commutation_error(u, kind, delta_field, axis)` handle a width that
varies in space. The commutation error `∂(ū)/∂x − (∂u/∂x)‾` is zero to round-off for a uniform width and non-zero
otherwise (Ghosal & Moin 1995). A residual of the filtered equations picks it up from filtering alone, so it must not
be blamed on the model.

## Closures and how to test them

Filtering the momentum equations leaves the subfilter stress `τᵢⱼ = (uᵢuⱼ)‾ − ūᵢūⱼ`. In 2-D vorticity form the
filtered equation gets the tendency `−curl(div τ) = −div σ`, with the subfilter vorticity flux
`σⱼ = (uⱼω)‾ − ūⱼω̄`. Each closure below is a callable `closure(omega, solver) -> tendency` for
`VorticityNS2D(n, nu, closure=...)`, with a `stress(omega, solver)` method:

| closure | model | reference |
|---|---|---|
| `Smagorinsky(cs=0.17)` | τ = −2 (C_s Δ)² \|S̄\| S̄ | Smagorinsky 1963; Lilly 1967 |
| `DynamicSmagorinsky()` | coefficient from the Germano identity between grid and test filter, least squares, domain average, clipped at 0 | Germano et al. 1991; Lilly 1992 |
| `GradientModel()` | τ = Δ²/12 ∂ₖūᵢ ∂ₖūⱼ | Clark, Ferziger & Reynolds 1979 |
| `learned.LearnedClosure` | periodic CNN that returns the flux σ, so the tendency −div σ conserves circulation | trained a posteriori, see below |

Tests:

- `apriori_test(w_fine, filter, closure, n_coarse)` filters a resolved field and compares the closure, computed
  from the filtered field, with the exact subfilter terms. It reports the correlation of the stress components and
  of the tendency, the mean subfilter energy transfer `Π = −τᵢⱼS̄ᵢⱼ`, and the backscatter fraction (Π < 0), which
  no eddy viscosity can represent.
- `aposteriori_test(les, w0_bar, ref_bar, times)` runs the LES from the filtered initial field. It returns the
  energy-spectrum error, the field error, the resolved-energy dissipation rate against the reference, the long-run
  stability and the enstrophy growth.
- `closure_health(w_les, w0_bar, kc, closure=..., solver=...)` needs no reference, so it can run in operation:
  - the field must stay finite;
  - resolved enstrophy must not grow in unforced 2-D flow. Advection conserves enstrophy, and viscosity and the
    forward enstrophy cascade only remove it;
  - the spectrum near the cut-off must not pile up or collapse against the power law of the resolved range;
  - the Germano-identity error `germano_error` must stay below 1.05. This error is the resolved stress between the
    grid filter and a test filter, as the closure predicts it, compared through the tendency it produces: 1 means
    no better than no model, above 1 means worse than no model.

A 2-D result to keep in mind: for Gaussian and box filters the leading-order subfilter stress is
`τ ≈ Δ²/12 ω (−S̄₁₂, S̄₁₁)`, the strain rotated by 45° times the vorticity. It is orthogonal to S̄, so an eddy-viscosity
stress has near-zero correlation with the true stress (measured −0.03 to 0.00 below), the gradient model has 0.97 to
0.99, and the gradient model's mean energy transfer is zero. The vorticity tendency, which is what the resolved
equation sees, correlates 0.1 to 0.4 for all of them. Stress correlations of 0.3 to 0.4, quoted for Smagorinsky in
3-D channel flow, do not carry over to 2-D.

## Multiscale residuals

```python
from pinneapple_physics.multiscale import ns2d_multiscale_residuals, Filter

r = ns2d_multiscale_residuals(w_t0, w_t1, dt, nu, [Filter("gaussian", d) for d in (0.1, 0.2, 0.4)],
                              subfilter="exact")      # or "dynamic", a closure, or None
print(r.to_markdown())                                # raw | subfilter | commutation | model, per width
```

The residual of an exact field, filtered, is the subfilter term, not a model error. The table splits it:
`raw = subfilter + commutation + model`. For a DNS pair (64², ν = 10⁻³), the model part is about 1 % of the raw
residual at the coarser widths. On an exact steady Taylor-Green vortex every part is zero at every width. With white
noise added, the residual follows the filter's transfer function (`sqrt(mean Ĝ²)`), as the tests check.
`multiscale_residuals(state, residual_fn, filters, subfilter_fn=..., commutation_fn=...)` is the generic version for
any equation.

## Cross-scale consistency

```python
from pinneapple_physics.multiscale import cross_scale_consistency

rep = cross_scale_consistency(fine, coarse, coarse_filter=Filter("sharp", L / 16), tol=0.05, gci=None)
rep.passed, rep.field_error, rep.spectrum_error, rep.failed      # e.g. ['circulation (mean)']
```

The fine prediction is refiltered and restricted to the coarse grid. It is then compared with the coarse prediction
on the field, on the spectrum over the shared wavenumbers, and on integrals (circulation, energy, enstrophy, or any
`{name: fn}`). Tolerances can be scaled by a known discretisation error (`gci`). Restricting an exact field keeps
its integrals to round-off. A 32² solver started from the same state as a 64² solver passes. The same 32² solver with
a biased source term fails, and the report names the broken integral, `circulation (mean)`. In the ensemble, the
scale-conditioned FNO checks itself this way: its prediction at Δ, filtered to 2Δ, against its prediction at 2Δ.

## Scale-aware selection and adaptive fidelity

```python
from pinneapple_physics.ensemble import PhysicsEnsemble
from pinneapple_physics.multiscale import ScaleAwareRouter, ScaleProfile, ScaleQuery, Filter

profiles = [
    ScaleProfile("fno", Filter("gaussian", 2 * h), h, re_range=(150, 300), fidelity=0),
    ScaleProfile("les_dynamic", Filter("sharp", pi / 16), L / 48, closure="dynamic Smagorinsky", fidelity=1),
    ScaleProfile("dns", Filter("sharp", 0.0), L / 128, re_range=(0, 6000), fidelity=2),
]
router = ScaleAwareRouter(profiles, query_fn=lambda case: ScaleQuery(case["filter"], re=case["re"]))
ens = PhysicsEnsemble(experts, mode="select", router=router,
                      prediction_checks={"les_dynamic": lambda pred, case: les_health(pred, case)})
router.route(ScaleQuery(Filter("sharp", h / 4), re=3000)).reasons[0]   # 'not covered: ... a resolved solver ...'
```

For each case:

1. Models whose resolved width (filter width or grid spacing) is larger than the requested width sleep. So do
   models whose time step is longer than the requested time scale.
2. Every other model gets a compatibility of `exp(−(mismatch/0.25)²)` times a Reynolds-range score, where mismatch
   is the realizability mismatch of the requested filter from the model's own output filter. This compatibility is a
   prior on the learned weights.
3. Fidelity ladder: only the lowest rung with a compatible model (≥ 0.3) is awake. A question about small scales
   goes straight to the solver; a question about large scales stays on the surrogates.
4. Prediction checks: the label-free closure health of each LES and the consistency check of the scale-conditioned
   FNO. A failing model is left out of the case, and when every model on the rung fails the ensemble moves up a rung.
5. The residual look-ahead uses the residual of the filtered equations with the subfilter term removed, at the
   requested width and at twice it.

## Models that connect scales

- `learned.ScaleConditionedFNO2d(conditioned=True)` takes Δ as input: FiLM modulation of every Fourier layer by
  log Δ. It is trained on one dataset filtered at several widths with `fit_scale_conditioned`. The
  `filtering_consistency_loss` makes its prediction at Δ₁, filtered to Δ₂, equal its prediction at Δ₂.
  `conditioned=False` gives the scale-blind network for comparison.
- `learned.TorchVorticityLES2D` + `LearnedClosure` + `train_closure_aposteriori`: the closure is trained *through
  the solver*. The LES is rolled out 60 steps, the loss compares its trajectory and spectra with the filtered DNS,
  and the gradient goes through every step. `as_numpy_closure` plugs the result into `VorticityNS2D`.

## Experiment: decaying 2-D turbulence with shifting Reynolds number and requested scale

RESULTS_PLACEHOLDER
