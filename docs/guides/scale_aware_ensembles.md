# Choose a model by the scale of the question

This guide shows how to make an adaptive ensemble respect scales: declare what each model resolves, ask
each question at a scale, check closures, and let the ensemble escalate to a finer model when needed. The concepts
are in [Multiscale ensembles: scales, filters and closures](../core_concepts/multiscale_ensembles.md).

## 1. Declare the scales of every model

```python
from math import pi
from pinneapple_physics.multiscale import Filter, ScaleProfile

L, h = 2 * pi, 2 * pi / 64
profiles = [
    ScaleProfile("fno", Filter("gaussian", 2 * h), h, re_range=(150, 300), fidelity=0),         # surrogate
    ScaleProfile("les", Filter("sharp", pi / 16), L / 48, closure="dynamic Smagorinsky", fidelity=1),
    ScaleProfile("dns", Filter("sharp", 0.0), L / 128, re_range=(0, 6000), fidelity=2),          # resolved
]
```

Declare the filter the model's output actually carries:

- a surrogate trained on Gaussian-filtered data carries that Gaussian;
- a dealiased spectral LES on n points carries a sharp cut-off at k = n/3;
- a resolved solver carries no filter.

## 2. Ask each question at a scale

```python
from pinneapple_physics.multiscale import ScaleQuery, ScaleAwareRouter

router = ScaleAwareRouter(profiles, query_fn=lambda case: ScaleQuery(case["filter"], re=case["re"]))
print(router.route(ScaleQuery(Filter("sharp", h), re=3000)))     # only the DNS resolves this
print(router.route(ScaleQuery(Filter("gaussian", 8 * h), re=200)))  # the surrogate is enough
```

## 3. Bring every output to the requested filter

Use `refilter(output, profile.filter, query.filter)`, not `apply_filter(output, query.filter)`. Filtering an
already-filtered output again gives a wider filter than the one asked for. `refilter` applies only the part of the
requested filter that the model's own filter has not applied yet.

## 4. Check closures without a reference

```python
from pinneapple_physics.multiscale import closure_health, DynamicSmagorinsky, VorticityNS2D

h = closure_health(w_les, w0_les, kc=16, closure=DynamicSmagorinsky(), solver=VorticityNS2D(48, nu))
h.ok, h.reasons      # enstrophy growth, spectral pile-up or collapse, Germano-identity error
```

## 5. Put it together

```python
from pinneapple_physics.ensemble import PhysicsEnsemble

ens = PhysicsEnsemble(experts, mode="select", router=router,
                      prediction_checks={"les": lambda pred, case: les_health(case)},
                      error_tolerance=0.25, error_context=lambda case: case["filter"])
out = ens.predict(case)
out["active"], out["failed_checks"], out["escalated_by_checks"]
```

For every case the ensemble:

- uses the lowest fidelity rung that resolves the requested scale;
- skips models whose checks fail;
- moves up a rung when the models on the current rung last missed the tolerance at this scale.

`examples/physics_ensemble/multiscale_turbulence.py` runs the whole comparison on 2-D turbulence.
