# OpenFOAM parameter campaign

One base case, six runs that change the jet speed and elevation, split between two machines.

```bash
python examples/openfoam_campaign/make_base.py                 # 2-D trough with a bath and an inclined jet
M="python -m pinneapple_simulation.external_solvers.openfoam.campaign examples/openfoam_campaign/plan.json"
$M --status                                                    # done / running / pending
$M --commands 2                                                # the line to start on each machine
$M --worker 1                                                  # or: --shard 1/2; each run writes _out/results/<id>.json
```

Each result holds:
- the parameters;
- the solver health: Courant number, phase-fraction bounds, phase volume, and whether the run ended;
- the profiles of k and the liquid fraction on the `mid_depth` line, averaged from 2 s to 10 s.

See `docs/core_concepts/openfoam_campaigns.md`.
