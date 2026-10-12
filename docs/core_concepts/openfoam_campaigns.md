# OpenFOAM parameter campaigns on several machines

`pinneapple_simulation.external_solvers.openfoam.campaign` runs one verified OpenFOAM case many times with
different inputs. The runs can be spread over workstations, cluster nodes or cloud sessions. A study that takes
days on one machine finishes in the time of a few runs, and its results come back in a form a surrogate model can
be trained on.

## What a run does

1. **Copy the base case.** The base is a case directory or a `.tar.gz`. A run can swap in another mesh, such as
   a geometry variant. After a swap, fields sized for the old mesh are reset, and `setFields` or the solver
   rebuilds them.
2. **Apply the parameters.**
   - Dictionary edits by path, such as `boundaryField/inlet/value` in `0/U`. Their values are templates filled
     with the parameters and with derived values (`"Uy": "U*cos(radians(theta))"`).
   - An optional patch rebuilt to an area (`area_patch`). It takes the boundary faces of a plane that are nearest
     to a centre point until they add up to the area, which makes an inlet diameter an input.
   - An optional Python hook (`"setup": "module:function"`) for anything else.
3. **Run.** The pre-processing commands run (`checkMesh`, `setFields`, ...). The case is decomposed with scotch
   when the OpenFOAM installation has it, otherwise `simple` along the longest side. The solver then runs with
   `mpirun`.
4. **Sample lines while the solver runs** (`LiveLineSampler`). The sampling reads the written fields, so it does
   not depend on the function objects of the OpenFOAM version. Sampled times can be deleted, keeping one every
   `keep_every` seconds, which bounds the disk use of long transients.
5. **Write `<results>/<id>.json`** with:
   - the parameters and the values derived from them;
   - the solver health read from the log: last time, Courant number, phase-fraction bounds, phase volume
     history, whether the run reached `End`, and the first fatal error;
   - the time-mean profiles on the lines;
   - the host and the wall time.

## Splitting the work

Every machine runs the same command on the same plan file and takes its share:

| Option | Share |
|---|---|
| `--worker W` | the runs tagged `"worker": "W"` in the plan |
| `--shard i/n` | every n-th pending run, starting at the i-th |
| run ids | only those runs |

- **Runs already done** (their result file exists) are skipped, so a stopped worker restarts where it was.
- **Claims:** a run is claimed with a lock file next to the results, so machines that share a disk never run the
  same case. `--force` clears a lock left by a worker that died.
- **Machines that share nothing**, such as cloud sessions, use `--publish-git`. Each result is committed and
  pushed to the current branch as soon as it is written, with a rebase on what other workers pushed meanwhile.
  The author can be set with `--git-name` and `--git-email`.

```bash
M="python -m pinneapple_simulation.external_solvers.openfoam.campaign plan.json"
$M --commands 5            # prints the command to start on each of 5 machines
$M --shard 3/5 --procs 4 --publish-git
$M --status                # done / running / pending, on any machine
$M T07 --stage-only cases  # write a case without running it, to inspect it
```

## The plan

```json
{"name": "jet sweep", "base": "base_case.tar.gz", "solver": "interFoam", "procs": 4, "end_time": 30,
 "pre": ["checkMesh", "setFields"],
 "derived": {"Uy": "U*cos(radians(theta))", "Uz": "U*sin(radians(theta))"},
 "edits": [{"file": "0/U", "key": "boundaryField/inlet/value", "value": "uniform (0 {Uy} {Uz})"}],
 "area_patch": {"patch": "inlet", "area": "pi*(d/1000)**2/4", "centre": [0, -7.4, -1.3], "normal": [0, -1, 0],
                "from": ["inlet", "wall"], "rest": "wall"},
 "sample": {"fields": ["k", "alpha.water"], "keep_every": 10, "average_from": 0,
            "lines": [{"name": "L1", "start": [0, -7.4, -2.2], "end": [0, 15.3, -2.2], "n": 101}]},
 "results": "results",
 "runs": [{"id": "T01", "params": {"U": 8.6, "theta": 5.0, "d": 109}, "worker": "1"},
          {"id": "D1", "params": {"U": 10, "theta": 10, "d": 85}, "mesh": "mesh_variant.tar.gz", "group": "deflector"}]}
```

- Paths are relative to the plan file.
- Expressions accept the parameters, earlier derived values, numbers, arithmetic and the functions of `math`.
  Nothing else is evaluated, so a plan received from someone else cannot run code. The `setup` hook is the
  exception: it imports a Python module, so use it only from plans you trust.
- In edit values, `{Uy}` writes the number in full, `{Uy:.4f}` formats it, and a list is written as an OpenFOAM
  vector.
- `area_patch` exposes the area it actually reached as `patch_area`, so an edit can use it, for example to set a
  flow rate.

From Python:

```python
from pinneapple_simulation.external_solvers.openfoam.campaign import Campaign, GitPublisher
c = Campaign.from_json("plan.json", procs=8)
c.run(shard=(1, 3), publish=GitPublisher("me", "me@example.org"))
```

## Requirements and limits

- OpenFOAM is sourced from `FOAM_BASHRC` (default `/usr/share/openfoam/etc/bashrc`), with MPI for `procs > 1`.
  Tested with OpenFOAM v1912.
- Meshes must be in ASCII format: `area_patch`, the decomposition box and the line sampler read them directly. Run
  `foamFormatConvert` once on a binary base case.
- Lines are sampled with the value of the nearest cell, without interpolation. Points with no cell nearby, such as
  points inside an obstacle, are dropped.

Example: `examples/openfoam_campaign/` (a jet into a bath in a 2-D trough, six runs on two workers). Tests:
`tests/test_openfoam_campaign.py`. It covers:
- dictionary edits and safe expressions;
- shards, workers, claims and the plan file;
- log parsing and field reset after a mesh swap;
- a patch cut to half its area;
- two short parallel interFoam runs.
