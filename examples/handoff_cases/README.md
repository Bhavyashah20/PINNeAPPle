# Engineering hand-off cases

Two self-contained case packages for testing an engineering-simulation agent (or an engineer) from a cold start.
Each folder holds the task (`brief.md`), the geometry, the simulation set-up and scripts, results that were really
computed, and verified references; `<case>.zip` is the same folder as one archive.

| Case | Physics | Task | Zip |
|---|---|---|---|
| [`airfoil_ld/`](airfoil_ld/README.md) | 2-D airfoil, steady RANS (OpenFOAM simpleFoam, k-omega SST), Re 4 x 10^6 | maximise L/D at alpha 4 deg with thickness and pitching-moment limits; 6 CST design variables | `airfoil_ld/airfoil_ld.zip` |
| [`free_surface_trough/`](free_surface_trough/README.md) | water jet into an open trough, two-phase VOF (OpenFOAM interFoam, k-omega SST) | choose jet speed and angle to limit floor shear, impingement position and splash | `free_surface_trough/free_surface_trough.zip` |

Rebuild the zips after changing a package: `python examples/handoff_cases/make_zips.py`.

Software: OpenFOAM v1912 (Ubuntu 24.04 package `openfoam`), Python 3 with numpy / scipy / matplotlib, this
repository (`pip install -e .`), XFOIL 6.99 (airfoil reference), cadquery 2.8 (STEP / STL).
