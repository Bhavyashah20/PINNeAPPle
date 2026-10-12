# Free-surface flow (interFoam, VOF)

`01_runner_slope_study.py`: a tapping jet falling into a melt runner held by a weir; the floor slope (+2°, 0°,
-2°) changes the depth of metal under the jet. 2-D sections with `pp.cfd.FreeSurfaceFlow`, k sampled 5 cm above
the floor while the solver runs and averaged in time. Needs OpenFOAM. Results: `docs/core_concepts/free_surface.md`.
