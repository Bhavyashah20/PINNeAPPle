"""Fit-for-purpose geometry and mesh: from a problem description to checked CAD, planned meshes and one report.

1. read the problem (physics, fluid, speed, size, purpose, quantities) and ask for what is missing;
2. suggest reference geometries from a catalogue with their published sources;
3. suggest the parameters to vary and their ranges, and sample the variants (Latin hypercube, base first);
4. build STEP and STL per variant with validity, volume, tessellation and reference-object checks;
5. plan the mesh for the purpose (screening / design / verification): method, sizes, prism layers from y+,
   domain, cell and cost estimate, and the mesh family for the grid-convergence index;
6. generate the meshes (Gmsh, snappyHexMesh) and measure their quality;
7. write one report with a verdict per stage and the evidence behind it.

Optional language-model help (local via Ollama, or hosted) at every step through :class:`LLMAssist`.
"""
from .catalog import TEMPLATES, Param, Reference, Template
from .context import ProblemContext, parse
from .layers import layer_total, size_layers
from .llm import LLMAssist
from .mesh_plan import MeshPlan, plan_mesh
from .report import gci
from .session import FitMeshSession
from .suggest import suggest_geometries, suggest_parameters

__all__ = ["FitMeshSession", "LLMAssist", "TEMPLATES", "Template", "Param", "Reference", "ProblemContext", "parse",
           "MeshPlan", "plan_mesh", "gci", "suggest_geometries", "suggest_parameters",
           "size_layers", "layer_total"]
