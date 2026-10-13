"""rea_bridge -- study reference implementations with REA, under a license gate.

`REA <https://github.com/morluto/rea>`_ (Reverse Engineer Anything, MIT) is a
Node.js CLI/MCP server that inspects software without its source and returns
Evidence JSON. Within PINNeAPPle it is a third source for the research
pipeline (:mod:`pinneapple_tools.hpo_experiments`): arXiv gives papers,
GitHub gives repositories, REA gives *how a feature is actually built*,
with the artifact's SHA-256 and REA's stated limitations attached.

Modules
-------
client      run the ``rea`` CLI and parse its JSON (no Python dependency)
licenses    license detection and the adapt / clean_room / reference_only gate
evidence    JavaScript Evidence -> compact feature map; keyword ``locate``
harvest     JSON ledger of studied projects and findings, gate-enforced
tools       registration on :class:`pinneapple_worldmodel.PhysicsToolRegistry`

See ``docs/dev/REA_INTEGRATION.md`` for setup and the harvest workflow.
"""
from .client import PINNED_VERSION, ReaClient, ReaError, ReaResult, ReaUnavailable, resolve_rea_command
from .evidence import (FeatureHit, FeatureMap, FunctionRef, exported_functions_from_layout, locate,
                       summarize_js_evidence)
from .harvest import Finding, HarvestEntry, HarvestLedger, LicenseGateError
from .licenses import LicensePolicy, detect_licenses, policy_for, policy_for_path

__all__ = [
    "PINNED_VERSION", "ReaClient", "ReaError", "ReaResult", "ReaUnavailable", "resolve_rea_command",
    "FeatureHit", "FeatureMap", "FunctionRef", "exported_functions_from_layout", "locate",
    "summarize_js_evidence",
    "Finding", "HarvestEntry", "HarvestLedger", "LicenseGateError",
    "LicensePolicy", "detect_licenses", "policy_for", "policy_for_path",
]
