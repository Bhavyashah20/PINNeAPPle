"""Turn REA's JavaScript application Evidence into a compact feature map.

``rea analyze-javascript-application`` returns a large (often tens of MB)
Evidence JSON whose ``normalized_result.semantic_graph`` holds every
function, call site, binding and relation REA recovered from the code, each
with an exact source range. This module keeps only what is useful for
studying *how a feature is implemented*:

- named functions per module with their line ranges;
- the resolved static call graph between them (``calls`` relations from a
  call site to a function, attributed to the call site's enclosing function);
- timers and storage mechanisms the app uses;
- REA's own limitations, copied verbatim.

:func:`locate` then answers "where is feature X?" by matching keywords
against function names and the source lines inside each function's range,
and expands the hits one level along the call graph. Every hit points back to
a file, a line range and the Evidence id, so a finding can always be checked
against the original.

Nothing here infers semantics REA did not record: an unresolved call stays
unresolved, an anonymous function stays anonymous.
"""
from __future__ import annotations

import fnmatch
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["FunctionRef", "FeatureMap", "FeatureHit", "summarize_js_evidence", "locate"]

#: Third-party code that is almost never the feature being studied.
DEFAULT_VENDOR_GLOBS = ("*.min.js", "node_modules/*", "*/vendor/*", "vendor/*", "*/lib/*")


@dataclass(frozen=True)
class FunctionRef:
    node_id: str
    name: Optional[str]
    module: str
    start_line: int
    end_line: int

    @property
    def label(self) -> str:
        return f"{self.module}:{self.start_line}-{self.end_line} {self.name or '<anonymous>'}"


@dataclass
class FeatureMap:
    evidence_id: Optional[str]
    subject_sha256: Optional[str]
    root: Optional[str]
    functions: Dict[str, FunctionRef] = field(default_factory=dict)
    calls: Dict[str, List[str]] = field(default_factory=dict)       # caller -> callees
    callers: Dict[str, List[str]] = field(default_factory=dict)     # callee -> callers
    timers: List[str] = field(default_factory=list)
    storage: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    vendor_modules: List[str] = field(default_factory=list)

    def named_functions(self, module: Optional[str] = None) -> List[FunctionRef]:
        fs = [f for f in self.functions.values() if f.name and (module is None or f.module == module)]
        return sorted(fs, key=lambda f: (f.module, f.start_line))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "subject_sha256": self.subject_sha256,
            "root": self.root,
            "modules": sorted({f.module for f in self.functions.values()}),
            "vendor_modules": self.vendor_modules,
            "functions": [f.label for f in self.named_functions()],
            "call_edges": sum(len(v) for v in self.calls.values()),
            "timers": self.timers,
            "storage": self.storage,
            "limitations": self.limitations,
        }


@dataclass
class FeatureHit:
    function: FunctionRef
    matched: List[str]
    snippet: str
    callees: List[str]
    callers: List[str]


def _is_vendor(module: str, globs: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(module, g) for g in globs)


def _range(node: Dict[str, Any]) -> Tuple[int, int]:
    rng = (node.get("identity") or {}).get("source_range") or {}
    return int((rng.get("start") or {}).get("line", 0)), int((rng.get("end") or {}).get("line", 0))


def summarize_js_evidence(evidence: Dict[str, Any], *,
                          vendor_globs: Sequence[str] = DEFAULT_VENDOR_GLOBS,
                          include_vendor: bool = False) -> FeatureMap:
    """Build a :class:`FeatureMap` from one ``analyze_javascript_application``
    Evidence payload (the parsed JSON, not the path)."""
    nr = evidence.get("normalized_result") or {}
    sg = nr.get("semantic_graph") or {}
    nodes = {n["node_id"]: n for n in sg.get("nodes", [])}
    fmap = FeatureMap(
        evidence_id=evidence.get("evidence_id"),
        subject_sha256=((evidence.get("subject") or {}).get("digest") or {}).get("sha256"),
        root=nr.get("input_path"),
        limitations=list(dict.fromkeys([*(evidence.get("limitations") or []),
                                        *(nr.get("limitations") or [])])),
    )
    vendor = set()
    for nid, n in nodes.items():
        if n.get("kind") != "function":
            continue
        module = (n.get("identity") or {}).get("module_path") or "?"
        if _is_vendor(module, vendor_globs):
            vendor.add(module)
            if not include_vendor:
                continue
        start, end = _range(n)
        fmap.functions[nid] = FunctionRef(nid, (n.get("properties") or {}).get("name"),
                                          module, start, end)
    fmap.vendor_modules = sorted(vendor)

    calls, callers = defaultdict(set), defaultdict(set)
    for r in sg.get("relations", []):
        kind = r.get("relation")
        src = nodes.get(r.get("source_node_id"), {})
        if kind == "calls" and src.get("kind") == "call-site":
            caller, callee = src.get("function_node_id"), r.get("target_node_id")
            if caller in fmap.functions and callee in fmap.functions:
                calls[caller].add(callee)
                callers[callee].add(caller)
        elif kind == "schedules-timer":
            fmap.timers.append(f"{(src.get('identity') or {}).get('module_path')}:{_range(src)[0]}")
    fmap.calls = {k: sorted(v) for k, v in calls.items()}
    fmap.callers = {k: sorted(v) for k, v in callers.items()}

    for n in (nr.get("graph") or {}).get("nodes", []):
        if n.get("kind") == "storage":
            key = (n.get("identity") or {}).get("key")
            if key and key not in fmap.storage:
                fmap.storage.append(key)
    return fmap


def locate(fmap: FeatureMap, keywords: Iterable[str], *, root: Optional[str | Path] = None,
           max_hits: int = 20, snippet_lines: int = 40) -> List[FeatureHit]:
    """Functions whose name or body mentions any keyword (case-insensitive),
    most matches first. ``root`` defaults to the analysed directory."""
    kws = [k.lower() for k in keywords if k]
    if not kws:
        return []
    base = Path(root or fmap.root or ".")
    cache: Dict[str, List[str]] = {}

    def lines_of(module: str) -> List[str]:
        if module not in cache:
            p = base / module
            cache[module] = p.read_text(errors="replace").splitlines() if p.is_file() else []
        return cache[module]

    scored = []
    for f in fmap.functions.values():
        body = lines_of(f.module)[max(f.start_line - 1, 0): f.end_line]
        hay = ((f.name or "") + "\n" + "\n".join(body)).lower()
        matched = [k for k in kws if re.search(re.escape(k), hay)]
        if matched:
            # Prefer the innermost function that matches: shorter body first on ties.
            scored.append((len(matched), -(f.end_line - f.start_line), f, matched, body))
    # Drop an enclosing function when a function nested inside it already
    # matches every keyword it matches (e.g. a 14k-line module wrapper).
    def _inside(g: FunctionRef, f: FunctionRef) -> bool:
        return (g is not f and g.module == f.module and f.start_line <= g.start_line
                and g.end_line <= f.end_line and (g.end_line - g.start_line) < (f.end_line - f.start_line))

    scored = [t for t in scored
              if not any(_inside(u[2], t[2]) and set(t[3]) <= set(u[3]) for u in scored)]
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

    hits = []
    for _, _, f, matched, body in scored[:max_hits]:
        hits.append(FeatureHit(
            function=f,
            matched=matched,
            snippet="\n".join(body[:snippet_lines]),
            callees=[fmap.functions[c].label for c in fmap.calls.get(f.node_id, [])],
            callers=[fmap.functions[c].label for c in fmap.callers.get(f.node_id, [])],
        ))
    return hits


def exported_functions_from_layout(evidence: Dict[str, Any]) -> List[str]:
    """Exported functions recorded by ``rea inspect-binary-layout`` (ELF):
    entries of the ``SHT_DYNSYM`` table that are defined, global/weak,
    default-visibility FUNC/GNU_IFUNC symbols.

    Lets a check done in Python (``fmu_inspect.elf_exported_functions``) be
    confirmed against REA's independent parser, with REA's Evidence id."""
    nr = evidence.get("normalized_result") or {}
    dynsym = {s.get("index") for s in nr.get("sections", []) if s.get("type") == "SHT_DYNSYM"}
    out = set()
    for sym in nr.get("symbols", []):
        if sym.get("table_index") not in dynsym:
            continue
        if sym.get("section_index") in ("SHN_UNDEF", 0, None):
            continue
        if sym.get("binding") not in ("STB_GLOBAL", "STB_WEAK"):
            continue
        if sym.get("type") not in ("STT_FUNC", "STT_GNU_IFUNC") or sym.get("visibility") != "STV_DEFAULT":
            continue
        name = (sym.get("name") or {}).get("display")
        if name:
            out.add(name)
    return sorted(out)
