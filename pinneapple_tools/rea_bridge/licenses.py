"""License gate for anything harvested from another project into PINNeAPPle.

PINNeAPPle is Apache-2.0. What may flow from a studied project into this
repository depends on that project's license, so every harvest is tagged
with one of three policies -- the same "a missing license means all rights
reserved" stance as :mod:`pinneapple_catalog.resources`:

``adapt``
    Permissive and Apache-2.0-compatible (MIT, BSD, Apache-2.0, ISC, Zlib,
    CC-BY-4.0, CC0, Unlicense). Code may be adapted; the original notice
    must be kept (``attribution_required``) -- recorded in the harvest
    ledger and, for copied code, in ``NOTICE``.
``clean_room``
    Copyleft (GPL, LGPL, AGPL, MPL, EPL). Only *behaviour* may be taken:
    equations, algorithms as published in papers, interfaces, observed
    input/output. The PINNeAPPle implementation is written from the
    published method, never from the code, and the ledger refuses an entry
    marked ``adapted`` under this policy.
``reference_only``
    No license, unrecognised license, or a non-commercial / no-derivatives
    term. Read for orientation; nothing is reused.

Detection is deliberately conservative: when a repository carries several
licenses (e.g. VisualPDE: MIT for code, CC-BY-4.0 for HTML/CSS) the most
restrictive one decides, and anything not positively recognised is
``reference_only``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

__all__ = ["LicensePolicy", "policy_for", "detect_licenses", "policy_for_path", "POLICIES"]

POLICIES = ("adapt", "clean_room", "reference_only")
_RANK = {"adapt": 0, "clean_room": 1, "reference_only": 2}

_PERMISSIVE = {
    "MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC", "Zlib",
    "CC-BY-4.0", "CC0-1.0", "Unlicense", "0BSD", "BSL-1.0", "PSF-2.0",
}
_COPYLEFT = {
    "GPL-2.0", "GPL-3.0", "LGPL-2.1", "LGPL-3.0", "AGPL-3.0", "MPL-2.0", "EPL-2.0",
}

# (SPDX id, regex over the normalised license text). Order matters: the more
# specific texts (LGPL, AGPL) must be tested before their substrings (GPL).
_TEXT_RULES: List[Tuple[str, str]] = [
    ("AGPL-3.0", r"gnu affero general public license"),
    ("LGPL-3.0", r"gnu lesser general public license\s+version 3"),
    ("LGPL-2.1", r"gnu lesser general public license\s+version 2\.1"),
    ("GPL-3.0", r"gnu general public license\s+version 3"),
    ("GPL-2.0", r"gnu general public license\s+version 2"),
    ("MPL-2.0", r"mozilla public license,?\s+(v\.|version)\s*2\.0"),
    ("EPL-2.0", r"eclipse public license\s*-?\s*v?\s*2\.0"),
    ("Apache-2.0", r"apache license,?\s+version 2\.0"),
    ("MIT", r"permission is hereby granted, free of charge, to any person obtaining a copy"),
    ("BSD-3-Clause", r"neither the name of .{0,200}? nor the names of its\s+contributors"),
    ("BSD-2-Clause", r"redistributions in binary form must reproduce the above copyright"),
    ("ISC", r"permission to use, copy, modify, and/or distribute this software for any"),
    ("CC-BY-NC", r"creativecommons\.org/licenses/by-nc|attribution-noncommercial"),
    ("CC-BY-4.0", r"creativecommons\.org/licenses/by/4\.0|attribution 4\.0 international"),
    ("CC0-1.0", r"cc0 1\.0 universal"),
    ("Unlicense", r"this is free and unencumbered software released into the public domain"),
]

_LICENSE_FILES = re.compile(r"^(licen[sc]e|copying|copyright)([._-].*)?$", re.I)


@dataclass(frozen=True)
class LicensePolicy:
    policy: str
    licenses: Tuple[str, ...]
    attribution_required: bool
    reason: str
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def may_copy_code(self) -> bool:
        return self.policy == "adapt"


def _family(spdx: str) -> str:
    base = spdx.replace("-only", "").replace("-or-later", "").rstrip("+")
    if base in _PERMISSIVE:
        return "adapt"
    if base in _COPYLEFT:
        return "clean_room"
    return "reference_only"


def policy_for(spdx_ids: Iterable[str]) -> LicensePolicy:
    """Most restrictive policy over the given SPDX identifiers."""
    ids = tuple(dict.fromkeys(s for s in spdx_ids if s))
    if not ids:
        return LicensePolicy("reference_only", (), False,
                             "no license found: all rights reserved")
    worst = max(ids, key=lambda s: _RANK[_family(s)])
    pol = _family(worst)
    reason = {
        "adapt": "permissive, Apache-2.0 compatible; keep the original notice",
        "clean_room": f"{worst} is copyleft: reimplement from the published method, never from the code",
        "reference_only": f"{worst} is not a recognised reusable license",
    }[pol]
    return LicensePolicy(pol, ids, pol != "reference_only", reason)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def detect_licenses(root: str | Path, *, max_bytes: int = 200_000) -> List[str]:
    """SPDX ids recognised in the license files at the top of ``root``
    (plus ``package.json``/``pyproject.toml`` declarations)."""
    root = Path(root)
    found: List[str] = []
    for p in sorted(root.iterdir()) if root.is_dir() else []:
        if p.is_file() and _LICENSE_FILES.match(p.name):
            text = _normalise(p.read_text(errors="replace")[:max_bytes])
            # GNU licenses quote each other (GPL-3.0 section 13 names the
            # AGPL; the LGPL incorporates the GPL), so the family is decided
            # by the title at the top of the file only.
            head = text[:1200]
            for spdx, pattern in _TEXT_RULES:
                gnu = spdx.startswith(("GPL", "LGPL", "AGPL"))
                if spdx in found or not re.search(pattern, head if gnu else text):
                    continue
                if gnu and any(f.startswith(("GPL", "LGPL", "AGPL")) for f in found):
                    continue
                found.append(spdx)
    for manifest, pattern in (("package.json", r'"license"\s*:\s*"([^"]+)"'),
                              ("pyproject.toml", r'^license\s*=\s*"([^"]+)"')):
        f = root / manifest
        if f.is_file():
            m = re.search(pattern, f.read_text(errors="replace"), re.M)
            if m and m.group(1) not in found:
                found.append(m.group(1))
    return found


def policy_for_path(root: str | Path) -> LicensePolicy:
    return policy_for(detect_licenses(root))
