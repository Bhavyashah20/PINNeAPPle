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
    # Short "notice" form used at the top of many LICENSE files ("... under the
    # terms of the GNU Lesser General Public License as published by the Free
    # Software Foundation, either version 3 ...").
    ("AGPL-3.0", r"gnu affero general public license as published by the free software foundation,? either version 3"),
    ("LGPL-3.0", r"gnu lesser general public license as published by the free software foundation,? either version 3"),
    ("LGPL-2.1", r"gnu lesser general public license as published by the free software foundation,? either version 2\.1"),
    ("GPL-3.0", r"gnu general public license as published by the free software foundation,? either version 3"),
    ("GPL-2.0", r"gnu general public license as published by the free software foundation,? either version 2"),
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

_LICENSE_FILES = re.compile(r"^((licen[sc]e|copying|copyright)([._-].*)?|[al]?gpl[-_.].*)$", re.I)


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


_ALIASES = {
    "gpl3": "GPL-3.0", "gplv3": "GPL-3.0", "gpl-3": "GPL-3.0", "gpl v3": "GPL-3.0",
    "gpl2": "GPL-2.0", "gplv2": "GPL-2.0", "lgpl3": "LGPL-3.0", "lgplv3": "LGPL-3.0",
    "lgpl2.1": "LGPL-2.1", "lgplv2.1": "LGPL-2.1", "mit license": "MIT",
    "apache 2.0": "Apache-2.0", "apache-2": "Apache-2.0", "apache license 2.0": "Apache-2.0",
    "bsd-3": "BSD-3-Clause", "bsd 3-clause": "BSD-3-Clause", "bsd-2": "BSD-2-Clause",
}


def _alias(declared: str) -> str:
    """Map common free-text manifest values to SPDX ids; unknown text is
    kept as-is (and therefore treated as ``reference_only``)."""
    s = declared.strip()
    s2 = re.sub(r"[-_ ](only|or-later)$", "", s, flags=re.I)
    return _ALIASES.get(s2.lower(), s)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def detect_licenses(root: str | Path, *, max_bytes: int = 200_000) -> List[str]:
    """SPDX ids recognised in the license files at the top of ``root``
    (plus ``package.json``/``pyproject.toml``/``setup.py`` declarations)."""
    root = Path(root)
    found: List[str] = []
    _seen_in: dict = {}
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
                if gnu and any(f.startswith(("GPL", "LGPL", "AGPL")) for f in found
                               if p.name in _seen_in.get(f, ())):
                    continue                     # one GNU family per file
                found.append(spdx)
                _seen_in.setdefault(spdx, set()).add(p.name)
    # COPYING (GPL text) + COPYING.LESSER (LGPL) is how LGPL projects ship:
    # the LGPL is the license; the GPL text is incorporated by reference.
    if any(f.startswith("LGPL") for f in found):
        found = [f for f in found if not f.startswith("GPL")]
    for manifest, pattern in (("package.json", r'"license"\s*:\s*"([^"]+)"'),
                              ("pyproject.toml", r'^license\s*=\s*"([^"]+)"'),
                              ("setup.py", r"""license\s*=\s*["']([^"']+)["']""")):
        f = root / manifest
        if f.is_file():
            m = re.search(pattern, f.read_text(errors="replace"), re.M)
            spdx = _alias(m.group(1)) if m else None
            if spdx and spdx not in found:
                found.append(spdx)
    return found


def policy_for_path(root: str | Path) -> LicensePolicy:
    return policy_for(detect_licenses(root))
