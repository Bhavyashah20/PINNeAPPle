"""What a check produces: an :class:`Outcome` per (rule, target, subject), passed or not, and the :class:`Report`."""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .rules import RULES

__all__ = ["Location", "Outcome", "Report"]


@dataclass
class Location:
    file: str                    # path relative to the project root (posix)
    line: int = 1
    column: int = 1


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if hasattr(v, "item") and callable(v.item):                     # numpy scalar
        try:
            return _clean(v.item())
        except (TypeError, ValueError):
            pass
    return v


@dataclass
class Outcome:
    """One evaluation of a rule. ``passed`` is False for a violation, True for a check that ran and held."""
    rule: str
    target: str                  # the target's name (its path by default)
    subject: str                 # what was checked inside it: a field, a patch, a QoI, a query row...
    passed: bool
    message: str
    location: Location
    value: Any = None
    threshold: Any = None
    severity: str = ""           # filled from the rule (or the config override) by the runner
    baselined: bool = False
    hint: str = ""               # a fix hint specific to this violation (the rule's generic hint otherwise)

    @property
    def fingerprint(self) -> str:
        """Stable across runs and value changes: rule, target and subject, not the message or the numbers."""
        return hashlib.sha256(f"{self.rule}|{self.target}|{self.subject}".encode()).hexdigest()[:16]

    @property
    def title(self) -> str:
        return RULES[self.rule].title

    @property
    def fix(self) -> str:
        return self.hint or RULES[self.rule].fix

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["value"], d["threshold"] = _clean(self.value), _clean(self.threshold)
        d["fingerprint"] = self.fingerprint
        d["name"] = RULES[self.rule].name
        d["stage"] = RULES[self.rule].stage
        d["fix"] = self.fix
        return d


@dataclass
class Report:
    outcomes: list[Outcome] = field(default_factory=list)
    targets: list[dict[str, Any]] = field(default_factory=list)       # {name, kind, path, physics}
    config_file: str | None = None
    root: str = "."
    notes: list[str] = field(default_factory=list)       # what could not be checked, and why

    @property
    def violations(self) -> list[Outcome]:
        return [o for o in self.outcomes if not o.passed and not o.baselined]

    @property
    def baselined(self) -> list[Outcome]:
        return [o for o in self.outcomes if not o.passed and o.baselined]

    @property
    def errors(self) -> list[Outcome]:
        return [o for o in self.violations if o.severity == "error"]

    @property
    def warnings(self) -> list[Outcome]:
        return [o for o in self.violations if o.severity == "warning"]

    def exit_code(self, fail_on: str = "error") -> int:
        bad = self.errors if fail_on == "error" else self.violations if fail_on == "warning" else []
        return 1 if bad else 0

    def trust_card(self) -> dict[str, dict[str, Any]]:
        from .evidence import trust_card
        return trust_card(self)

    def to_dict(self) -> dict[str, Any]:
        from . import __version__
        return {
            "tool": "pinneapple check", "version": __version__, "config": self.config_file,
            "summary": {"targets": len(self.targets), "checks": len(self.outcomes),
                        "passed": sum(o.passed for o in self.outcomes), "errors": len(self.errors),
                        "warnings": len(self.warnings), "baselined": len(self.baselined)},
            "targets": self.targets,
            "notes": self.notes,
            "violations": [o.to_dict() for o in self.violations],
            "checks": [o.to_dict() for o in self.outcomes],
            "trust_card": self.trust_card(),
        }
