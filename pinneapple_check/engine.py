"""Running the checks: configuration -> targets -> outcomes -> :class:`~pinneapple_check.result.Report`."""
from __future__ import annotations

from pathlib import Path

from .baseline import apply_baseline
from .checks import Ctx, check_integrity, check_model, check_pin, check_result
from .config import Config, discover, load_config
from .result import Report
from .rules import select_rules

__all__ = ["run"]


def _under(p: Path, roots: list[Path]) -> bool:
    p = Path(str(p).rsplit(":", 1)[0]) if ":" in str(p)[2:] else Path(p)
    try:
        rp = p.resolve()
    except OSError:
        return False
    return any(rp == r or r in rp.parents or rp in r.parents for r in roots)


def run(paths: list[str | Path] | None = None, *, config: Config | str | Path | None = None,
        select: list[str] | None = None, ignore: list[str] | None = None,
        extend_select: list[str] | None = None, baseline: str | Path | None = None,
        use_baseline: bool = True) -> Report:
    """Check the targets of ``config`` (found from the first path or the working directory when None).

    ``paths`` restrict the declared targets to those under them; without declared targets they are searched for
    results. ``select``/``ignore`` take rule-id prefixes and replace those of the configuration."""
    roots = [Path(p).resolve() for p in (paths or [])]
    if not isinstance(config, Config):
        start = roots[0] if roots else Path.cwd()
        cfg = load_config(Path(config) if config else None, start=start if start.is_dir() else start.parent)
    else:
        cfg = config
    enabled = select_rules(select if select is not None else cfg.select,
                           ignore if ignore is not None else cfg.ignore,
                           extend_select if extend_select is not None else cfg.extend_select)
    report = Report(config_file=cfg.rel(cfg.file) if cfg.file else None, root=str(cfg.root))
    targets = cfg.targets
    if targets and roots and not (cfg.file and any(r == cfg.file.resolve() for r in roots)):
        here = cfg.file.parent if cfg.file else cfg.root
        targets = [t for t in targets if _under(t.path, roots)
                   or ((getattr(t, "zoo", None) or getattr(t, "hub", None)) and _under(here, roots))]
    elif not targets:
        base = cfg.file.parent if cfg.file else Path.cwd()
        targets = discover([r for r in roots] or [base], cfg.exclude, cfg.root)
        cfg.results = targets
    notes: list[str] = []
    for t in targets:
        ctx = Ctx(cfg, t, enabled, report.outcomes)
        kind = "model" if t in cfg.models else "result"
        if kind == "result":
            fd = check_result(ctx, t)
            info = {"name": t.name, "kind": fd.kind if fd is not None else "unreadable", "path": cfg.rel(t.path),
                    "physics": t.physics, "declared": t.declared}
        else:
            check_model(ctx, t)
            where = f"zoo:{t.zoo}" if t.zoo else t.hub if t.hub else cfg.rel(Path(str(t.path).rsplit(":", 1)[0]))
            info = {"name": t.name, "kind": "model", "path": where, "physics": t.physics, "declared": True}
        if t.sha256:
            check_pin(ctx, t.path, t.sha256)
        notes += ctx.notes
        report.targets.append(info)
    if cfg.integrity.manifests or cfg.integrity.signed:
        ctx = Ctx(cfg, None, enabled, report.outcomes)
        check_integrity(ctx, cfg)
        notes += ctx.notes
    report.notes = notes
    bpath = Path(baseline) if baseline else cfg.baseline
    if use_baseline and bpath is not None and bpath.is_file():
        apply_baseline(report, bpath)
    return report
