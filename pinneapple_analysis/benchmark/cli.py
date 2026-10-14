"""``pinneapple benchmark prepare|run|report`` (also ``python -m pinneapple_analysis.benchmark``).

    pinneapple benchmark prepare --config bench.yaml --out results/x     # data, geometry, cost, validation only
    pinneapple benchmark run --config bench.yaml --out results/x         # (prepares what is missing) + train + report
    pinneapple benchmark report results/x                                # tables and figures from the cache
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pinneapple")
    sub = ap.add_subparsers(dest="group", required=True)
    bench = sub.add_parser("benchmark", help="equal-budget benchmarks of physics-AI methods")
    cmd = bench.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "run"):
        p = cmd.add_parser(name)
        p.add_argument("--config", required=True, help="YAML with the same fields as pp.benchmark.run")
        p.add_argument("--out", required=True, help="output folder (re-running resumes)")
        p.add_argument("--force", action="store_true", help="reuse an out_dir prepared/run with another configuration")
        p.add_argument("--smoke", action="store_true", help="tiny data and budgets: a pipeline check in minutes")
        if name == "run":
            p.add_argument("--skip-checks", action="store_true", help="skip the gates (the report then says so)")
            p.add_argument("--allow-failed-validation", action="store_true",
                           help="train even if the prepared artifacts failed validation (shown in the table)")
            p.add_argument("--apply-suggestions", action="store_true",
                           help="write config.suggested.yaml with the advisor's edits")
            p.add_argument("--no-prepare", action="store_true", help="refuse to run if artifacts are not prepared")
    r = cmd.add_parser("report")
    r.add_argument("out")
    a = ap.parse_args(argv)
    from . import api
    from .config import BenchmarkConfig
    if a.cmd == "report":
        res = api.report(a.out)
        print((res.out_dir / "table.md").read_text())
        return 0
    cfg = BenchmarkConfig.from_yaml(a.config)
    if a.smoke:
        cfg.smoke = True
    if a.cmd == "prepare":
        man = api.prepare(cfg, a.out, force=a.force)
        for pid, m in man.items():
            print(f"{pid}: {m['status']}" + (f" (blocking: {m['blocking']})" if m.get("blocking") else ""))
        return 0 if all(m["status"] == "ok" for m in man.values()) else 1
    if a.skip_checks:
        cfg.skip_checks = True
    if a.allow_failed_validation:
        cfg.allow_failed_validation = True
    res = api.run(cfg, a.out, force=a.force, prepare_missing=not a.no_prepare)
    for d in res.diagnostics:
        print("-", d.line())
    if a.apply_suggestions:
        res.apply_suggestions()
        print(f"wrote {res.out_dir / 'config.suggested.yaml'}")
    print(f"report: {res.out_dir / 'table.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
