"""An equal-budget benchmark in one call: PINN vs FNO vs DeepONet on 1-D Burgers.

    python examples/benchmark/01_equal_budget.py            # smoke: tiny data and budgets, a few minutes on a laptop
    python examples/benchmark/01_equal_budget.py --full     # the real thing (300 s per run, 3 seeds; hours on a CPU)

Everything goes to results/benchmark_burgers/: config.yaml, hardware.json, prepared/ (data, cost, hashes),
validation/, runs/ (the cache: re-running resumes), table.md, table.csv, summary.json, figures/.
The same run from the shell: pinneapple benchmark run --config examples/benchmark/bench.yaml --out results/x --smoke
"""
import sys

import pinneapple as pp

full = "--full" in sys.argv
res = pp.benchmark.run(
    problems=["burgers"],
    methods=["pinn", "fno", "deeponet"],
    budget=pp.ComputeBudget(wall_s=300),          # per training run, one core; or flops=..., steps=...
    budget_levels=[1 / 9, 1 / 3, 1.0],            # error-versus-budget curve
    seeds=3, n_trials=4, search_fraction=1 / 3,   # the same search effort for every method
    equal_flops=True,                             # the extra equal-FLOPs arm
    n_workers=4, threads_per_run=1,
    smoke=not full,
    out_dir="results/benchmark_burgers" + ("" if full else "_smoke"),
)
print(res.table().to_string())
for d in res.diagnostics:                         # what likely hurt the result, with the evidence and a fix
    print("-", d.line())
