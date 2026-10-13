from __future__ import annotations

import argparse

from pinneapple_tools.benchmark_suite.runner.leaderboard import load_leaderboard


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/artifacts/leaderboard.json")
    ap.add_argument("--sort", default="test_pde_rms", help="Metric key to sort by (ascending)")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    rows = load_leaderboard(args.path)      # [] when the file is missing, empty or not valid JSON
    if not rows:
        print(f"No leaderboard entries in {args.path}.")
        return 0
    key = args.sort

    def k(r):
        v = r.get(key, None)
        try:
            return float(v)
        except Exception:
            return float("inf")

    rows = sorted(rows, key=k)

    print(f"=== Leaderboard (sorted by {key}) ===")
    for r in rows[: int(args.top)]:
        name = r.get("run_name", "n/a")
        rid = r.get("run_id", "n/a")
        pde = r.get("test_pde_rms", None)
        div = r.get("test_div_rms", None)
        bc = r.get("bc_mse", None)
        print(f"- {name:28s} run_id={rid}  pde={pde}  div={div}  bc={bc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
