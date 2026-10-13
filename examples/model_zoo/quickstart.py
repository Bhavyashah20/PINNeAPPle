"""Model zoo quickstart: load a pre-trained model, predict, compare with the reference, validate, benchmark and
fine-tune -- every step on the shipped, verified airfoil surrogate and its dataset.

    python examples/model_zoo/quickstart.py            # ~1 min on a laptop CPU (the benchmark step needs pp.benchmark)
"""
from __future__ import annotations

import tempfile

import numpy as np

import pinneapple as pp

# 1. What is in the zoo ------------------------------------------------------------------------------------------
print(pp.zoo.list())
print(pp.datasets.list())
card = pp.zoo.info("airfoil_cst_rans_mlp")
print(card.task)
print({d["name"]: d["unit"] for d in card.inputs}, "->", {d["name"]: d["unit"] for d in card.outputs})
print("valid for alpha", card.validity_range["alpha_deg"], "deg, Re", card.validity_range["reynolds"])

# 2. Load and predict -------------------------------------------------------------------------------------------
model = pp.zoo.load("airfoil_cst_rans_mlp")
naca2412 = dict(T0=0.169591, T1=0.144959, T2=0.131433, M0=0.024922, M1=0.071088, M2=0.058960)  # NACA 2412 (CST)
alphas = np.array([0.0, 2.0, 4.0, 6.0])
pred = model.predict(**naca2412, alpha_deg=alphas)
for a, cl, cd, s in zip(alphas, pred["Cl"], pred["Cd"], pred["Cl_std"], strict=True):
    print(f"alpha {a:4.1f}  Cl {cl:.4f} +- {s:.4f}  Cd {cd:.5f}  L/D {cl / cd:6.1f}")

# 3. Compare with the reference -----------------------------------------------------------------------------------
ev = pp.zoo.evaluate("airfoil_cst_rans_mlp")              # held-out shapes, NACA runs and the XFOIL polar
print("card metrics reproduced:", ev["matches_card"])
for k in ("test/Cl_mae", "test/Cd_mape_pct", "naca_reference/Cd_mape_pct", "xfoil_naca2412/Cd_mape_pct"):
    print(f"  {k:28s} {ev['metrics'][k]:.4g}")
xfoil = pp.datasets.load("airfoil_rans_cst_re4e6", split="xfoil_naca2412")   # independent solver (XFOIL 6.99)
p = model.predict({k: xfoil[k] for k in model.input_names})
print("vs XFOIL, max |dCl|:", float(np.max(np.abs(p["Cl"] - xfoil["Cl"]))))

# 4. Validate: physics checks + validity range ---------------------------------------------------------------------
report = pp.zoo.validate("airfoil_cst_rans_mlp", inputs={**naca2412, "alpha_deg": [4.0, 18.0]})
print(report.decisao)                                     # REVIEW: alpha = 18 deg is outside the validity range
print(report.to_markdown())

# 5. Benchmark with pp.benchmark: the shipped model vs the same network trained from scratch for 30 s -------------
try:
    res = pp.zoo.benchmark("airfoil_cst_rans_mlp", budget_s=30, out_dir=tempfile.mkdtemp())
    print(res.to_markdown().split("### Cost")[0])
except ImportError as e:
    print("benchmark skipped:", e)

# 6. Fine-tune with a held-out guard ---------------------------------------------------------------------------------
new = pp.datasets.load("airfoil_rans_cst_re4e6", split="naca_reference")    # stand-in for your own new runs
ft = pp.zoo.finetune("airfoil_cst_rans_mlp", new, epochs=50, lr=1e-5)
print(ft.summary())
if ft.accepted:
    ft.save(tempfile.mkdtemp())
else:
    print("not saved: the fine-tuned model is worse on the original test set (force=True overrides)")
