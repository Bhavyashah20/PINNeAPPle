"""The zoo catalogue: one Markdown page listing every model and dataset with its metrics, size and Hub link.

    import pinneapple as pp
    print(pp.zoo.catalogue())                  # the page docs/guides/zoo_catalogue.md is generated from

Everything comes from ``manifest.json`` and the model cards, so the page cannot drift from what ships
(``tests/test_model_zoo.py`` checks the committed page against this function).
"""
from __future__ import annotations

from . import _artifacts as art

__all__ = ["markdown"]

HF = "https://huggingface.co"


def _size(entry: dict) -> str:
    b = sum(f["size"] for f in entry["files"].values())
    return f"{b / 1e3:.0f} kB" if b < 1e5 else f"{b / 1e6:.2f} MB" if b < 1e7 else f"{b / 1e6:.0f} MB"


def _where(entry: dict) -> str:
    return "bundled" if all(f.get("bundled") for f in entry["files"].values()) else "Hub-only"


def _fmt(v: float) -> str:
    return f"{v:.3g}" if abs(v) < 1e4 else f"{v:.3e}"


def markdown() -> str:
    """The catalogue page (Markdown)."""
    from . import datasets, info

    man = art.manifest()
    lines = ["# Zoo catalogue", "",
             "Every pre-trained model and dataset of `pp.zoo` / `pp.datasets`, generated from the zoo manifest by "
             "`pp.zoo.catalogue()`. Each item passed the zoo's gate; the candidates that did not, and why, are in "
             "[the inventory](https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/docs/dev/"
             "model_dataset_inventory.md). How to use them: [Pre-trained models and datasets](model_zoo.md).", "",
             "## Models", "",
             "| Model | Version | Task | Held-out metrics (reference) | Trust report | Size | Hugging Face |",
             "|---|---|---|---|---|---|---|"]
    for name, e in man["models"].items():
        card = info(name)
        vm = card.validation_metrics or {}
        metrics = ", ".join(f"{k} {_fmt(vm[k])}" for k in e.get("guard_metrics", []) if k in vm)
        repo = e["hf_repo"]
        lines.append(f"| `{name}` | {e['version']} | {e['task']} | {metrics} | {card.trust_decision or '-'} | "
                     f"{_size(e)} | [{repo}]({HF}/{repo}/tree/{e['revision']}) |")
    lines += ["", "Metric names are `<split>/<metric>`; the full set, the validity range and the physics checks are in "
              "each model card (`pp.zoo.info(name)`).", "",
              "## Datasets", "",
              "| Dataset | Version | Content | Splits (rows) | Licence | Size | Hugging Face |",
              "|---|---|---|---|---|---|---|"]
    for name, e in man["datasets"].items():
        d = datasets.info(name)
        splits = ", ".join(f"{s} {v['n']}" for s, v in d.splits.items())
        repo = e["hf_repo"]
        lines.append(f"| `{name}` | {e['version']} | {d.title} | {splits} | {d.license} | {_size(e)} "
                     f"({_where(e)}) | [{repo}]({HF}/datasets/{repo}/tree/{e['revision']}) |")
    total = sum(f["size"] for k in ("models", "datasets") for x in man[k].values() for f in x["files"].values())
    bundled = sum(f["size"] for k in ("models", "datasets") for x in man[k].values() for f in x["files"].values()
                  if f.get("bundled"))
    lines += ["", f"{len(man['models'])} models and {len(man['datasets'])} datasets, {total / 1e6:.1f} MB in total "
              f"({bundled / 1e6:.1f} MB bundled with the package; the rest is downloaded on first use and checked "
              "against its SHA-256).", "",
              "## Cite", "",
              "Each Hub card carries its own citation. For the software, cite PINNeAPPle (Zenodo DOI "
              "[10.5281/zenodo.23343062](https://doi.org/10.5281/zenodo.23343062), "
              "[CITATION.cff](https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/CITATION.cff)).", ""]
    return "\n".join(lines)
