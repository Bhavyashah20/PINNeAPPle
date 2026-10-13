"""Publish every zoo model and dataset in ``pinneapple_hub/zoo/manifest.json`` to the Hugging Face Hub.

    HF_TOKEN=... python scripts/zoo/publish_to_hub.py            # create/update repos, upload, tag v<version>
    python scripts/zoo/publish_to_hub.py --dry-run               # list what would be uploaded
    python scripts/zoo/publish_to_hub.py --verify                # anonymous download + SHA-256 check of every file
    python scripts/zoo/publish_to_hub.py --staging DIR           # also Hub-only items, files in DIR/<name>/<file>

One repo per item under the PINNeAPPle-Labs organisation (model repos for models, dataset repos for datasets), the
files of the manifest plus a README (the model card / dataset card), and a tag ``v<version>`` -- the revision the
loaders download. Each file's SHA-256 is checked against the manifest before upload, so what is published is
exactly what the loaders accept. Hub-only items (files too large to bundle, no ``bundled`` entry in the manifest) are
uploaded only when their files are found in the ``--staging`` directory; otherwise they are skipped with a note.
Without a token the script prints a warning and exits 0 (the CI workflow relies on
that to skip cleanly when the HF_TOKEN secret is not set).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pinneapple_hub.model_card import ModelCard  # noqa: E402
from pinneapple_hub.zoo import _artifacts as art  # noqa: E402
from pinneapple_hub.zoo import datasets as zds  # noqa: E402

CITE = ("\n\n## Cite\n\nIf you use this {what}, please cite PINNeAPPle (software, Zenodo DOI "
        "[10.5281/zenodo.23343062](https://doi.org/10.5281/zenodo.23343062); see "
        "[CITATION.cff](https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/CITATION.cff)):\n\n```bibtex\n"
        "@software{{pinneapple,\n  author = {{Barros, Yan and Schiassi, Enrico}},\n  title  = {{PINNeAPPle}},\n"
        "  doi    = {{10.5281/zenodo.23343062}},\n  url    = {{https://github.com/PINNeAPPle-Labs/PINNeAPPle}}\n}}\n```\n")


def _local(kind: str, name: str, fn: str, spec: dict, staging: Path | None) -> Path | None:
    """The local copy of a file: bundled, else ``staging/<name>/<fn>``; None when neither exists."""
    if spec.get("bundled"):
        p = art.ASSETS / spec["bundled"]
        if not p.exists():
            raise FileNotFoundError(f"{kind}/{name}/{fn}: bundled file {p} is missing; rebuild first")
        return p
    if staging is not None and (staging / name / fn).exists():
        return staging / name / fn
    return None


def plan(staging: Path | None = None, *, need_files: bool = True) -> list[dict]:
    items = []
    for kind in ("models", "datasets"):
        for name, e in art.manifest()[kind].items():
            files: dict = {}
            hub_only = not all(spec.get("bundled") for spec in e["files"].values())
            if need_files:
                for fn, spec in e["files"].items():
                    p = _local(kind, name, fn, spec, staging)
                    if p is None:
                        files = None
                        break
                    got = art.sha256_file(p)
                    if got != spec["sha256"]:
                        raise SystemExit(f"{kind}/{name}/{fn}: SHA-256 {got} != manifest {spec['sha256']}; "
                                         "rebuild first")
                    files[fn] = p
                if files is None:
                    print(f"  skip {kind[:-1]} {name}: Hub-only and its files are not in the staging directory")
                    continue
            else:
                files = dict.fromkeys(e["files"])
            if kind == "models":
                readme = ModelCard.load(str(files["model_card.json"])).to_markdown() if need_files else ""
                meta = f"---\nlicense: {e.get('license', 'apache-2.0')}\nlibrary_name: pinneapple\ntags:\n" + \
                       "".join(f"- {t}\n" for t in ["pinneapple", *e.get("tags", [])]) + "---\n\n"
            else:
                readme = zds.info(name).to_markdown()
                meta = f"---\nlicense: {e['license']}\ntags:\n" + "".join(f"- {t}\n" for t in
                                                                          ["pinneapple", *e.get("tags", [])]) + "---\n\n"
            usage = ("\n\n## Use\n\n```python\nimport pinneapple as pp\n" +
                     (f"m = pp.zoo.load(\"{name}\")\npp.zoo.evaluate(\"{name}\")\n" if kind == "models" else
                      f"d = pp.datasets.load(\"{name}\", split=\"test\")\n") + "```\n")
            items.append({"kind": kind, "name": name, "repo": e["hf_repo"], "repo_type": e["repo_type"],
                          "revision": e["revision"], "files": files, "hub_only": hub_only,
                          "readme": meta + readme + usage + CITE.format(what="model" if kind == "models" else
                                                                        "dataset")})
    return items


def publish(items: list[dict], token: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    for it in items:
        api.create_repo(it["repo"], repo_type=it["repo_type"], exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            for fn, p in it["files"].items():
                (Path(tmp) / fn).write_bytes(p.read_bytes())
            (Path(tmp) / "README.md").write_text(it["readme"])
            commit = api.upload_folder(repo_id=it["repo"], repo_type=it["repo_type"], folder_path=tmp,
                                       commit_message=f"{it['name']} {it['revision']}")
        refs = api.list_repo_refs(it["repo"], repo_type=it["repo_type"])
        if it["revision"] in {t.name for t in refs.tags}:
            print(f"  {it['repo']}: tag {it['revision']} already exists (kept; bump the version to republish)")
        else:
            api.create_tag(it["repo"], tag=it["revision"], repo_type=it["repo_type"],
                           revision=getattr(commit, "oid", None))
        print(f"published {it['kind'][:-1]} {it['name']} -> https://huggingface.co/"
              f"{'datasets/' if it['repo_type'] == 'dataset' else ''}{it['repo']} @ {it['revision']}")


def verify(items: list[dict], allow_unpublished: bool = False) -> int:
    """Download every file anonymously (no token) from the tagged revision and compare its SHA-256 with the manifest;
    returns the number of failures. With ``allow_unpublished``, a Hub-only item that is not on the Hub yet (HTTP 404)
    is reported but not counted (it waits for a run with its staging files); a wrong checksum always fails."""
    import hashlib
    import urllib.error
    import urllib.request
    bad = 0
    for it in items:
        prefix = "datasets/" if it["repo_type"] == "dataset" else ""
        for fn in it["files"]:
            url = f"{art.HF_ENDPOINT}/{prefix}{it['repo']}/resolve/{it['revision']}/{fn}"
            want = art.entry(it["kind"], it["name"])["files"][fn]["sha256"]
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "pinneapple-zoo-verify"})
                h = hashlib.sha256()
                with urllib.request.urlopen(req, timeout=600) as r:
                    for block in iter(lambda: r.read(1 << 20), b""):
                        h.update(block)
                got = h.hexdigest()
                ok = got == want
            except urllib.error.HTTPError as e:
                ok, got = False, f"error: {e}"
                if e.code == 404 and allow_unpublished and it["hub_only"]:
                    print(f"  ... {url} not published yet (Hub-only; publish it with its staging files)")
                    continue
            except Exception as e:  # noqa: BLE001 - report every failure, keep going
                ok, got = False, f"error: {e}"
            bad += not ok
            print(f"  {'OK ' if ok else 'BAD'} {url} sha256 {got[:16]}{'' if ok else ' != ' + want[:16]}")
    print(f"verify: {sum(len(it['files']) for it in items) - bad} OK, {bad} failed")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true", help="only verify the published files (anonymous download)")
    ap.add_argument("--staging", type=Path, help="directory with the files of Hub-only items (<name>/<file>)")
    ap.add_argument("--allow-unpublished", action="store_true",
                    help="--verify: a Hub-only item missing from the Hub is a note, not a failure")
    a = ap.parse_args(argv)
    if a.verify:
        return 1 if verify(plan(need_files=False), a.allow_unpublished) else 0
    items = plan(a.staging)
    size = sum(p.stat().st_size for it in items for p in it["files"].values())
    print(f"{len(items)} items, {sum(len(it['files']) for it in items)} files, {size / 1e6:.2f} MB")
    for it in items:
        print(f"  {it['repo_type']:7s} {it['repo']} @ {it['revision']}: {', '.join(it['files'])}")
    if a.dry_run:
        return 0
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("::warning::HF_TOKEN is not set: nothing was uploaded. Add the HF_TOKEN repository secret (a token "
              "with write access to the PINNeAPPle-Labs organisation) and re-run this workflow.")
        return 0
    publish(items, token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
