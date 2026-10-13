"""Publish every zoo model and dataset in ``pinneapple_hub/zoo/manifest.json`` to the Hugging Face Hub.

    HF_TOKEN=... python scripts/zoo/publish_to_hub.py            # create/update repos, upload, tag v<version>
    python scripts/zoo/publish_to_hub.py --dry-run               # list what would be uploaded
    python scripts/zoo/publish_to_hub.py --verify                # anonymous download + SHA-256 check of every file

One repo per item under the PINNeAPPle-Labs organisation (model repos for models, dataset repos for datasets), the
files of the manifest plus a README (the model card / dataset card), and a tag ``v<version>`` -- the revision the
loaders download. Each file's SHA-256 is checked against the manifest before upload, so what is published is
exactly what the loaders accept. Without a token the script prints a warning and exits 0 (the CI workflow relies on
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


def _local(spec: dict) -> Path:
    p = art.ASSETS / spec["bundled"] if spec.get("bundled") else None
    if p is None or not p.exists():
        raise FileNotFoundError(f"{spec} is not bundled; place it under {art.ASSETS} before publishing")
    return p


def plan() -> list[dict]:
    items = []
    for kind in ("models", "datasets"):
        for name, e in art.manifest()[kind].items():
            files = {}
            for fn, spec in e["files"].items():
                p = _local(spec)
                got = art.sha256_file(p)
                if got != spec["sha256"]:
                    raise SystemExit(f"{kind}/{name}/{fn}: SHA-256 {got} != manifest {spec['sha256']}; rebuild first")
                files[fn] = p
            if kind == "models":
                readme = ModelCard.load(str(files["model_card.json"])).to_markdown()
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
                          "revision": e["revision"], "files": files, "readme": meta + readme + usage})
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


def verify(items: list[dict]) -> int:
    """Download every file anonymously (no token) from the tagged revision and compare its SHA-256 with the manifest;
    returns the number of failures."""
    import hashlib
    import urllib.request
    bad = 0
    for it in items:
        prefix = "datasets/" if it["repo_type"] == "dataset" else ""
        for fn in it["files"]:
            url = f"{art.HF_ENDPOINT}/{prefix}{it['repo']}/resolve/{it['revision']}/{fn}"
            want = art.entry(it["kind"], it["name"])["files"][fn]["sha256"]
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "pinneapple-zoo-verify"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    got = hashlib.sha256(r.read()).hexdigest()
                ok = got == want
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
    a = ap.parse_args(argv)
    items = plan()
    if a.verify:
        return 1 if verify(items) else 0
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
