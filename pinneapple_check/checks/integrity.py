"""Integrity of artefacts, with :mod:`pinneapple_security`: manifests of SHA-256 digests (PC801, PC804),
detached Ed25519 signatures (PC802) and per-target hash pins (PC801)."""
from __future__ import annotations

import glob
import json
from pathlib import Path

from ..config import Config
from ..result import Location
from .common import Ctx

__all__ = ["check_integrity", "check_pin"]

_DEFAULT_EXCLUDE = (".git", "__pycache__", "*.sig", "*.sig.json", "manifest.json", "attestation.json")


def check_pin(ctx: Ctx, path: Path, expected: str) -> None:
    from pinneapple_security.integrity import sha256_file
    p = Path(str(path).rsplit(":", 1)[0]) if ":" in str(path)[2:] else Path(path)
    files = sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p]
    if not files:
        ctx.emit("PC801", "sha256", False, f"{ctx.cfg.rel(p)} does not exist")
        return
    if p.is_dir():
        from pinneapple_security.integrity import build_manifest
        got = build_manifest(str(p)).root
        what = "Merkle root of the folder"
    else:
        got, what = sha256_file(str(p)), "SHA-256"
    ok = got == expected.lower().removeprefix("sha256:")
    ctx.emit("PC801", "sha256", ok, f"{ctx.cfg.rel(p)}: {what} {got[:16]}... "
             + ("matches the pinned value" if ok else f"differs from the pinned {expected[:16]}..."))


def check_integrity(ctx: Ctx, cfg: Config) -> None:
    it = cfg.integrity
    loc = Location(cfg.rel(cfg.file), it.line) if cfg.file else Location(".", 1)
    from pinneapple_security.integrity import Manifest
    for mp in it.manifests:
        if not mp.is_file():
            ctx.emit("PC801", cfg.rel(mp), False, f"manifest {cfg.rel(mp)} not found", loc)
            continue
        try:
            man = Manifest.load(str(mp))
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            ctx.emit("PC801", cfg.rel(mp), False, f"manifest {cfg.rel(mp)} is not valid: {e}",
                     Location(cfg.rel(mp), 1))
            continue
        folder = mp.parent
        r = man.verify(str(folder), exclude=(*_DEFAULT_EXCLUDE, mp.name))
        base = cfg.rel(folder)
        for f in r["changed"]:
            ctx.emit("PC801", f"{base}/{f}", False, f"{base}/{f} changed since {cfg.rel(mp)} was built",
                     Location(cfg.rel(folder / f), 1))
        for f in r["removed"]:
            ctx.emit("PC801", f"{base}/{f}", False, f"{base}/{f} is listed in {cfg.rel(mp)} but missing",
                     Location(cfg.rel(mp), _line_in(mp, f)))
        for f in r["added"]:
            ctx.emit("PC804", f"{base}/{f}", False, f"{base}/{f} is not in {cfg.rel(mp)}",
                     Location(cfg.rel(folder / f), 1))
        if not (r["changed"] or r["removed"]):
            ctx.emit("PC801", cfg.rel(mp), True, f"{len(man.files)} files of {base} match {cfg.rel(mp)}",
                     Location(cfg.rel(mp), 1))
        if not r["added"]:
            ctx.emit("PC804", cfg.rel(mp), True, f"every file of {base} is in {cfg.rel(mp)}", Location(cfg.rel(mp), 1))
    if it.signed:
        _signatures(ctx, cfg, loc)


def _line_in(path: Path, needle: str) -> int:
    try:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if f'"{needle}"' in line:
                return i
    except OSError:
        pass
    return 1


def _signatures(ctx: Ctx, cfg: Config, loc: Location) -> None:
    from pinneapple_security.signing import VerificationError, VerifyKey, verify_file
    it = cfg.integrity
    if it.public_key is None or not it.public_key.is_file():
        ctx.emit("PC802", "public-key", False, "signatures are required (integrity.signed) but no public-key file "
                                              "is configured or it does not exist", loc)
        return
    try:
        key = VerifyKey.from_dict(json.loads(it.public_key.read_text()))
    except Exception as e:
        ctx.emit("PC802", "public-key", False, f"cannot read the public key {cfg.rel(it.public_key)}: {e}", loc)
        return
    base = cfg.file.parent if cfg.file else cfg.root
    files = sorted({Path(f) for g in it.signed for f in glob.glob(str(base / g), recursive=True)
                    if Path(f).is_file() and not f.endswith((".sig", ".sig.json"))})
    if not files:
        ctx.emit("PC802", "signed", False, f"no file matches integrity.signed {it.signed}", loc)
        return
    for f in files:
        rel = cfg.rel(f)
        sig = Path(str(f) + ".sig")
        if not sig.is_file():
            ctx.emit("PC802", rel, False, f"{rel} has no signature ({rel}.sig)", Location(rel, 1))
            continue
        try:
            verify_file(str(f), key)
            ctx.emit("PC802", rel, True, f"{rel}: signature verifies with key {key.keyid}", Location(rel, 1))
        except (VerificationError, ValueError, KeyError, json.JSONDecodeError) as e:
            ctx.emit("PC802", rel, False, f"{rel}: signature does not verify with key {key.keyid}: "
                     f"{str(e).replace(str(f), rel)}",
                     Location(rel, 1))
