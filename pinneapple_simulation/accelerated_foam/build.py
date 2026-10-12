"""Compile the ``pinneWarmStart`` linear-solver library against an installed OpenFOAM.

The C++ source (``plugin/pinneWarmStart.C``) is part of this package. Building it needs the OpenFOAM
headers of the installed version and a C++ compiler:

- a full OpenFOAM installation (``$FOAM_SRC/OpenFOAM/lnInclude`` exists): those headers and ``$FOAM_LIBBIN``;
- a binary-only installation (e.g. the Debian/Ubuntu ``openfoam`` package, which ships no headers): the
  source tree of the *same* version, e.g. ``apt-get source openfoam`` (needs a ``deb-src`` entry), passed
  as ``openfoam_src`` or ``$PINNEAPPLE_OPENFOAM_SRC``. Its headers are only read at build time; nothing is
  copied into this repository.

The library is cached under ``~/.cache/pinneapple/accelerated_foam`` keyed on the source and the flags, and
loaded by a case with ``libs ("<path>");`` in ``controlDict``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

PLUGIN_SOURCE = Path(__file__).with_name("plugin") / "pinneWarmStart.C"
LIB_NAME = "libpinneWarmStart.so"


def _flat_include(tree: Path, dest: Path) -> Path:
    """A flat directory of symlinks to every header/template of ``tree`` (what wmake's lnInclude holds)."""
    dest.mkdir(parents=True, exist_ok=True)
    for p in tree.rglob("*"):
        if p.suffix in (".H", ".C", ".hpp") and p.is_file():
            link = dest / p.name
            if not link.exists():
                link.symlink_to(p.resolve())
    return dest


def include_dirs(
    openfoam_src: str | os.PathLike | None = None, work: Path | None = None
) -> list[Path]:
    """Header directories for ``OpenFOAM`` and ``OSspecific/POSIX``: from ``$FOAM_SRC`` lnInclude when present,
    else built from a source tree. Raises ``FileNotFoundError`` when neither is available."""
    foam_src = os.environ.get("FOAM_SRC")
    if foam_src:
        cand = [
            Path(foam_src) / "OpenFOAM" / "lnInclude",
            Path(foam_src) / "OSspecific" / "POSIX" / "lnInclude",
        ]
        if all((c / "lduMatrix.H").exists() or (c / "OSspecific.H").exists() for c in cand):
            return cand
    src = openfoam_src or os.environ.get("PINNEAPPLE_OPENFOAM_SRC")
    if not src:
        raise FileNotFoundError(
            "OpenFOAM headers not found: set FOAM_SRC (full install) or pass openfoam_src / "
            "PINNEAPPLE_OPENFOAM_SRC (source tree of the installed version)"
        )
    root = Path(src)
    if (root / "src").is_dir():
        root = root / "src"
    if not (root / "OpenFOAM").is_dir():
        raise FileNotFoundError(f"{src}: no OpenFOAM/ directory (expected an OpenFOAM source tree)")
    work = work or (
        _cache_root() / "include" / hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:12]
    )
    return [
        _flat_include(root / "OpenFOAM", work / "OpenFOAM"),
        _flat_include(root / "OSspecific" / "POSIX", work / "OSspecific"),
    ]


def _cache_root() -> Path:
    return (
        Path(os.environ.get("PINNEAPPLE_CACHE", Path.home() / ".cache" / "pinneapple"))
        / "accelerated_foam"
    )


def _lib_dir() -> Path:
    for cand in (os.environ.get("FOAM_LIBBIN"), "/usr/lib/openfoam", "/usr/lib"):
        if cand and (Path(cand) / "libOpenFOAM.so").exists():
            return Path(cand)
    raise FileNotFoundError("libOpenFOAM.so not found (set FOAM_LIBBIN)")


def compile_flags(
    label_size: int | None = None, precision: str | None = None, api: int | None = None
) -> list[str]:
    label_size = label_size or int(os.environ.get("WM_LABEL_SIZE", 32))
    precision = precision or os.environ.get("WM_PRECISION_OPTION", "DP")
    api = api or int(os.environ.get("FOAM_API", 1912))
    return [
        "-std=c++11",
        "-m64",
        "-pthread",
        f"-DOPENFOAM={api}",
        f"-DWM_{precision}",
        f"-DWM_LABEL_SIZE={label_size}",
        "-O2",
        "-fPIC",
        "-DNoRepository",
        "-ftemplate-depth-100",
    ]


def build_plugin(
    openfoam_src: str | os.PathLike | None = None,
    out_dir: str | os.PathLike | None = None,
    force: bool = False,
    cxx: str | None = None,
) -> Path:
    """Compile ``libpinneWarmStart.so``; return its path (cached unless ``force``)."""
    cxx = cxx or os.environ.get("CXX") or shutil.which("g++") or "g++"
    incs = include_dirs(openfoam_src)
    libdir = _lib_dir()
    flags = compile_flags()
    key = hashlib.sha1(
        PLUGIN_SOURCE.read_bytes() + repr((flags, [str(i) for i in incs], str(libdir))).encode()
    )
    out = Path(out_dir) if out_dir else _cache_root() / key.hexdigest()[:16]
    lib = out / LIB_NAME
    if lib.exists() and not force:
        return lib
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        cxx,
        *flags,
        *[f"-I{i}" for i in incs],
        "-shared",
        "-Xlinker",
        "--add-needed",
        "-Xlinker",
        "--no-as-needed",
        str(PLUGIN_SOURCE),
        "-o",
        str(lib),
        f"-L{libdir}",
        "-lOpenFOAM",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("building pinneWarmStart failed:\n" + r.stderr[-4000:])
    return lib


def find_plugin(openfoam_src: str | os.PathLike | None = None) -> Path | None:
    """``$PINNEAPPLE_WARMSTART_LIB`` if set, else a build (cached); None when it cannot be built."""
    env = os.environ.get("PINNEAPPLE_WARMSTART_LIB")
    if env and Path(env).exists():
        return Path(env)
    try:
        return build_plugin(openfoam_src)
    except (FileNotFoundError, RuntimeError, OSError):
        return None
