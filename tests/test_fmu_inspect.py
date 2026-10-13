"""fmu_inspect: static FMU preflight.

References: the ELF parser is checked against binutils ``nm -D`` (an
independent implementation) on a system library; the required-function sets
are checked against the counts of the FMI standard headers (FMI 2: 44
functions, FMI 3: 75 -- counted from fmi2FunctionTypes.h / fmi3FunctionTypes.h); real Modelica Reference-FMUs are checked when
``PINNEAPPLE_REFERENCE_FMUS`` points at a folder of built ``.fmu`` files.
"""
from __future__ import annotations

import ctypes.util
import glob
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from pinneapple_simulation.external_solvers.modelica.fmu_inspect import (
    REQUIRED_FUNCTIONS, elf_exported_functions, inspect_fmu)


def _system_elf():
    for cand in ("/lib/x86_64-linux-gnu/libm.so.6", "/usr/lib/x86_64-linux-gnu/libm.so.6", "/lib64/libm.so.6"):
        if os.path.isfile(cand):
            return cand
    name = ctypes.util.find_library("m")
    return name if name and os.path.isfile(name) else None


def test_required_function_counts_match_the_standard_headers():
    v2 = set().union(*REQUIRED_FUNCTIONS["2"].values())
    v3 = set().union(*REQUIRED_FUNCTIONS["3"].values())
    assert len(v2) == 44 and len(v3) == 75
    # interface groups are disjoint
    for v in ("2", "3"):
        groups = list(REQUIRED_FUNCTIONS[v].values())
        assert sum(len(g) for g in groups) == len(set().union(*groups))


@pytest.mark.skipif(not (_system_elf() and shutil.which("nm")), reason="needs an ELF64 libm and binutils nm")
def test_elf_parser_agrees_with_nm():
    lib = _system_elf()
    mine = set(elf_exported_functions(Path(lib).read_bytes()))
    out = subprocess.run(["nm", "-D", "--defined-only", lib], capture_output=True, text=True).stdout
    ref = {l.split()[-1].split("@")[0] for l in out.splitlines() if len(l.split()) >= 3 and l.split()[1] in ("T", "W", "i")}
    assert mine == ref and len(mine) > 100


def _fmu(tmp_path: Path, md: str, binary: bytes | None, ident: str = "M") -> Path:
    p = tmp_path / "m.fmu"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("modelDescription.xml", md)
        if binary is not None:
            z.writestr(f"binaries/linux64/{ident}.so", binary)
    return p


MD2 = """<?xml version="1.0"?>
<fmiModelDescription fmiVersion="2.0" modelName="m" guid="{x}">
  <CoSimulation modelIdentifier="M"/>
  <ModelVariables>
    <ScalarVariable name="a" valueReference="1" causality="output"><Real/></ScalarVariable>
    <ScalarVariable name="b" valueReference="1" causality="parameter"><Real start="1"/></ScalarVariable>
  </ModelVariables>
  <ModelStructure><Outputs><Unknown index="2"/></Outputs></ModelStructure>
</fmiModelDescription>"""


@pytest.mark.skipif(not _system_elf(), reason="needs an ELF64 library to use as a fake FMU binary")
def test_broken_fmu_is_reported(tmp_path):
    rep = inspect_fmu(_fmu(tmp_path, MD2, Path(_system_elf()).read_bytes()))
    assert not rep.ok
    (b,) = rep.binaries
    assert b.status == "missing-functions"
    expected = REQUIRED_FUNCTIONS["2"]["common"] | REQUIRED_FUNCTIONS["2"]["CS"]
    assert set(b.missing) == expected                    # libm exports none of them
    assert any("duplicate valueReference" in p for p in rep.problems)
    assert any("not a causality='output'" in p for p in rep.problems)


def test_missing_binary_and_unknown_format(tmp_path):
    rep = inspect_fmu(_fmu(tmp_path, MD2.replace('valueReference="1" causality="parameter"',
                                                 'valueReference="2" causality="parameter"')
                           .replace('index="2"', 'index="1"'), None))
    assert rep.problems == ["no binary for any modelIdentifier and no sources/"]
    rep = inspect_fmu(_fmu(tmp_path, MD2, b"MZ\x90\x00not-really-a-dll"))
    assert rep.binaries[0].status == "unchecked" and rep.binaries[0].format == "pe"


@pytest.mark.skipif(not os.environ.get("PINNEAPPLE_REFERENCE_FMUS"),
                    reason="set PINNEAPPLE_REFERENCE_FMUS to a folder of built Modelica Reference-FMUs")
def test_modelica_reference_fmus_are_clean():
    fmus = sorted(glob.glob(os.path.join(os.environ["PINNEAPPLE_REFERENCE_FMUS"], "**", "*.fmu"), recursive=True))
    assert fmus
    for f in fmus:
        rep = inspect_fmu(f)
        assert rep.ok, rep.summary()
        assert all(b.status == "ok" for b in rep.binaries), rep.summary()


@pytest.mark.skipif(not (os.environ.get("PINNEAPPLE_REFERENCE_FMUS") and os.environ.get("REA_PWNTOOLS_PYTHON")
                         and (os.environ.get("PINNEAPPLE_REA_BIN") or shutil.which("rea"))),
                    reason="needs built Reference-FMUs, an installed REA and REA_PWNTOOLS_PYTHON")
def test_exported_symbols_agree_with_rea(tmp_path):
    """Independent cross-check: REA's ELF layout (pwntools/pyelftools) must
    report the same exported function set as fmu_inspect's own parser."""
    from pinneapple_tools.rea_bridge import ReaClient, exported_functions_from_layout
    client = ReaClient()
    fmus = sorted(glob.glob(os.path.join(os.environ["PINNEAPPLE_REFERENCE_FMUS"], "**", "*.fmu"), recursive=True))
    for f in fmus:
        for b in inspect_fmu(f).binaries:
            if b.exported is None:
                continue
            so = zipfile.ZipFile(f).extract(b.path, tmp_path / Path(f).stem)
            rea = exported_functions_from_layout(client.inspect_binary_layout(so).payload)
            assert rea == b.exported, f
