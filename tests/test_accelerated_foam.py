"""pinneapple_simulation.accelerated_foam: LDU systems, solvers with OpenFOAM's stopping test, warm starts,
the guess-server protocol, and (when OpenFOAM and its headers are present) the pinneWarmStart wrapper
against plain OpenFOAM on a small lid-driven cavity."""

from __future__ import annotations

import os
import struct
import threading

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")

from pinneapple_simulation.accelerated_foam import case as fcase  # noqa: E402
from pinneapple_simulation.accelerated_foam import ldu, solvers, warm_start  # noqa: E402


def poisson(n: int, seed: int = 0, dirichlet_cell: bool = True):
    """Negative-definite 5-point Laplacian on an n x n grid with random positive face coefficients
    (a stand-in for OpenFOAM's laplacian(rAU, p)); one cell pinned like pRefCell."""
    rng = np.random.default_rng(seed)
    N = n * n
    rows, cols, vals = [], [], []
    idx = np.arange(N).reshape(n, n)
    for a, b in [(idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])]:
        a, b = a.ravel(), b.ravel()
        w = rng.uniform(0.5, 2.0, a.size)
        rows += [a, b]
        cols += [b, a]
        vals += [w, w]
    rows, cols, vals = map(np.concatenate, (rows, cols, vals))
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    d = -np.asarray(A.sum(axis=1)).ravel()
    if dirichlet_cell:
        d[0] -= 1.0
    return (A + sp.diags(d)).tocsr()


def test_ldu_roundtrip_and_products():
    A = poisson(12)
    b = np.random.default_rng(1).normal(size=A.shape[0])
    s = ldu.from_csr(A, b)
    assert s.n_cells == 144 and s.n_faces == 2 * 12 * 11
    assert np.all(s.lower_addr < s.upper_addr)
    x = np.random.default_rng(2).normal(size=s.n_cells)
    assert np.allclose(s.amul(x), A @ x)
    assert np.allclose(s.amul_ldu(x), A @ x)
    assert abs(s.to_csr() - A).max() == 0


def test_normalised_residual_matches_openfoam_definition():
    A = poisson(8)
    b = np.ones(A.shape[0])
    s = ldu.from_csr(A, b)
    x = np.linspace(0, 1, s.n_cells)
    Ax = A @ x
    tmp = np.asarray(A.sum(axis=1)).ravel() * x.mean()
    nf = np.sum(np.abs(Ax - tmp) + np.abs(b - tmp)) + 1e-20
    assert ldu.normalised_residual(s, x) == pytest.approx(np.sum(np.abs(b - Ax)) / nf, rel=1e-12)


@pytest.mark.parametrize("prec", ["jacobi", "dic"])
def test_pcg_converges_to_direct_solution(prec):
    from scipy.sparse.linalg import spsolve

    A = poisson(20)
    b = np.random.default_rng(3).normal(size=A.shape[0])
    s = ldu.from_csr(A, b)
    res = solvers.pcg(s, None, prec, tolerance=1e-12)
    assert res.converged
    xd = spsolve(A.tocsc(), b)
    assert np.linalg.norm(res.x - xd) / np.linalg.norm(xd) < 1e-8


def test_dic_needs_fewer_iterations_than_jacobi():
    s = ldu.from_csr(poisson(24), np.random.default_rng(4).normal(size=576))
    it_j = solvers.pcg(s, None, "jacobi", 1e-8).iterations
    it_d = solvers.pcg(s, None, "dic", 1e-8).iterations
    assert it_d < it_j


def test_amg_preconditioner_if_available():
    pytest.importorskip("pyamg")
    s = ldu.from_csr(poisson(32), np.random.default_rng(5).normal(size=1024))
    r = solvers.pcg(s, None, "amg", 1e-8)
    assert r.converged and r.iterations < solvers.pcg(s, None, "dic", 1e-8).iterations


def _sequence(n=16, steps=12, seed=6):
    """Systems whose matrix and right-hand side drift slowly, like successive pressure corrections."""
    A0 = poisson(n, seed)
    rng = np.random.default_rng(seed + 1)
    b0, b1, b2 = (rng.normal(size=n * n) for _ in range(3))
    out = []
    for t in range(steps):
        A = A0 * (1.0 + 0.01 * np.sin(t))
        b = b0 + 0.3 * t * b1 + 0.02 * t * t * b2
        out.append(ldu.from_csr(A, b))
    return out


def test_projection_warm_start_is_never_worse_in_energy_norm_and_saves_iterations():
    from scipy.sparse.linalg import spsolve

    systems = _sequence()
    ws = warm_start.ProjectionWarmStart(6)
    x_prev = np.zeros(systems[0].n_cells)
    it_plain = it_warm = 0
    for s in systems:
        A = s.to_csr()
        xs = spsolve(A.tocsc(), s.source)
        g = ws.guess(s, x_prev)

        def enorm(e, A=A):
            return float(-(e @ (A @ e)))  # A is negative definite

        assert enorm(g - xs) <= enorm(x_prev - xs) * (1 + 1e-9) + 1e-12
        it_plain += solvers.pcg(s, x_prev, "dic", 1e-8).iterations
        r = solvers.pcg(s, g, "dic", 1e-8)
        it_warm += r.iterations
        ws.update(r.x)
        x_prev = r.x
    assert it_warm < 0.7 * it_plain


def test_galerkin_drops_dependent_columns():
    s = _sequence(steps=1)[0]
    v = np.random.default_rng(0).normal(size=s.n_cells)
    x = warm_start.galerkin(s, np.stack([v, 2 * v, v], axis=1))
    assert x is not None and np.all(np.isfinite(x))


def test_pod_warm_start_from_other_runs():
    from scipy.sparse.linalg import spsolve

    train = _sequence(seed=6, steps=20)
    snaps = np.stack([spsolve(s.to_csr().tocsc(), s.source) for s in train], axis=1)
    pod = warm_start.PODWarmStart(energy=0.999999, max_rank=10).fit(snaps)
    assert 1 <= pod.rank <= 10
    s = train[7]
    x_prev = np.zeros(s.n_cells)
    g = pod.guess(s, x_prev)
    assert ldu.normalised_residual(s, g) < 0.1 * ldu.normalised_residual(s, x_prev)


def test_torch_pcg_matches_numpy():
    pytest.importorskip("torch")
    s = ldu.from_csr(poisson(16), np.random.default_rng(7).normal(size=256))
    a = solvers.pcg(s, None, "jacobi", 1e-10)
    b = solvers.torch_pcg(s, None, 1e-10, device="cpu")
    c = solvers.torch_pcg(s, None, 1e-6, device="cpu", dtype="float32")
    assert abs(a.iterations - b.iterations) <= 1
    assert np.linalg.norm(a.x - b.x) / np.linalg.norm(a.x) < 1e-8
    assert c.converged and c.final_residual < 1e-6


def test_learned_warm_start_is_safeguarded():
    pytest.importorskip("torch")
    from scipy.sparse.linalg import spsolve

    from pinneapple_simulation.accelerated_foam.learned import LearnedWarmStart

    systems = _sequence(n=10, steps=10)
    x = np.zeros(systems[0].n_cells)
    for s in systems:
        s.psi0 = x
        s.psi = spsolve(s.to_csr().tocsc(), s.source)
        x = s.psi
    lw = LearnedWarmStart(n_snapshots=4, hidden=8, layers=2)
    losses = lw.fit(systems[:6], epochs=15, lr=1e-2)
    assert losses[-1] <= losses[0] + 1e-12
    proj = warm_start.ProjectionWarmStart(4)
    for s in systems[:6]:
        proj.update(s.psi)
        lw.update(s.psi)
    s = systems[6]
    g, gp = lw.guess(s, s.psi0), proj.guess(s, s.psi0)
    assert np.linalg.norm(s.source - s.amul(g)) <= np.linalg.norm(s.source - s.amul(gp)) * (
        1 + 1e-12
    )


def test_parse_log():
    log = (
        "Time = 0.1\nDICPCG:  Solving for p, Initial residual = 1, Final residual = 9e-07, No Iterations 35\n"
        "pinneWarmStartDICPCG:  Solving for p, Initial residual = 0.01, Final residual = 8e-07, No Iterations 12\n"
        "smoothSolver:  Solving for Ux, Initial residual = 0.1, Final residual = 1e-09, No Iterations 4\n"
        "ExecutionTime = 0.5 s  ClockTime = 1 s\nExecutionTime = 0.75 s  ClockTime = 1 s\n"
    )
    solves, t = fcase.parse_log(log)
    assert (
        solves["p"].n_solves == 2
        and solves["p"].iterations == 47
        and solves["p"].per_solve == [35, 12]
    )
    assert solves["Ux"].iterations == 4 and t == 0.75


def test_write_cavity(tmp_path):
    c = fcase.write_cavity(
        tmp_path / "cav",
        n=8,
        end_time=0.05,
        p_solver=fcase.warm_solver("GAMG", 1e-6, 0.05, dumpDir="dump"),
        libs=("/x/lib.so",),
    )
    fv = (c / "system/fvSolution").read_text()
    assert "pinneWarmStart" in fv and "innerSolver     GAMG" in fv and "dumpDir         dump;" in fv
    assert 'libs ("/x/lib.so");' in (c / "system/controlDict").read_text()
    assert "(8 8 1)" in (c / "system/blockMeshDict").read_text()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes")
def test_guess_server_protocol(tmp_path):
    """Play the OpenFOAM side of the pipe protocol and check the guess and the solution message."""
    from pinneapple_simulation.accelerated_foam.guess_server import GuessServer

    s = ldu.from_csr(poisson(6), np.random.default_rng(8).normal(size=36))
    seen = []

    class Echo:
        def guess(self, system, x_prev):
            assert np.allclose(system.amul(x_prev), s.amul(x_prev))
            return x_prev + 1.0

        def update(self, x):
            seen.append(x.copy())

    psi = np.arange(36, dtype=float)
    with GuessServer(tmp_path / "pipes", {"p": Echo()}) as srv:

        def send(kind, payload):
            with open(srv.request, "wb") as f:
                f.write(
                    struct.pack("<5q", kind, 0, s.n_cells, s.n_faces if kind == 1 else 0, 1)
                    + b"p"
                    + payload
                )

        body = (
            s.lower_addr.astype("<i4").tobytes()
            + s.upper_addr.astype("<i4").tobytes()
            + s.diag.astype("<f8").tobytes()
            + s.upper.astype("<f8").tobytes()
            + s.source.astype("<f8").tobytes()
            + psi.astype("<f8").tobytes()
        )
        t = threading.Thread(target=send, args=(1, body))
        t.start()
        with open(srv.reply, "rb") as r:
            status = struct.unpack("<q", r.read(8))[0]
            guess = np.frombuffer(r.read(8 * 36), "<f8")
        t.join()
        assert status == 1 and np.allclose(guess, psi + 1)
        send(2, (2 * psi).astype("<f8").tobytes())
        for _ in range(200):
            if seen:
                break
            threading.Event().wait(0.01)
    assert seen and np.allclose(seen[0], 2 * psi)
    assert srv.stats["p"]["guesses"] == 1 and srv.stats["p"]["solutions"] == 1


# ------------------------------------------------------------------------------------------ with OpenFOAM


def _plugin_or_skip():
    if not fcase.openfoam_available("icoFoam"):
        pytest.skip("OpenFOAM not installed")
    from pinneapple_simulation.accelerated_foam.build import find_plugin

    lib = find_plugin()
    if lib is None:
        pytest.skip(
            "pinneWarmStart cannot be built (no OpenFOAM headers: set FOAM_SRC or PINNEAPPLE_OPENFOAM_SRC)"
        )
    return lib


def test_openfoam_dump_is_reproduced_by_python(tmp_path):
    lib = _plugin_or_skip()
    c = fcase.write_cavity(
        tmp_path / "dump",
        n=16,
        end_time=0.03,
        p_solver=fcase.warm_solver("PCG", 1e-6, 0.0, guess="none", dumpDir="systems"),
        libs=(str(lib),),
    )
    run = fcase.run_case(c)
    systems = ldu.read_dump(c / "systems")
    assert len(systems) == run.solves["p"].n_solves > 0
    for s in systems[:4]:
        assert ldu.normalised_residual(s, s.psi0) == pytest.approx(
            s.meta["initialResidualPrevious"], rel=1e-9
        )
        r = solvers.pcg(s, s.psi0, "dic", 1e-6, 0.0)
        assert r.iterations == s.meta["iterations"]
        assert np.linalg.norm(r.x - s.psi) <= 1e-10 * np.linalg.norm(s.psi)


def test_openfoam_warm_start_same_solution_fewer_iterations(tmp_path):
    lib = _plugin_or_skip()
    kw = dict(n=32, end_time=0.1, libs=(str(lib),))
    tight = fcase.plain_solver("PCG", 1e-11, 0.0)
    ref = fcase.write_cavity(tmp_path / "ref", p_solver=tight, p_final_solver=tight, **kw)
    plain = fcase.write_cavity(
        tmp_path / "plain", p_solver=fcase.plain_solver("PCG", 1e-6, 0.05), **kw
    )
    warm = fcase.write_cavity(
        tmp_path / "warm", p_solver=fcase.warm_solver("PCG", 1e-6, 0.05), **kw
    )
    fcase.run_case(ref)
    rp, rw = fcase.run_case(plain), fcase.run_case(warm)
    assert rw.solves["p"].n_solves == rp.solves["p"].n_solves
    assert rw.solves["p"].iterations < 0.8 * rp.solves["p"].iterations
    # both runs are within solver tolerance of the tightly converged reference, to the same order
    dp_plain, dp_warm = (fcase.field_difference(ref, d, "p") for d in (plain, warm))
    dU_plain, dU_warm = (fcase.field_difference(ref, d, "U") for d in (plain, warm))
    assert dU_plain < 1e-4 and dp_plain < 1e-2  # plain OpenFOAM at tolerance 1e-6 vs 1e-11
    assert dU_warm < 3 * dU_plain + 1e-9 and dp_warm < 3 * dp_plain + 1e-9
