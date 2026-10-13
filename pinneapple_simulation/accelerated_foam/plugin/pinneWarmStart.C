/*---------------------------------------------------------------------------*\
    pinneWarmStart: a linear-solver wrapper for OpenFOAM's lduMatrix.

    Part of PINNeAPPle (Apache-2.0). This file is original code written
    against OpenFOAM's public lduMatrix::solver interface; it contains no
    OpenFOAM source. A shared library built from it and linked against
    OpenFOAM is subject to OpenFOAM's license (GPL-3.0-or-later) when
    distributed in binary form.

    The wrapper changes only the initial guess handed to an ordinary inner
    OpenFOAM solver (PCG, GAMG, ...). The inner solver and its stopping test
    are unchanged, so the converged solution satisfies the same tolerance as
    a plain run.

    fvSolution entry:

        p
        {
            solver          pinneWarmStart;
            innerSolver     PCG;          // any symmetric lduMatrix solver
            preconditioner  DIC;          // passed to the inner solver
            tolerance       1e-6;
            relTol          0;
            guess           projection;   // none | projection | external
            nSnapshots      8;            // projection: past solutions kept
            relTolReference previous;     // previous | guess
            safeguard       true;         // keep the better of guess / psi
            snapshots       all;          // all | converged (relTol 0 solves only)
            applyTo         all;          // all | converged: guess only for relTol 0 solves
            dumpDir         "";           // write every system to disk
            pipeDir         "";           // external: FIFOs request/reply
            log             false;
        }

    guess projection: the Galerkin projection of the new system onto the
    span of the last nSnapshots solutions of the same field (Fischer 1998,
    CMAME 163:193-204). With the current matrix A and right-hand side b,
    x0 = Q (Q^T A Q)^-1 Q^T b, Q an orthonormal basis of the snapshots and
    of the incoming psi. It costs nSnapshots matrix-vector products.

    guess external: the system is written to <pipeDir>/request and the
    guess is read from <pipeDir>/reply (named pipes served by Python).

    relTolReference previous: relTol is measured against the residual of
    the incoming psi (the guess a plain run starts from), so a run with a
    warm start stops at the same absolute residual a plain run would.
\*---------------------------------------------------------------------------*/

#include "lduMatrix.H"
#include "addToRunTimeSelectionTable.H"
#include "OSspecific.H"

#include <cstdint>
#include <cmath>
#include <deque>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace Foam
{

class pinneWarmStart
:
    public lduMatrix::solver
{
    // Past solutions per field name (and per processor in parallel)
    static std::map<std::string, std::deque<std::vector<double>>> history_;
    static std::map<std::string, label> counter_;

    word innerSolver_;
    word guess_;
    label nSnapshots_;
    word relTolReference_;
    bool safeguard_;
    word snapshots_;
    word applyTo_;
    fileName dumpDir_;
    fileName pipeDir_;
    bool log_;

    // A x for a plain array
    void amul(scalarField& y, const scalarField& x, const direction cmpt) const
    {
        matrix_.Amul(y, x, interfaceBouCoeffs_, interfaces_, cmpt);
    }

    // Galerkin projection onto span{snapshots, psi}; sets x0 and A x0.
    // False when there is nothing to project on.
    bool projectionGuess
    (
        scalarField& x0,
        scalarField& Ax0,
        const scalarField& psi,
        const scalarField& Apsi,
        const scalarField& b,
        const direction cmpt
    ) const;

    // Exchange with an external process through named pipes
    bool externalGuess
    (
        scalarField& x0,
        const scalarField& psi,
        const scalarField& b,
        const label count
    ) const;

    void reportSolution(const scalarField& psi, const label count) const;

    void dump
    (
        const scalarField& psi0,
        const scalarField& psi,
        const scalarField& b,
        const label count,
        const solverPerformance& perf,
        const scalar r0Previous
    ) const;


public:

    TypeName("pinneWarmStart");

    pinneWarmStart
    (
        const word& fieldName,
        const lduMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
    )
    :
        lduMatrix::solver
        (
            fieldName,
            matrix,
            interfaceBouCoeffs,
            interfaceIntCoeffs,
            interfaces,
            solverControls
        ),
        innerSolver_(solverControls.lookupOrDefault<word>("innerSolver", "PCG")),
        guess_(solverControls.lookupOrDefault<word>("guess", "projection")),
        nSnapshots_(solverControls.lookupOrDefault<label>("nSnapshots", 8)),
        relTolReference_
        (
            solverControls.lookupOrDefault<word>("relTolReference", "previous")
        ),
        safeguard_(solverControls.lookupOrDefault<bool>("safeguard", true)),
        snapshots_(solverControls.lookupOrDefault<word>("snapshots", "all")),
        applyTo_(solverControls.lookupOrDefault<word>("applyTo", "all")),
        dumpDir_(solverControls.lookupOrDefault<fileName>("dumpDir", "")),
        pipeDir_(solverControls.lookupOrDefault<fileName>("pipeDir", "")),
        log_(solverControls.lookupOrDefault<bool>("log", false))
    {}

    virtual ~pinneWarmStart() = default;

    virtual solverPerformance solve
    (
        scalarField& psi,
        const scalarField& source,
        const direction cmpt = 0
    ) const;
};


std::map<std::string, std::deque<std::vector<double>>>
    pinneWarmStart::history_;
std::map<std::string, label> pinneWarmStart::counter_;

defineTypeNameAndDebug(pinneWarmStart, 0);

lduMatrix::solver::addsymMatrixConstructorToTable<pinneWarmStart>
    addpinneWarmStartSymMatrixConstructorToTable_;


bool pinneWarmStart::projectionGuess
(
    scalarField& x0,
    scalarField& Ax0,
    const scalarField& psi,
    const scalarField& Apsi,
    const scalarField& b,
    const direction cmpt
) const
{
    const label n = psi.size();
    const label comm = matrix_.mesh().comm();

    // Basis: the stored solutions, then psi unless it repeats the last one
    std::vector<scalarField> X, AX;
    const auto it = history_.find(fieldName_);
    if (it != history_.end())
    {
        for (const std::vector<double>& s : it->second)
        {
            if (label(s.size()) != n) continue;
            X.emplace_back(n);
            scalarField& v = X.back();
            forAll(v, i) { v[i] = s[i]; }
        }
    }
    bool psiNew = true;
    if (!X.empty())
    {
        const scalarField& last = X.back();
        scalar d = 0;
        forAll(psi, i) { d = max(d, mag(psi[i] - last[i])); }
        psiNew = returnReduce(d, maxOp<scalar>(), Pstream::msgType(), comm) > 0;
    }
    for (const scalarField& v : X)
    {
        AX.emplace_back(n);
        matrix_.Amul(AX.back(), v, interfaceBouCoeffs_, interfaces_, cmpt);
    }
    if (psiNew)
    {
        X.push_back(psi);
        AX.push_back(Apsi);
    }

    const label k = X.size();
    if (k == 0) return false;

    // G = X^T A X (symmetric), r = X^T b
    std::vector<double> G(k*k), r(k);
    for (label i = 0; i < k; ++i)
    {
        r[i] = gSumProd(X[i], b, comm);
        for (label j = 0; j <= i; ++j)
        {
            G[i*k + j] = G[j*k + i] = gSumProd(X[i], AX[j], comm);
        }
    }

    // Symmetric elimination with diagonal pivoting; columns whose pivot is
    // negligible (linearly dependent snapshots) are dropped.
    std::vector<label> order;
    std::vector<bool> active(k, true);
    double dmax = 0;
    for (label i = 0; i < k; ++i) dmax = std::max(dmax, std::fabs(G[i*k + i]));
    if (dmax <= 0) return false;
    std::vector<double> M(G), y(r);
    for (label step = 0; step < k; ++step)
    {
        label p = -1;
        for (label i = 0; i < k; ++i)
        {
            if (active[i] && (p < 0 || std::fabs(M[i*k + i]) > std::fabs(M[p*k + p])))
            {
                p = i;
            }
        }
        if (p < 0 || std::fabs(M[p*k + p]) <= 1e-12*dmax) break;
        active[p] = false;
        order.push_back(p);
        for (label i = 0; i < k; ++i)
        {
            if (!active[i]) continue;
            const double f = M[i*k + p]/M[p*k + p];
            for (label j = 0; j < k; ++j) M[i*k + j] -= f*M[p*k + j];
            y[i] -= f*y[p];
        }
    }
    if (order.empty()) return false;

    // Back substitution over the chosen pivots (dropped columns get 0)
    std::vector<double> coef(k, 0.0);
    for (label o = label(order.size()) - 1; o >= 0; --o)
    {
        const label p = order[o];
        double s = y[p];
        for (label q = o + 1; q < label(order.size()); ++q)
        {
            s -= M[p*k + order[q]]*coef[order[q]];
        }
        coef[p] = s/M[p*k + p];
    }

    x0 = 0;
    Ax0 = 0;
    for (label i = 0; i < k; ++i)
    {
        if (coef[i] == 0) continue;
        x0 += coef[i]*X[i];
        Ax0 += coef[i]*AX[i];
    }
    return true;
}


namespace
{
    template<class T>
    void writeRaw(std::ostream& os, const T* data, const std::size_t n)
    {
        os.write(reinterpret_cast<const char*>(data), n*sizeof(T));
    }

    void writeI64(std::ostream& os, const int64_t v)
    {
        writeRaw(os, &v, 1);
    }

    void writeField(std::ostream& os, const scalarField& f)
    {
        std::vector<double> buf(f.size());
        forAll(f, i) { buf[i] = f[i]; }
        writeRaw(os, buf.data(), buf.size());
    }

    void writeLabels(std::ostream& os, const labelUList& f)
    {
        std::vector<int32_t> buf(f.size());
        forAll(f, i) { buf[i] = int32_t(f[i]); }
        writeRaw(os, buf.data(), buf.size());
    }
}


bool pinneWarmStart::externalGuess
(
    scalarField& x0,
    const scalarField& psi,
    const scalarField& b,
    const label count
) const
{
    if (pipeDir_.empty()) return false;

    const lduAddressing& addr = matrix_.lduAddr();
    {
        std::ofstream req(pipeDir_/"request", std::ios::binary);
        if (!req) return false;
        // Message 1: request a guess for this system
        writeI64(req, 1);
        writeI64(req, count);
        writeI64(req, psi.size());
        writeI64(req, addr.lowerAddr().size());
        writeI64(req, fieldName_.size());
        req.write(fieldName_.data(), fieldName_.size());
        writeLabels(req, addr.lowerAddr());
        writeLabels(req, addr.upperAddr());
        writeField(req, matrix_.diag());
        writeField(req, matrix_.upper());
        writeField(req, b);
        writeField(req, psi);
    }

    std::ifstream rep(pipeDir_/"reply", std::ios::binary);
    if (!rep) return false;
    int64_t status = 0;
    rep.read(reinterpret_cast<char*>(&status), sizeof(status));
    if (!rep || status != 1) return false;
    std::vector<double> buf(psi.size());
    rep.read(reinterpret_cast<char*>(buf.data()), buf.size()*sizeof(double));
    if (!rep) return false;
    forAll(x0, i) { x0[i] = buf[i]; }
    return true;
}


void pinneWarmStart::reportSolution(const scalarField& psi, const label count)
const
{
    if (pipeDir_.empty()) return;
    std::ofstream req(pipeDir_/"request", std::ios::binary);
    if (!req) return;
    // Message 2: the converged solution of system `count`
    writeI64(req, 2);
    writeI64(req, count);
    writeI64(req, psi.size());
    writeI64(req, 0);
    writeI64(req, fieldName_.size());
    req.write(fieldName_.data(), fieldName_.size());
    writeField(req, psi);
}


void pinneWarmStart::dump
(
    const scalarField& psi0,
    const scalarField& psi,
    const scalarField& b,
    const label count,
    const solverPerformance& perf,
    const scalar r0Previous
) const
{
    if (dumpDir_.empty()) return;

    word tag = fieldName_ + "_" + Foam::name(count);
    if (Pstream::parRun())
    {
        tag += "_proc" + Foam::name(Pstream::myProcNo());
    }
    const fileName dir(dumpDir_/tag);
    mkDir(dir);

    const lduAddressing& addr = matrix_.lduAddr();
    label nCoupled = 0;
    forAll(interfaces_, i)
    {
        if (interfaces_.set(i)) ++nCoupled;
    }

    {
        std::ofstream os(dir/"system.bin", std::ios::binary);
        writeLabels(os, addr.lowerAddr());
        writeLabels(os, addr.upperAddr());
        writeField(os, matrix_.diag());
        writeField(os, matrix_.upper());
        writeField(os, b);
        writeField(os, psi0);
        writeField(os, psi);
    }
    {
        std::ofstream os(dir/"meta.txt");
        os.precision(17);
        os  << "field " << fieldName_ << "\n"
            << "count " << count << "\n"
            << "nCells " << psi.size() << "\n"
            << "nFaces " << addr.lowerAddr().size() << "\n"
            << "nCoupledInterfaces " << nCoupled << "\n"
            << "innerSolver " << innerSolver_ << "\n"
            << "guess " << guess_ << "\n"
            << "tolerance " << tolerance_ << "\n"
            << "relTol " << relTol_ << "\n"
            << "iterations " << perf.nIterations() << "\n"
            << "initialResidual " << perf.initialResidual() << "\n"
            << "finalResidual " << perf.finalResidual() << "\n"
            << "initialResidualPrevious " << r0Previous << "\n";
    }
}


solverPerformance pinneWarmStart::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    const label count = counter_[fieldName_]++;
    const scalarField psi0(psi);
    const label comm = matrix_.mesh().comm();
    const bool apply = (applyTo_ != "converged" || relTol_ == 0);

    // Residual of the guess a plain run would start from (not needed when
    // this solve is left to the inner solver unchanged)
    scalarField Apsi0(psi.size(), Zero);
    scalar res0 = 0;
    scalar r0Previous = -1;
    if (apply || !dumpDir_.empty())
    {
        matrix_.Amul(Apsi0, psi0, interfaceBouCoeffs_, interfaces_, cmpt);
        res0 = gSumMag((source - Apsi0)(), comm);
        scalarField tmp(psi.size());
        r0Previous = res0/this->normFactor(psi0, source, Apsi0, tmp);
    }

    scalarField x0(psi.size(), Zero);
    scalarField Ax0(psi.size(), Zero);
    bool haveGuess = false;
    if (guess_ != "none" && guess_ != "projection" && guess_ != "external")
    {
        FatalErrorInFunction
            << "Unknown guess " << guess_
            << " (none | projection | external)" << exit(FatalError);
    }
    if (apply && guess_ == "projection")
    {
        haveGuess = projectionGuess(x0, Ax0, psi0, Apsi0, source, cmpt);
    }
    else if (apply && guess_ == "external")
    {
        haveGuess = externalGuess(x0, psi0, source, count);
        if (haveGuess)
        {
            matrix_.Amul(Ax0, x0, interfaceBouCoeffs_, interfaces_, cmpt);
        }
    }

    bool used = false;
    if (haveGuess)
    {
        if (!safeguard_ || gSumMag((source - Ax0)(), comm) < res0)
        {
            psi = x0;
            used = true;
        }
    }

    // Inner solver: same controls, the stopping test of a plain run
    dictionary innerDict(controlDict_);
    innerDict.set("solver", innerSolver_);
    if (apply && relTolReference_ == "previous" && relTol_ > 0)
    {
        innerDict.set("tolerance", max(tolerance_, relTol_*r0Previous));
        innerDict.set("relTol", scalar(0));
    }

    autoPtr<lduMatrix::solver> inner = lduMatrix::solver::New
    (
        fieldName_,
        matrix_,
        interfaceBouCoeffs_,
        interfaceIntCoeffs_,
        interfaces_,
        innerDict
    );
    solverPerformance perf = inner->solve(psi, source, cmpt);
    perf.solverName() = typeName + perf.solverName();

    if (log_)
    {
        Info<< typeName << ": " << fieldName_ << " system " << count
            << " residual of previous psi " << r0Previous
            << ", of guess " << perf.initialResidual()
            << (used ? " (guess used)" : " (guess not used)") << endl;
    }

    // Keep the solution for the next projection
    if (guess_ == "projection" && (snapshots_ != "converged" || relTol_ == 0))
    {
        std::deque<std::vector<double>>& h = history_[fieldName_];
        h.emplace_back(psi.begin(), psi.end());
        while (label(h.size()) > nSnapshots_) h.pop_front();
    }
    else if (guess_ == "external")
    {
        reportSolution(psi, count);
    }

    dump(psi0, psi, source, count, perf, r0Previous);
    return perf;
}

} // End namespace Foam
