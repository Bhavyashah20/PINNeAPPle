"""A protein on a gold surface: orient, build the Au(111) slab, run a short OpenMM MD and visualise it.

    python examples/molecular/protein_on_gold.py path/to/protein.pdb [--ps 20] [--out out_dir]

Steps (each logged by ``StepLog``):

1. repair the structure with pdbfixer (missing atoms, hydrogens at pH 7; heterogens and water removed);
2. orient it with its broadest face toward the surface (principal axes) and place it 3 A above the slab;
3. build an Au(111) slab covering its footprint plus 6 A, in whole lattice repeats;
4. Amber14 protein with GBn2 implicit solvent; the gold acts through a 12-6 Lennard-Jones interaction with every
   protein atom (Lorentz-Berthelot mixing; gold sigma 2.629 A, epsilon 5.29 kcal/mol from the INTERFACE force field,
   Heinz et al., J. Phys. Chem. C 112, 17281 (2008)), summed over the fcc lattice as Steele's 10-4-3 wall (flat:
   no lateral corrugation; explicit gold atoms, as metal-surface pipelines such as RoyalMD Gold use, are kept for
   the pictures and cost far more on a CPU);
5. minimise, then Langevin dynamics at 300 K, 2 fs, saving a frame every 0.5 ps;
6. visualise: stills (space-filling and backbone), an MP4 of the trajectory with live charts, an MD dashboard
   (energy, temperature, RMSD, height above the surface, contacts) and a glTF scene for the web viewer.

The run is a short demonstration on CPU, not an equilibrated adsorption study: tens of picoseconds show thermal
motion and the first contacts, not the adsorbed state. Needs ``openmm``, ``pdbfixer``, ``pyvista`` and
``imageio-ffmpeg``.
"""
import argparse
import os

import numpy as np

from pinneapple_tools.visualization.molecular import (
    Molecule,
    StepLog,
    contacts,
    encode_mp4,
    height_above,
    md_dashboard,
    orient_on_surface,
    render,
    render_frames,
    rmsd,
    slab_for_footprint,
    to_glb,
)

AU_SIGMA_NM, AU_EPS_KJ = 0.2629, 5.29 * 4.184


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb")
    ap.add_argument("--ps", type=float, default=20.0, help="production length (ps)")
    ap.add_argument("--frame-ps", type=float, default=0.5)
    ap.add_argument("--out", default="protein_on_gold")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import openmm as mm
    from openmm import app, unit
    from pdbfixer import PDBFixer

    log = StepLog("Protein on Au(111): OpenMM MD and visualisation")
    log.step("Repairing the structure (pdbfixer)")
    fixer = PDBFixer(filename=args.pdb)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.missingResidues = {}                                      # do not build missing loops
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    n_res = sum(1 for _ in fixer.topology.residues())
    log.ok(f"{fixer.topology.getNumAtoms()} atoms, {n_res} residues")

    log.step("Orienting on the surface and building the Au(111) slab")
    xyz = np.array(fixer.positions.value_in_unit(unit.angstrom))
    masses = np.array([a.element.mass.value_in_unit(unit.dalton) for a in fixer.topology.atoms()])
    X = orient_on_surface(xyz, masses, gap=3.0)
    slab, cell, (nx, ny) = slab_for_footprint(X, "Au", "111", padding=6.0, layers=3)
    log.info(f"slab {nx} x {ny} cells, {len(slab)} atoms, {cell[0, 0]:.1f} x {np.linalg.norm(cell[1]):.1f} A")

    log.step("Force field: Amber14 + GBn2; gold as a Steele 10-4-3 Lennard-Jones wall")
    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = ff.createSystem(fixer.topology, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=1.2 * unit.nanometer,
                             constraints=app.HBonds)
    n_prot = system.getNumParticles()
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    a_nm = 0.4078
    rho, delta = 4.0 / a_nm ** 3, a_nm / np.sqrt(3.0)                # gold number density, (111) layer spacing
    z_top = float(slab[:, 2].max()) * 0.1
    wall = mm.CustomExternalForce(
        "6.283185307179586*eps*rho*sig^2*delta*(0.4*(sig/zz)^10 - (sig/zz)^4 - sig^4/(3*delta*(zz+0.61*delta)^3));"
        "zz=max(z-ztop, 0.08); sig=0.5*(sig_i+sig_au); eps=sqrt(eps_i*eps_au)")
    for name, val in (("rho", rho), ("delta", delta), ("ztop", z_top), ("sig_au", AU_SIGMA_NM), ("eps_au", AU_EPS_KJ)):
        wall.addGlobalParameter(name, val)
    wall.addPerParticleParameter("sig_i")
    wall.addPerParticleParameter("eps_i")
    for i in range(n_prot):
        _, s, e = nb.getParticleParameters(i)
        wall.addParticle(i, [s.value_in_unit(unit.nanometer), e.value_in_unit(unit.kilojoule_per_mole)])
    system.addForce(wall)
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    sim = app.Simulation(fixer.topology, system, integ, mm.Platform.getPlatformByName("CPU"))
    sim.context.setPositions(X * 0.1 * unit.nanometer)
    log.ok(f"{n_prot} particles; Au(111) wall at z = {10 * z_top:.1f} A (sigma 2.629 A, epsilon 5.29 kcal/mol)")

    log.step("Minimising and running Langevin dynamics at 300 K")
    sim.minimizeEnergy(tolerance=10 * unit.kilojoule_per_mole / unit.nanometer, maxIterations=200)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    steps_per_frame = int(round(args.frame_ps / 0.002))
    n_frames = int(round(args.ps / args.frame_ps))
    n_cons = system.getNumConstraints()
    dof = 3 * n_prot - n_cons
    frames, t, epot, temp = [], [], [], []
    for k in range(n_frames + 1):
        if k:
            sim.step(steps_per_frame)
        st = sim.context.getState(getPositions=True, getEnergy=True)
        P = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)[:n_prot]
        frames.append(np.array(P))
        t.append(k * args.frame_ps)
        epot.append(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        ke = st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        temp.append(2 * ke / (dof * 0.0083144626))
        if k % max(1, n_frames // 5) == 0:
            log.info(f"t = {t[-1]:5.1f} ps   E_pot = {epot[-1]:10.0f} kJ/mol   T = {temp[-1]:5.0f} K")
    log.ok(f"{n_frames} frames, {args.ps:.1f} ps")

    log.step("Visualising")
    names = [a.name for a in fixer.topology.atoms()]
    mol = Molecule(frames[0], [a.element.symbol.upper() for a in fixer.topology.atoms()], names,
                   [a.residue.name for a in fixer.topology.atoms()], [int(a.residue.id) for a in fixer.topology.atoms()],
                   [a.residue.chain.id for a in fixer.topology.atoms()])
    top_z = slab[:, 2].max()
    ca = [i for i, n in enumerate(names) if n == "CA"]
    r = rmsd([f[ca] for f in frames], frames[0][ca])
    h = height_above(frames, top_z, masses)
    c = contacts(frames, top_z)
    render(mol, os.path.join(args.out, "still_spheres.png"), xyz=frames[-1], slab=slab,
           title=f"Carbonic anhydrase II on Au(111) - {args.ps:.0f} ps")
    render(mol, os.path.join(args.out, "still_backbone.png"), xyz=frames[-1], slab=slab, style="cartoon")
    md_dashboard(os.path.join(args.out, "md_dashboard.png"), t, potential=epot, temperature=temp, rmsd_A=r,
                 height_A=h, contacts_n=c, target_temperature=300, title="Protein on Au(111), GBn2 implicit solvent")
    pngs = render_frames(mol, frames, os.path.join(args.out, "frames"), slab=slab, orbit_deg=60.0,
                         labels=[f"{x:.1f} ps" for x in t])
    encode_mp4(pngs, os.path.join(args.out, "trajectory.mp4"), fps=10)
    to_glb(mol, os.path.join(args.out, "scene.glb"), xyz=frames[-1], slab=slab)
    log.ok(f"RMSD(CA) {r[-1]:.2f} A, height {h[0]:.1f} -> {h[-1]:.1f} A, contacts {c[0]} -> {c[-1]}")
    log.ok(f"wrote {args.out}/: still_spheres.png, still_backbone.png, md_dashboard.png, trajectory.mp4, scene.glb")


if __name__ == "__main__":
    main()
