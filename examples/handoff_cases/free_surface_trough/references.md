# References

Each entry was checked against a publisher or library record before it was listed. A DOI is given only where the
record confirmed it.

## Volume of fluid and interFoam

1. Hirt, C. W., Nichols, B. D. (1981). Volume of fluid (VOF) method for the dynamics of free boundaries.
   *Journal of Computational Physics* 39(1), 201-225.
   DOI: [10.1016/0021-9991(81)90145-5](https://doi.org/10.1016/0021-9991(81)90145-5). The liquid-fraction (alpha)
   formulation of the free surface.
2. Brackbill, J. U., Kothe, D. B., Zemach, C. (1992). A continuum method for modeling surface tension.
   *Journal of Computational Physics* 100(2), 335-354. The continuum-surface-force model of surface tension used by
   interFoam.
3. Deshpande, S. S., Anumolu, L., Trujillo, M. F. (2012). Evaluating the performance of the two-phase flow solver
   interFoam. *Computational Science & Discovery* 5, 014016.
   DOI: [10.1088/1749-4699/5/1/014016](https://doi.org/10.1088/1749-4699/5/1/014016). Accuracy of interFoam's
   algebraic VOF (MULES) interface advection and its limits; why the interface Courant number is kept below 1.
4. Weller, H. G., Tabor, G., Jasak, H., Fureby, C. (1998). A tensorial approach to computational continuum mechanics
   using object-oriented techniques. *Computers in Physics* 12(6), 620-631.
   DOI: [10.1063/1.168744](https://doi.org/10.1063/1.168744). OpenFOAM; the release used here is v1912
   (<https://www.openfoam.com>).

## Turbulence modelling

5. Menter, F. R. (1994). Two-equation eddy-viscosity turbulence models for engineering applications.
   *AIAA Journal* 32(8), 1598-1605. DOI: [10.2514/3.12149](https://doi.org/10.2514/3.12149). The k-omega SST model.
6. Larsen, B. E., Fuhrman, D. R. (2018). On the over-production of turbulence beneath surface waves in
   Reynolds-averaged Navier-Stokes models. *Journal of Fluid Mechanics* 853, 419-460.
   DOI: [10.1017/jfm.2018.577](https://doi.org/10.1017/jfm.2018.577). Standard two-equation models over-produce k
   near a free surface in nearly potential flow; read the k levels near the interface in this package with this in
   mind.

## Plunging and impinging jets

7. Kiger, K. T., Duncan, J. H. (2012). Air-entrainment mechanisms in plunging jets and breaking waves.
   *Annual Review of Fluid Mechanics* 44, 563-596.
   DOI: [10.1146/annurev-fluid-122109-160724](https://doi.org/10.1146/annurev-fluid-122109-160724). Review of how a
   jet entering a pool entrains air; the CFD here resolves the cavity and splash only at the scale of the cells and
   does not model sub-grid air entrainment.
8. Biń, A. K. (1993). Gas entrainment by plunging liquid jets. *Chemical Engineering Science* 48(21), 3585-3630.
   Review of plunging-jet data and correlations (penetration depth, entrainment rate).

## Fluid properties

9. Lemmon, E. W., McLinden, M. O., Friend, D. G. Thermophysical properties of fluid systems. In *NIST Chemistry
   WebBook*, NIST Standard Reference Database 69 (P. J. Linstrom, W. G. Mallard, eds.), National Institute of
   Standards and Technology, Gaithersburg MD. <https://webbook.nist.gov/chemistry/fluid/>. Density and viscosity of
   water and air at 20 °C, 1 atm (values in `brief.md`).
10. IAPWS (2014). Revised Release on the Surface Tension of Ordinary Water Substance (IAPWS R1-76(2014)).
    International Association for the Properties of Water and Steam. <http://www.iapws.org/release.html>.
    Surface tension of water against air, 0.0728 N/m at 20 °C.

## Software in this package

- PINNeAPPle, `pinneapple_simulation.numerical_solvers.free_surface` (`pp.cfd.FreeSurfaceFlow`: the interFoam case
  writer, line sampling): this repository.
- cadquery 2.8 (STEP / STL export): <https://github.com/CadQuery/cadquery>.
