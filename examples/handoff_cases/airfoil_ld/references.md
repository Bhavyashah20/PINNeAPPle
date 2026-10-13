# References

Each entry was checked against a publisher or library record before it was listed. A DOI is given only where the
record confirmed it.

## Airfoil data and theory

1. Abbott, I. H., von Doenhoff, A. E., Stivers, L. S. Jr. (1945). *Summary of Airfoil Data.* NACA Report 824
   (also NACA ACR L5C05 / Wartime Report L-560). NASA Technical Reports Server, document 19930090976:
   <https://ntrs.nasa.gov/citations/19930090976>. Measured section data of the NACA 4-digit airfoils,
   NACA 2412 included (Reynolds numbers about 3 to 9 million, smooth and with standard roughness). This package
   does not digitise its plots; the quantitative baseline check is against XFOIL and thin-airfoil theory, below.
2. Abbott, I. H., von Doenhoff, A. E. (1959). *Theory of Wing Sections, Including a Summary of Airfoil Data.*
   Dover Publications, New York (first published by McGraw-Hill, 1949). The NACA 4-digit thickness and camber
   equations used in `geometry/make_geometry.py`; thin-airfoil theory (the zero-lift angle of the NACA 2412 mean line,
   -2.08 deg, computed in `simulation/summarise_baseline.py`).

## Reference code

3. Drela, M. (1989). XFOIL: An analysis and design system for low Reynolds number airfoils. In T. J. Mueller (ed.),
   *Low Reynolds Number Aerodynamics*, Lecture Notes in Engineering, vol. 54, Springer, pp. 1-12.
   DOI: [10.1007/978-3-642-84010-4_1](https://doi.org/10.1007/978-3-642-84010-4_1). XFOIL 6.99 is used for the reference
   polars (`results/reference/`).

## CFD

4. Weller, H. G., Tabor, G., Jasak, H., Fureby, C. (1998). A tensorial approach to computational continuum mechanics
   using object-oriented techniques. *Computers in Physics* 12(6), 620-631.
   DOI: [10.1063/1.168744](https://doi.org/10.1063/1.168744). The original OpenFOAM paper. The release used here is
   OpenFOAM v1912 (<https://www.openfoam.com>).
5. Menter, F. R. (1994). Two-equation eddy-viscosity turbulence models for engineering applications.
   *AIAA Journal* 32(8), 1598-1605. DOI: [10.2514/3.12149](https://doi.org/10.2514/3.12149). The k-omega SST model.
6. Celik, I. B., Ghia, U., Roache, P. J., Freitas, C. J. (2008). Procedure for estimation and reporting of uncertainty
   due to discretization in CFD applications. *Journal of Fluids Engineering* 130(7), 078001. The grid convergence
   index (GCI) used for the three-grid study.
7. NASA Langley Research Center, Turbulence Modeling Resource: 2D NACA 0012 airfoil validation case (Mach 0.15,
   Re 6 million). <https://turbmodels.larc.nasa.gov/naca0012_val.html>. The PINNeAPPle aero optimizer
   (`apps/aero_optimizer/README.md`) checked its CFD set-up against the SST results of this case before the campaign
   (zero-lift drag 0.00812, lift at 10 deg 1.086); those numbers are quoted from that README, not re-run here.

## Geometry parametrisation

8. Kulfan, B. M. (2008). Universal parametric geometry representation method. *Journal of Aircraft* 45(1), 142-158.
   DOI: [10.2514/1.29958](https://doi.org/10.2514/1.29958). The class-shape transformation (CST) used for every shape
   in this package (class function sqrt(x)(1-x), Bernstein shape functions of order 2).

## Software in this package

- PINNeAPPle, `pinneapple_design.aero` (geometry, O-grid, OpenFOAM case writer) and `apps/aero_optimizer` (the
  436-run campaign, the surrogates, the earlier aircraft-level search): this repository.
- cadquery 2.8 (STEP / STL export): <https://github.com/CadQuery/cadquery>.
