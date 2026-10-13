---
title: 'PINNeAPPle: A Python platform for physics-informed machine learning with physics-based trust checks'
tags:
  - Python
  - PyTorch
  - physics-informed neural networks
  - neural operators
  - scientific machine learning
  - partial differential equations
  - uncertainty quantification
authors:
  - name: Yan Barros
    orcid: 0000-0000-0000-0000 # TODO: replace with the real ORCID
    corresponding: true
    affiliation: 1
  - name: Enrico Schiassi
    orcid: 0000-0000-0000-0000 # TODO: replace with the real ORCID
    affiliation: 2
affiliations:
  - name: TODO, Country # TODO: affiliation of Yan Barros
    index: 1
  - name: TODO, Country # TODO: affiliation of Enrico Schiassi
    index: 2
date: 13 October 2026
bibliography: paper.bib
---

# Summary

Many questions in science and engineering come down to solving equations that
describe how a physical quantity, such as temperature, velocity or stress,
changes in space and time. Neural networks can now approximate such solutions
quickly, either by being trained against the equations themselves or by
learning from many earlier simulations. Their weak point is trust: a network
always returns an answer, even when the case in front of it lies outside what
it has learned.

PINNeAPPle is an open-source Python library, built on PyTorch
[@ansel2024pytorch2], that brings these methods together under one interface
and checks their predictions against the governing physics. A user describes a
physical problem once and solves it with an exact solution, a classical
numerical method or a neural model, then compares the results on the same
footing. The library measures how well each prediction satisfies the governing
equations, conservation laws and boundary conditions, and uses this to decide
which model to use on each case and to flag when a model should not be trusted.

# Statement of need

Physics-informed neural networks (PINNs) [@raissi2019pinn], neural operators
such as DeepONet [@lu2021deeponet] and the Fourier neural operator (FNO)
[@li2021fno], and graph networks on meshes [@pfaff2021meshgraphnets] are now
standard tools of physics-informed machine learning [@karniadakis2021piml].
Researchers who use them face two practical problems. First, each family lives
in its own library with its own problem description, so a fair comparison
between a PINN, an operator, a graph network and a classical solver on the same
problem takes substantial glue code. Second, these models are usually judged by
their error against a reference solution, which is not available when the model
is used on new cases.

PINNeAPPle is written for researchers and engineers in computational science
who train or compare physics-based surrogate models and need to know when those
models stop being valid. It addresses the two problems with (i) a single
problem description that every method consumes, together with a comparison
interface and a record of each run, and (ii) a set of physics-based checks that
need no reference solution: the residual of the governing equations, physical
bounds, conservation and the validity conditions that each model declares.

# State of the field

Several mature libraries cover parts of this space. DeepXDE [@lu2021deepxde] is
the reference implementation for PINNs and DeepONet, with a wide range of
geometries, boundary conditions and training strategies. NVIDIA PhysicsNeMo
[@physicsnemo2023] targets large-scale, GPU-accelerated training of PINNs,
operators and graph networks. NeuralPDE.jl [@zubov2021neuralpde] generates
PINNs from symbolic PDE descriptions in Julia. PINA [@coscia2023pina] offers a
PyTorch Lightning interface for PINNs and neural operators. SciANN
[@haghighat2021sciann] wraps Keras/TensorFlow for physics-informed learning,
IDRLnet [@peng2021idrlnet] provides a PINN library built on PyTorch, and
neuraloperator [@kossaifi2024neuraloperator] is the reference library for FNO
and related operators.

PINNeAPPle does not try to replace these libraries in their core tasks: for a
large PINN, DeepXDE or PhysicsNeMo will usually be the more mature choice, and
neuraloperator for operator learning at scale. Instead of rebuilding them,
PINNeAPPle provides bridges: a DeepXDE adapter that turns a PINNeAPPle problem
into a DeepXDE model and trains it with DeepXDE itself, and a PhysicsNeMo bridge
for models trained there. What the libraries above do not provide, and the
reason PINNeAPPle was built as a separate package rather than as a contribution
to one of them, is a layer that sits across model families: online selection
and combination of PINNs, operators, graph networks, solvers and closed forms
from their physics residuals, a physics audit of each model's assumptions, and a
common record of runs and checks. These functions depend on treating very
different models as interchangeable experts, which is outside the scope of any
single-family library. PINNeAPPle also implements the extreme theory of
functional connections (X-TFC) [@schiassi2021xtfc], which few general-purpose
libraries include.

# Software design

The library is organised as a set of Python packages around three shared
abstractions. A `PhysicalProblem` holds the equations, domain, parameters and
conditions of a problem. Solving methods are registered by name with
`register_method` and are called through `solve` and `compare`, so a new method
is compared against existing ones without changing them. An `Experiment`
records a run with a fingerprint of the problem, so results stay attached to the
exact problem that produced them.

Two design choices follow from the goal of judging models by their physics.
First, the ensemble and the physics audit are written in plain NumPy and see
every model only through a prediction function, so PINNs, operators, graph
networks, numerical solvers and closed-form solutions are handled the same way.
Second, model weights in the ensemble are updated only from cases already seen
(a prequential Fixed-Share and AdaHedge scheme), so the choice of model cannot
overfit the evaluation data; point-wise intervals are calibrated online by
adaptive conformal inference. The trade-off is that the ensemble does not exploit
model internals, such as gradients, that a single-family library could use.

Heavy dependencies (OpenFOAM, CalculiX, FEniCS, DeepXDE, PhysicsNeMo and 3-D
viewers) are optional extras, so the core installs with NumPy, SciPy and
PyTorch, and tests that need an optional backend are skipped when it is absent.

# Key features

- **Neural models:** PINN variants (including XPINN, VPINN and Bayesian PINNs),
  DeepONet, FNO, physics-informed neural operators, MeshGraphNet, SIREN,
  physics-informed extreme learning machines and sparse identification of
  nonlinear dynamics (SINDy) [@brunton2016sindy].
- **X-TFC solvers:** constrained expressions from the theory of functional
  connections [@leake2020mtfc] combined with extreme learning machines and
  least-squares or Gauss-Newton solves, for systems of ODE initial-value
  problems, for 1-D heat and Burgers equations and on subdomains
  [@schiassi2021xtfc].
- **Symbolic PDEs:** SymPy expressions compiled into PyTorch autograd residuals
  with hard or soft boundary conditions.
- **Physics ensembles:** `PhysicsEnsemble` selects or mixes experts online from
  reference errors, physics residuals and cost, and can detect a regime change
  from the residual before any reference is available.
- **Physics audit:** monitors residuals, conservation, bounds and extrapolation,
  checks each model's declared assumptions against dimensionless numbers, and
  attributes failures to a probable cause.
- **Classical solvers and bridges:** finite differences, finite elements, finite
  volumes, spectral and lattice Boltzmann methods, with interfaces to OpenFOAM,
  CalculiX, FEniCS, DeepXDE and PhysicsNeMo.
- **Experiment records:** the PINNeAPPle Lab stores each run with its code,
  data, checks against references and figures, and builds an HTML catalogue.

# Research impact statement

PINNeAPPle is distributed on PyPI (`pip install pinneapple`) under the
Apache-2.0 licence and has had tagged releases since version 0.5.0. Its
automated test suite runs on every pull request. The repository publishes a
validation status table, generated from the tests, which marks a method as
validated only when at least one test compares it with an independent reference
(a closed-form or manufactured solution, or a published value), and lists the
methods that are tested without such a reference or not tested at all. For
example, the X-TFC solvers are validated against exact solutions, and the README
example of a physics ensemble switches models on the first case of a new regime
of an advection-diffusion problem, using the PDE residual and no reference.

The Lab includes reproducible experiments whose data, code and checks are
stored in the repository, including recovering the pendulum law from video,
Kepler's third law from planetary data and the Lorenz parameters from noisy
data, and checking finite-element and OpenFOAM results against analytical
solutions, engineering correlations and CalculiX. The repository has received issues and pull
requests from contributors outside the author team.

<!-- TODO (authors): add publications, preprints or external research groups
that use PINNeAPPle. JOSS requires evidence of research use at submission. -->

# AI usage disclosure

<!-- TODO (authors): JOSS requires this section. State which generative AI
tools and versions were used, where (code, documentation, paper text), the
nature and scope of the assistance, and confirm that the human authors
reviewed, edited and validated all assisted output and made the core design
decisions. If no such tools were used, state this explicitly. -->

TODO: to be completed by the authors.

# Acknowledgements

We thank everyone who has contributed code, issues and reviews to PINNeAPPle.

<!-- TODO (authors): acknowledge all sources of financial support and state
whether the sponsors had any involvement in the work, or state that the work
received no specific funding. -->

# References
