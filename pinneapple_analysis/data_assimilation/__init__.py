"""Data assimilation: variational (4D-Var through autograd) on top of the ensemble methods in
``pinneapple_analysis.inverse_problems`` (EKI) and ``pinneapple_analysis.state_estimation`` (Kalman filters),
plus Gaussian-process field reconstruction from sparse sensors and greedy sensor placement (``sensor_placement``)."""
from .models import lorenz63_step, lorenz96_step, lorenz96_tendency
from .sensor_placement import (
    GaussianProcessField,
    SensorPlacement,
    greedy_sensor_placement,
    matern_kernel,
)
from .var4d import Observation, Var4D, Var4DResult, gradient_check, twin_experiment

__all__ = ["Observation", "Var4D", "Var4DResult", "gradient_check", "twin_experiment", "lorenz96_step",
           "lorenz96_tendency", "lorenz63_step", "GaussianProcessField", "SensorPlacement", "greedy_sensor_placement",
           "matern_kernel"]
