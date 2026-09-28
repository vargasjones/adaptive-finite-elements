from .problem import exact_solution, exact_derivative, rhs
from .fem import FESolution, solve_poisson
from .metrics import h1_seminorm_error, residual_estimator
from .adaptivity import adaptive_history, mark_largest_fraction, bisect_marked
from .experiments import uniform_convergence, adaptive_convergence

__all__ = [
    "exact_solution",
    "exact_derivative",
    "rhs",
    "FESolution",
    "solve_poisson",
    "h1_seminorm_error",
    "residual_estimator",
    "adaptive_history",
    "mark_largest_fraction",
    "bisect_marked",
    "uniform_convergence",
    "adaptive_convergence",
]
