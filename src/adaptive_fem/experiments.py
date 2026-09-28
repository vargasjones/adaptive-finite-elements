import numpy as np

from .fem import solve_poisson
from .metrics import h1_seminorm_error
from .adaptivity import adaptive_history


def uniform_convergence(alpha, degree, element_counts=(4, 8, 16, 32, 64, 128, 256, 512)):
    rows = []
    for ne in element_counts:
        mesh = np.linspace(0.0, 1.0, ne + 1)
        solution = solve_poisson(mesh, degree, alpha)
        rows.append((solution.ndof, h1_seminorm_error(solution, alpha)))
    ndof = np.array([r[0] for r in rows], dtype=float)
    errors = np.array([r[1] for r in rows], dtype=float)
    slope = np.polyfit(np.log(ndof[-4:]), np.log(errors[-4:]), 1)[0]
    return ndof, errors, slope


def adaptive_convergence(alpha, degree, iterations=9):
    hist = adaptive_history(alpha, degree, iterations=iterations)
    ndof = np.array([h["ndof"] for h in hist], dtype=float)
    errors = np.array([h["error"] for h in hist], dtype=float)
    estimators = np.array([h["estimator"] for h in hist], dtype=float)
    slope = np.polyfit(np.log(ndof[-5:]), np.log(errors[-5:]), 1)[0]
    return hist, ndof, errors, estimators, slope
