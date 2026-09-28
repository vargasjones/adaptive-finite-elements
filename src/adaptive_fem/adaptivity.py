import numpy as np

from .fem import solve_poisson
from .metrics import h1_seminorm_error, residual_estimator


def mark_largest_fraction(indicators, fraction=0.25):
    indicators = np.asarray(indicators)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    nmark = max(1, int(np.ceil(fraction * indicators.size)))
    return np.argsort(indicators)[-nmark:]


def bisect_marked(mesh, marked_elements):
    mesh = np.asarray(mesh, dtype=float)
    marked = set(int(i) for i in np.asarray(marked_elements).ravel())
    new_nodes = [mesh[0]]
    for e in range(mesh.size - 1):
        if e in marked:
            new_nodes.append(0.5 * (mesh[e] + mesh[e + 1]))
        new_nodes.append(mesh[e + 1])
    return np.asarray(new_nodes)


def adaptive_history(alpha, degree, iterations=9, initial_elements=4, mark_fraction=0.25):
    mesh = np.linspace(0.0, 1.0, initial_elements + 1)
    history = []
    for k in range(iterations + 1):
        solution = solve_poisson(mesh, degree, alpha)
        error = h1_seminorm_error(solution, alpha)
        local_eta, global_eta = residual_estimator(solution, alpha)
        history.append(
            {
                "iteration": k,
                "mesh": mesh.copy(),
                "solution": solution,
                "ndof": solution.ndof,
                "error": error,
                "estimator": global_eta,
                "efficiency_index": global_eta / error,
            }
        )
        if k == iterations:
            break
        marked = mark_largest_fraction(local_eta, mark_fraction)
        mesh = bisect_marked(mesh, marked)
    return history
