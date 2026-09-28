import numpy as np
from numpy.polynomial.legendre import leggauss

from .problem import exact_derivative, rhs


def h1_seminorm_error(solution, alpha, quadrature_order=12):
    xi, w = leggauss(quadrature_order)
    total = 0.0
    for e in range(solution.nelements):
        a, b = solution.mesh[e], solution.mesh[e + 1]
        h = b - a
        xq = (a + b) / 2.0 + (h / 2.0) * xi
        diff = exact_derivative(xq, alpha) - solution.derivative_on_element(e, xq)
        total += np.sum(w * diff**2) * (h / 2.0)
    return float(np.sqrt(total))


def residual_estimator(solution, alpha, quadrature_order=10):
    xi, w = leggauss(quadrature_order)
    ne = solution.nelements
    eta2 = np.zeros(ne)

    jumps = np.zeros(solution.mesh.size)
    for j in range(1, solution.mesh.size - 1):
        xj = solution.mesh[j]
        left_trace = float(solution.derivative_on_element(j - 1, np.array([xj]))[0])
        right_trace = float(solution.derivative_on_element(j, np.array([xj]))[0])
        jumps[j] = left_trace - right_trace

    for e in range(ne):
        a, b = solution.mesh[e], solution.mesh[e + 1]
        h = b - a
        xq = (a + b) / 2.0 + (h / 2.0) * xi
        residual = rhs(xq, alpha) + solution.second_derivative_on_element(e, xq)
        volume = h**2 * np.sum(w * residual**2) * (h / 2.0)
        jump_term = 0.5 * h * (jumps[e] ** 2 + jumps[e + 1] ** 2)
        eta2[e] = volume + jump_term

    return np.sqrt(eta2), float(np.sqrt(np.sum(eta2)))
