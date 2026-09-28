import numpy as np


def exact_solution(x, alpha):
    x = np.asarray(x)
    return x**alpha * (1.0 - x)


def exact_derivative(x, alpha):
    x = np.asarray(x)
    return alpha * x**(alpha - 1.0) * (1.0 - x) - x**alpha


def exact_second_derivative(x, alpha):
    x = np.asarray(x)
    return (
        alpha * (alpha - 1.0) * x**(alpha - 2.0) * (1.0 - x)
        - 2.0 * alpha * x**(alpha - 1.0)
    )


def rhs(x, alpha):
    return -exact_second_derivative(x, alpha)
