from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from .problem import rhs


@dataclass
class FESolution:
    mesh: np.ndarray
    degree: int
    values: np.ndarray

    @property
    def nelements(self):
        return self.mesh.size - 1

    @property
    def ndof(self):
        if self.degree == 1:
            return self.mesh.size - 2
        return 2 * self.nelements - 1

    def element_dofs(self, e):
        if self.degree == 1:
            return np.array([e, e + 1], dtype=int)
        nvert = self.mesh.size
        return np.array([e, nvert + e, e + 1], dtype=int)

    def element_values(self, e):
        return self.values[self.element_dofs(e)]

    def derivative_on_element(self, e, x):
        a, b = self.mesh[e], self.mesh[e + 1]
        h = b - a
        x = np.asarray(x)
        if self.degree == 1:
            u0, u1 = self.element_values(e)
            return np.full_like(x, (u1 - u0) / h, dtype=float)

        u0, um, u1 = self.element_values(e)
        xi = 2.0 * (x - a) / h - 1.0
        dN_dxi = np.vstack((xi - 0.5, -2.0 * xi, xi + 0.5))
        return (2.0 / h) * (np.array([u0, um, u1]) @ dN_dxi)

    def second_derivative_on_element(self, e, x):
        x = np.asarray(x)
        if self.degree == 1:
            return np.zeros_like(x, dtype=float)
        a, b = self.mesh[e], self.mesh[e + 1]
        h = b - a
        u0, um, u1 = self.element_values(e)
        value = 4.0 * (u0 - 2.0 * um + u1) / h**2
        return np.full_like(x, value, dtype=float)


def _shape_data(degree, xi):
    if degree == 1:
        N = np.vstack(((1.0 - xi) / 2.0, (1.0 + xi) / 2.0))
        dN_dxi = np.vstack((-0.5 * np.ones_like(xi), 0.5 * np.ones_like(xi)))
        return N, dN_dxi
    if degree == 2:
        N = np.vstack((0.5 * xi * (xi - 1.0), 1.0 - xi**2, 0.5 * xi * (xi + 1.0)))
        dN_dxi = np.vstack((xi - 0.5, -2.0 * xi, xi + 0.5))
        return N, dN_dxi
    raise ValueError("degree must be 1 or 2")


def solve_poisson(mesh, degree, alpha, quadrature_order=8):
    mesh = np.asarray(mesh, dtype=float)
    if mesh.ndim != 1 or mesh.size < 2 or np.any(np.diff(mesh) <= 0):
        raise ValueError("mesh must be a strictly increasing 1-D array")
    if abs(mesh[0]) > 1e-14 or abs(mesh[-1] - 1.0) > 1e-14:
        raise ValueError("this demo assumes the domain [0, 1]")

    ne = mesh.size - 1
    if degree == 1:
        ndof_total = mesh.size
    elif degree == 2:
        ndof_total = mesh.size + ne
    else:
        raise ValueError("degree must be 1 or 2")

    A = lil_matrix((ndof_total, ndof_total), dtype=float)
    bvec = np.zeros(ndof_total)
    xi_q, w_q = leggauss(quadrature_order)
    Nq, dN_dxi_q = _shape_data(degree, xi_q)

    for e in range(ne):
        a, b = mesh[e], mesh[e + 1]
        h = b - a
        xq = (a + b) / 2.0 + (h / 2.0) * xi_q
        jac = h / 2.0
        dN_dx = (2.0 / h) * dN_dxi_q

        ke = (dN_dx * (w_q * jac)) @ dN_dx.T
        fe = Nq @ (rhs(xq, alpha) * w_q * jac)

        if degree == 1:
            dofs = np.array([e, e + 1], dtype=int)
        else:
            dofs = np.array([e, mesh.size + e, e + 1], dtype=int)

        for i_local, i_global in enumerate(dofs):
            bvec[i_global] += fe[i_local]
            for j_local, j_global in enumerate(dofs):
                A[i_global, j_global] += ke[i_local, j_local]

    boundary = np.array([0, mesh.size - 1], dtype=int)
    all_dofs = np.arange(ndof_total)
    free = np.setdiff1d(all_dofs, boundary)

    u = np.zeros(ndof_total)
    u[free] = spsolve(A.tocsr()[free][:, free], bvec[free])
    return FESolution(mesh=mesh, degree=degree, values=u)
