import unittest
import numpy as np

from adaptive_fem import solve_poisson, h1_seminorm_error
from adaptive_fem.experiments import uniform_convergence, adaptive_convergence


class TestFiniteElements(unittest.TestCase):
    def test_dirichlet_boundary_values(self):
        mesh = np.linspace(0.0, 1.0, 9)
        for degree in (1, 2):
            solution = solve_poisson(mesh, degree, 5.0 / 3.0)
            self.assertAlmostEqual(solution.values[0], 0.0)
            self.assertAlmostEqual(solution.values[mesh.size - 1], 0.0)

    def test_p1_error_decreases_under_uniform_refinement(self):
        coarse = solve_poisson(np.linspace(0.0, 1.0, 17), 1, 5.0 / 3.0)
        fine = solve_poisson(np.linspace(0.0, 1.0, 33), 1, 5.0 / 3.0)
        self.assertLess(h1_seminorm_error(fine, 5.0 / 3.0), h1_seminorm_error(coarse, 5.0 / 3.0))

    def test_uniform_rates_match_regularity_prediction(self):
        _, _, slope_p1 = uniform_convergence(5.0 / 3.0, 1, (16, 32, 64, 128))
        _, _, slope_p2 = uniform_convergence(5.0 / 3.0, 2, (16, 32, 64, 128))
        self.assertTrue(-1.1 < slope_p1 < -0.9)
        self.assertTrue(-1.3 < slope_p2 < -1.05)

    def test_adaptivity_recovers_p2_rate_and_reliability(self):
        _, _, errors, estimators, slope = adaptive_convergence(5.0 / 3.0, 2, iterations=9)
        self.assertLess(slope, -1.8)
        self.assertTrue(np.all(estimators > errors))


if __name__ == "__main__":
    unittest.main()
