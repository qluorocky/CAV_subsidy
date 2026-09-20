"""Numerical invariants, including a small-grid HJB smoke test."""

import unittest

import numpy as np

from av_subsidy.model import Parameters, bass_drift, benefit, deterministic_path, simulate_paths, solve_hjb


class ModelTests(unittest.TestCase):
    def test_benefit_matches_paper_parameters(self):
        p = Parameters()
        self.assertAlmostEqual(float(benefit(0, p)), 0)
        self.assertAlmostEqual(float(benefit(p.beta, p)), -p.benefit_scale*p.beta**2)
        self.assertAlmostEqual(float(benefit(1, p)), 2040)

    def test_bass_paths_and_seed(self):
        t, x = deterministic_path(.05, .38)
        self.assertEqual(len(t), 101)
        self.assertTrue(np.all(np.diff(x) >= 0))
        self.assertTrue(0 <= x.min() <= x.max() <= 1)
        _, paths = simulate_paths(a=.05, b=.38, sigma=.1, steps=15, trials=5)
        _, again = simulate_paths(a=.05, b=.38, sigma=.1, steps=15, trials=5)
        np.testing.assert_array_equal(paths, again)
        self.assertAlmostEqual(float(bass_drift(1, .05, .38)), 0)

    def test_hjb_boundary_and_convergence(self):
        r = solve_hjb(nx=20, nw=20)
        self.assertTrue(r.converged)
        self.assertTrue(np.all(np.isfinite(r.value)))
        np.testing.assert_allclose(r.value[0, :-1], 0)
        np.testing.assert_allclose(r.value[-1, 1:-1], 2040)
        np.testing.assert_allclose(r.value[:, 0], 0)
        np.testing.assert_allclose(r.value[:, -1], -2040)
        self.assertTrue(np.all((r.effort >= .05) & (r.effort <= .2)))
        self.assertTrue(np.all((r.subsidy >= 0) & (r.subsidy <= 1)))


if __name__ == "__main__":
    unittest.main()
