"""Checks for the independently solvable node crossing and extraction controls."""
import unittest

import numpy as np

from weyl_continuum_stress import (
    E, crossing_model, fit_selected, isolated_roots, node_numerator, node_response,
)


class WeylContinuumStressTests(unittest.TestCase):
    def test_node_reduction_matches_full_matrix_even_above_cut(self):
        model, q, center = crossing_model(.4)
        for z in (center + .003j, .999 + .002j, 1.01 + .001j):
            a = model.response_block(z, .006)
            actual = E @ np.linalg.solve(np.eye(4) - a @ model.interaction_full(q), a @ E)
            self.assertLess(abs(actual - node_response(model, z, .006)) / abs(actual), 1e-10)

    def test_isolates_close_poles_and_uncancelled_zero(self):
        model, _, _ = crossing_model(.2)
        poles, zero = isolated_roots(model, .006)
        self.assertLess(poles[1] - poles[0], 1e-5)
        self.assertTrue(1 < poles[0] < zero < poles[1])
        self.assertLess(abs(node_numerator(model, zero, .006)), 1e-8)

    def test_uniform_shift_and_canonical_control(self):
        model, _, center = crossing_model(.8)
        poles, zero = isolated_roots(model, .006)
        gamma = .2 * (center - 1)
        shifted = zero - 1j * gamma
        self.assertLess(abs(node_numerator(model, shifted + 1j * gamma, .006)), 1e-10)
        x = np.linspace(center - .05, center + .05, 301)
        y = (x - shifted) / ((x - poles[0] + 1j * gamma) * (x - poles[1] + 1j * gamma))
        fit = fit_selected(x, y, center, 1)
        self.assertTrue(fit["selected"])
        self.assertLess(abs(complex(fit["selected_real"], fit["selected_imag"]) - shifted), 1e-10)

    def test_zero_field_cancellation_is_not_called_two_isolated_poles(self):
        model, _, _ = crossing_model(.4)
        with self.assertRaises(ValueError):
            isolated_roots(model, 0.)

    def test_small_residual_can_accept_wrong_continued_zero(self):
        model, _, center = crossing_model(.2)
        _, zero = isolated_roots(model, .006)
        edge = center - 1
        x = np.linspace(center - .5 * edge, center + .5 * edge, 301)
        y = np.asarray([node_response(model, s + 1j * edge, .006) for s in x])
        fit = fit_selected(x, y, center, 1)
        self.assertTrue(fit["numerical_screen_passed"])
        self.assertLess(fit["fit_rms"], 1e-4)
        estimated = complex(fit["selected_real"], fit["selected_imag"])
        self.assertGreater(abs(estimated - (zero - 1j * edge)) / edge, .1)


if __name__ == "__main__":
    unittest.main()
