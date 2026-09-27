"""Runnable checks for the new inference logic; no production data required."""
import unittest
import numpy as np

from experiment import (OMEGA, Q0, physical_setup, physical_response, quadratic_poles,
                        paired_fit, predict, transformed_data, reference_lp,
                        verify_witness, exact_separation_witness,
                        instrument_reference, measure)


class ReferenceChecks(unittest.TestCase):
    def test_real_paired_fit_keeps_negative_frequency_partners(self):
        eta = 0.006
        variable = (OMEGA+1j*eta)**2
        values = 0.65/(variable-0.87**2) + 0.35/(variable-1.07**2) + 0.02+0.01*variable
        fitted = paired_fit(OMEGA, values, eta)
        np.testing.assert_allclose(predict(fitted, variable), values, rtol=1e-10, atol=1e-10)
        expected = np.sqrt(0.65*1.07**2+0.35*0.87**2)-1j*eta
        self.assertLess(abs(fitted["ppzero"]-expected), 1e-10)

    def test_schur_identity_with_independent_dark_damping(self):
        for gamma in (0.0015, 0.006):
            matrix, damping, source = physical_setup(Q0, 0.2, gamma)
            ref_matrix, ref_damping, source = physical_setup(Q0, 0, gamma)
            chi = physical_response(OMEGA, matrix, damping, source)
            reference = physical_response(OMEGA, ref_matrix, ref_damping, source)
            transform = chi*reference/(chi-reference)
            expected = (OMEGA**2-matrix[1, 1]+2j*OMEGA*gamma)/0.16**2
            np.testing.assert_allclose(transform, expected, rtol=1e-11, atol=1e-11)
            self.assertGreater(np.linalg.eigvalsh(matrix)[0], 0)
            self.assertLess(max(quadratic_poles(matrix, damping).imag), 0)
            self.assertGreater(min(-chi.imag), 0)

    def test_direct_quotient_disk_bound(self):
        rng = np.random.default_rng(92371)
        x = rng.normal(size=300)+1j*rng.normal(size=300)
        y = rng.normal(size=300)+1j*rng.normal(size=300)
        ex, ey = np.full(300, 0.001), np.full(300, 0.002)
        dx = ex*np.sqrt(rng.uniform(size=300))*np.exp(2j*np.pi*rng.uniform(size=300))
        dy = ey*np.sqrt(rng.uniform(size=300))*np.exp(2j*np.pi*rng.uniform(size=300))
        valid, transform, bound = transformed_data(x, ex, y, ey)
        exact = (x+dx)*(y+dy)/(x+dx-y-dy)
        self.assertTrue(np.all(abs(exact[valid]-transform) <= bound))

    def test_valid_star_parameter_intervals_and_unknown_gamma(self):
        gamma = 0.006
        matrix, damping, source = physical_setup(Q0, 0.2, gamma)
        ref_matrix, ref_damping, source = physical_setup(Q0, 0, gamma)
        chi = physical_response(OMEGA, matrix, damping, source)
        reference = physical_response(OMEGA, ref_matrix, ref_damping, source)
        epsilon = np.full(len(OMEGA), 1e-8)
        result = reference_lp(OMEGA, chi, epsilon, reference, epsilon)
        self.assertEqual(result["status"], "compatible")
        self.assertLessEqual(result["omega_lower"], np.sqrt(matrix[1, 1]))
        self.assertGreaterEqual(result["omega_upper"], np.sqrt(matrix[1, 1]))
        self.assertLessEqual(result["gamma_lower"], gamma)
        self.assertGreaterEqual(result["gamma_upper"], gamma)
        self.assertAlmostEqual(result["inferred_gamma"], gamma, places=10)
        # A supplied wrong common damping is rejected; freeing it restores the model.
        wrong = reference_lp(OMEGA, chi, epsilon, reference, epsilon, fixed_gamma=0.003)
        self.assertEqual(wrong["status"], "inconsistent")
        self.assertTrue(verify_witness(wrong["witness"]))

    def test_plainly_incompatible_reference_and_large_error(self):
        matrix, damping, source = physical_setup(Q0, 0.2, 0.003, "bright_drift")
        ref_matrix, ref_damping, source = physical_setup(Q0, 0, 0.003)
        chi = physical_response(OMEGA, matrix, damping, source)
        reference = physical_response(OMEGA, ref_matrix, ref_damping, source)
        small = np.full(len(OMEGA), 1e-9)
        result = reference_lp(OMEGA, chi, small, reference, small)
        self.assertEqual(result["status"], "inconsistent")
        self.assertTrue(verify_witness(result["witness"]))
        large = np.full(len(OMEGA), 1e3)
        self.assertEqual(reference_lp(OMEGA, chi, large, reference, large)["status"], "inconclusive")

    def test_bounded_noisy_valid_models_are_never_rejected(self):
        rng = np.random.default_rng(7236)
        for field in (0.05, 0.2):
            for noise in (1e-4, 1e-3):
                matrix, damping, source = physical_setup(Q0, field, 0.006)
                ref_matrix, ref_damping, source = physical_setup(Q0, 0, 0.006)
                truth = physical_response(OMEGA, matrix, damping, source)
                ref_truth = physical_response(OMEGA, ref_matrix, ref_damping, source)
                instrument = instrument_reference(OMEGA, rng, noise)
                chi, epsilon, _ = measure(OMEGA, truth, rng, noise, instrument)
                reference, ref_epsilon, _ = measure(OMEGA, ref_truth, rng, noise, instrument)
                result = reference_lp(OMEGA, chi, epsilon, reference, ref_epsilon)
                self.assertNotEqual(result["status"], "inconsistent")
                if result["status"] == "compatible":
                    self.assertLessEqual(result["omega_lower"], np.sqrt(matrix[1, 1]))
                    self.assertGreaterEqual(result["omega_upper"], np.sqrt(matrix[1, 1]))
                    self.assertLessEqual(result["gamma_lower"], 0.006)
                    self.assertGreaterEqual(result["gamma_upper"], 0.006)

    def test_tiny_numerical_perturbation_is_not_a_structural_rejection(self):
        matrix, damping, source = physical_setup(Q0, 0.2, 0.003)
        ref_matrix, ref_damping, source = physical_setup(Q0, 0, 0.003)
        chi = physical_response(OMEGA, matrix, damping, source) * (1+1e-12)
        reference = physical_response(OMEGA, ref_matrix, ref_damping, source)
        epsilon = np.full(len(OMEGA), 1e-14)
        result = reference_lp(OMEGA, chi, epsilon, reference, epsilon)
        self.assertNotEqual(result["status"], "inconsistent")

    def test_overdamped_parameter_box_does_not_publish_underdamped_root_bounds(self):
        matrix, damping, source = physical_setup(Q0, 0.2, 2.0)
        ref_matrix, ref_damping, source = physical_setup(Q0, 0, 2.0)
        chi = physical_response(OMEGA, matrix, damping, source)
        reference = physical_response(OMEGA, ref_matrix, ref_damping, source)
        epsilon = np.full(len(OMEGA), 1e-9)
        result = reference_lp(OMEGA, chi, epsilon, reference, epsilon)
        self.assertEqual(result["status"], "compatible")
        self.assertFalse(result["uniformly_underdamped_outer_box"])
        self.assertNotIn("root_imag_lower", result)

    def test_exact_witness_never_ignores_negative_unbounded_column(self):
        # -1e-12*x <= -1 is feasible for x>=0, however tiny the coefficient.
        self.assertIsNone(exact_separation_witness(np.array([[-1e-12]]), np.array([-1.]), np.array([1.])))
        witness = exact_separation_witness(np.array([[1.], [-1.]]), np.array([0., -1.]), np.array([0.5, 0.5]))
        self.assertIsNotNone(witness)
        self.assertTrue(verify_witness(witness))

    def test_invalid_measurement_bounds(self):
        with self.assertRaises(ValueError):
            transformed_data(np.ones(12), -np.ones(12), 2*np.ones(12), np.ones(12))


if __name__ == "__main__":
    unittest.main(verbosity=2)
