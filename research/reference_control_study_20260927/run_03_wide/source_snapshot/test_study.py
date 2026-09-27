"""Small independent checks for the raw-response fit and fixed profiles."""
import unittest
import numpy as np

from study import (TRAIN, STARTS, Q0, prediction, physical_setup, physical_response,
                   fit, residual)


class JointFitChecks(unittest.TestCase):
    def test_closed_response_matches_dense_inverse_with_both_drifts_and_extra_link(self):
        field, gamma = 0.2, 0.009
        matrix, damping, source = physical_setup(Q0, field, gamma)
        baseline = matrix[0, 0]
        matrix[0, 0] += 0.008
        damping[0, 0] += 0.002
        matrix[1, 2] = matrix[2, 1] = 0.04
        p = np.array([baseline, np.log(.003), 2.6**2, np.log(.007), 1.3*Q0,
                      np.log(gamma), np.log(.16**2), .008, np.log(.005), .04])
        fitted, reference = prediction(TRAIN, p)
        dense = physical_response(TRAIN, matrix, damping, source)
        reference_matrix, reference_damping, source = physical_setup(Q0, 0, gamma)
        np.testing.assert_allclose(fitted, dense, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(reference, physical_response(TRAIN, reference_matrix, reference_damping, source),
                                   rtol=2e-12, atol=2e-12)

    def test_joint_exact_fit_estimates_reference_and_dark_frequency(self):
        true = np.array([1+(.3*Q0)**2, np.log(.003), 2.6**2, np.log(.007), 1.3*Q0,
                         np.log(.009), np.log(.16**2), .008, np.log(.005)])
        signal, reference = prediction(TRAIN, true)
        solved = fit(STARTS[0, :9], TRAIN, signal, reference, .001)
        self.assertLess(solved["chi2"], 1e-7)
        self.assertLess(abs(solved["parameters"][4]-true[4]), 1e-6)
        self.assertLess(abs(np.exp(solved["parameters"][5])-.009), 1e-6)
        self.assertLess(abs(solved["parameters"][0]-true[0]), 1e-5)

    def test_profile_really_fixes_target_and_likelihood_uses_both_spectra(self):
        true = STARTS[0, :7].copy()
        signal, reference = prediction(TRAIN, true)
        sigma = .02
        perturbation = residual(true, TRAIN, signal+sigma, reference+2j*sigma, sigma)
        self.assertAlmostEqual(float(perturbation@perturbation), 5*len(TRAIN), places=8)
        fixed_value = true[4]+.001
        profiled = fit(true, TRAIN, signal, reference, sigma, fixed=(4, fixed_value), max_nfev=40)
        self.assertEqual(profiled["parameters"][4], fixed_value)
        self.assertGreater(profiled["chi2"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
