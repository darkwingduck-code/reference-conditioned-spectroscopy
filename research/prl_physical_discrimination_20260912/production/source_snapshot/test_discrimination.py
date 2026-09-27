"""Independent identities and negative controls; no production fixture mirroring."""
import unittest
import numpy as np
from oscillator_pilot import stiffness,modes,response,direct,rational_fit,predict,crossing
from discrimination_core import (infer_reference_zero,spectral_decomposition,compressed_zeros,
 instrument_reference,measure,transfer,blank,standard)

class PhysicsTests(unittest.TestCase):
 def test_spectral_sum_equals_direct_inverse(self):
  for q in (.66,.75,.83):
   for B in (-.2,.025,.2):
    for z in (.7+.03j,1.03+.02j,-.8+.1j):
     self.assertLess(abs(response(z,q,B)-direct(z,q,B)),2e-11)
 def test_full_cofactor_and_passivity(self):
  q=.75;B=.15;K=stiffness(q,B)
  for z in (.7+.1j,1.1+.03j,-.8+.04j,1j):
   chi=direct(z,q,B)
   self.assertLess(abs(chi-np.linalg.det((z*z*np.eye(3)-K)[1:,1:])/np.linalg.det(z*z*np.eye(3)-K)),2e-11)
   self.assertGreaterEqual(-(z*chi).imag,-1e-12)
   if z.real>0:self.assertGreaterEqual(-chi.imag,-1e-12)
 def test_field_parity_and_hamiltonian_positive(self):
  S=np.diag([1,-1,1])
  for q in (.66,.75,.83):
   for h in (0,.8,1.2):
    np.testing.assert_allclose(stiffness(q,-.2,h),S@stiffness(q,.2,h)@S,atol=0)
    self.assertGreater(min(np.linalg.eigvalsh(stiffness(q,.2,h))),0)
 def test_pp_negative_and_remote_split(self):
  for h in (0,.8,1.2):
   d=spectral_decomposition(.75,.2,h)
   self.assertAlmostEqual(d["pp_exact"]+d["negative_frequency_correction"]+d["remote_pair_correction"],d["full_zero"],13)
   if h==0:self.assertLess(abs(d["remote_pair_correction"]),1e-13)
 def test_zero_is_not_arbitrary_bare_coordinate(self):
  bare=.75*1.3
  self.assertLess(abs(compressed_zeros(.75,.2)[0]-bare),1e-13)
  self.assertGreater(abs(compressed_zeros(.75,.2,dark_remote=.12)[0]-bare),1e-4)
  self.assertGreater(abs(compressed_zeros(.75,.2,source=[1,0,.3])[0]-bare),1e-4)
 def test_zero_field_cancellation(self):
  # At B=0 chi is finite at dark frequency: zero cofactor is cancelled.
  q=.75;d=1.3*q
  self.assertLess(abs(response(d,q,0)),100)
  K=stiffness(q,0)
  reduced=np.linalg.inv(d*d*np.eye(2)-K[np.ix_([0,2],[0,2])])[0,0]
  self.assertAlmostEqual(response(d,q,0).real,reduced,10)
 def test_independent_reference_recovers_unknown_parameters(self):
  omega=np.linspace(.79,1.15,181);eta=.003;q=.73;B=.17
  chi=response(omega+1j*eta,q,B);ref=response(omega+1j*eta,q,0)
  out=infer_reference_zero(omega,eta,chi,np.full(181,1e-10),ref,np.full(181,1e-10))
  self.assertTrue(out["certified"]);self.assertAlmostEqual(out["inferred_frequency"],1.3*q,11)
  self.assertAlmostEqual(out["inferred_stiffness_coupling"],.8*B,11)
 def test_bounded_measurement_covers_true_response(self):
  rng=np.random.default_rng(710)
  omega=np.linspace(.79,1.15,181);truth=response(omega+.003j,.75,.1)
  for c in (0,1e-4,1e-3):
   inst=instrument_reference(omega,rng,c)
   chi,eps,_=measure(omega,truth,rng,1e-3,inst)
   self.assertTrue(np.all(abs(chi-truth)<=eps))
 def test_certificate_covers_injected_bounded_errors(self):
  omega=np.linspace(.79,1.15,181);eta=.003;q=.75;B=.15
  rng=np.random.default_rng(111)
  for _ in range(15):
   inst=instrument_reference(omega,rng,1e-4)
   chi,eps,_=measure(omega,response(omega+1j*eta,q,B),rng,1e-4,inst)
   ref,eps0,_=measure(omega,response(omega+1j*eta,q,0),rng,1e-4,inst)
   out=infer_reference_zero(omega,eta,chi,eps,ref,eps0)
   self.assertTrue(out["certified"])
   self.assertLessEqual(out["frequency_lower"],1.3*q)
   self.assertGreaterEqual(out["frequency_upper"],1.3*q)
 def test_large_uncertainty_is_not_certified(self):
  omega=np.linspace(.79,1.15,181)
  chi=response(omega+.003j,.75,.025);ref=response(omega+.003j,.75,0)
  out=infer_reference_zero(omega,.003,chi,np.ones(181)*100,ref,np.ones(181)*100)
  self.assertFalse(out["certified"])
 def test_model_violation_not_a_general_certificate(self):
  omega=np.linspace(.79,1.15,181);eta=.003;q=.75;B=.2
  chi=response(omega+1j*eta,q,B,dark_remote=.15)
  ref=response(omega+1j*eta,q,0)
  out=infer_reference_zero(omega,eta,chi,np.full(181,1e-10),ref,np.full(181,1e-10))
  # A formal box is valid only when the asserted star/reference model holds.
  self.assertGreater(abs(out["inferred_frequency"]-1.3*q),1e-4)
  self.assertGreater(out["transformed_relative_rms"],1e-5)

if __name__=="__main__":unittest.main(verbosity=2)

