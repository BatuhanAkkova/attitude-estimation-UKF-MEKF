import unittest
import numpy as np
import sys
import os

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import q_mult, q_norm, q_inv, q_to_dgm, mrp_to_quat, quat_to_mrp
from src.dynamics import kinematics, rk4_step

class TestUtils(unittest.TestCase):
    def test_q_norm(self):
        q = np.array([1, 2, 3, 4])
        q_n = q_norm(q)
        self.assertAlmostEqual(np.linalg.norm(q_n), 1.0)
        
    def test_q_mult_identity(self):
        q1 = np.array([0, 0, 0, 1]) # Identity
        q2 = np.array([0.5, 0.5, 0.5, 0.5])
        q_res = q_mult(q1, q2)
        np.testing.assert_array_almost_equal(q_res, q2)
        
    def test_q_inv(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        q_inv_val = q_inv(q)
        q_res = q_mult(q, q_inv_val)
        np.testing.assert_array_almost_equal(q_res, np.array([0, 0, 0, 1]))

    def test_mrp_conversion(self):
        q = np.array([0.1, 0.2, 0.3, 0.927])
        q = q_norm(q)
        mrp = quat_to_mrp(q)
        q_back = mrp_to_quat(mrp)
        # for small angles it should be close to q
        if np.dot(q, q_back) < 0:
            q_back = -q_back
        np.testing.assert_array_almost_equal(q, q_back)

class TestDynamics(unittest.TestCase):
    def test_kinematics_static(self):
        state = np.array([0, 0, 0, 1, 0, 0, 0]) # Identity q, zero bias
        omega = np.array([0, 0, 0])
        deriv = kinematics(state, omega)
        np.testing.assert_array_almost_equal(deriv, np.zeros(7))
        
    def test_rk4_integration(self):
        # Rotate 90 degrees around Z axis
        # omega = [0, 0, pi/2] for 1 second
        state = np.array([0, 0, 0, 1, 0, 0, 0])
        omega = np.array([0, 0, np.pi/2])
        dt = 0.1
        for _ in range(10):
            state = rk4_step(state, omega, dt)
            
        # Expected: q = [0, 0, sin(pi/4), cos(pi/4)] approx: [0, 0, 0.7071, 0.7071]
        expected = np.array([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
        np.testing.assert_array_almost_equal(state[:4], expected, decimal=2)

if __name__ == '__main__':
    unittest.main()
