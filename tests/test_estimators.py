import unittest
import numpy as np
import sys
import os

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mekf import MEKF
from src.ukf import UKF
from src.utils import q_norm, q_mult, q_to_dgm

class TestEstimators(unittest.TestCase):
    def setUp(self):
        self.dt = 0.1
        self.initial_state = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
        self.P0 = np.eye(6) * 0.1
        self.Q = np.eye(6) * 0.01
        self.R = np.eye(3) * 0.01
        
    def test_mekf_prediction(self):
        mekf = MEKF(self.initial_state.copy(), self.P0.copy(), self.Q.copy(), self.R.copy())
        # Rotate 90 deg/s around Z
        omega_meas = np.array([0, 0, np.pi/2]) # 90 deg/s
        mekf.predict(omega_meas, self.dt)
        
        # Check if state changed
        self.assertFalse(np.allclose(mekf.state[:4], self.initial_state[:4]))
        # Check P increased (approx) due to Q
        self.assertTrue(np.all(np.diag(mekf.P) > 0))

    def test_ukf_prediction(self):
        ukf = UKF(self.initial_state.copy(), self.P0.copy(), self.Q.copy(), self.R.copy())
        omega_meas = np.array([0, 0, np.pi/2])
        ukf.predict(omega_meas, self.dt)
        
        self.assertFalse(np.allclose(ukf.state[:4], self.initial_state[:4]))
        self.assertTrue(np.all(np.diag(ukf.P) > 0))
        
    def test_mekf_update(self):
        mekf = MEKF(self.initial_state.copy(), self.P0.copy(), self.Q.copy(), self.R.copy())
        
        # Predict first
        omega_meas = np.array([0, 0, 0])
        mekf.predict(omega_meas, self.dt)
        
        # Dummy measurement: Sun vector along X
        z_ref = np.array([1, 0, 0])
        z_meas = np.array([1, 0, 0]) # Perfect match
        
        P_before = mekf.P.copy()
        mekf.update(z_meas, z_ref)
        P_after = mekf.P.copy()
        
        # Covariance should decrease
        self.assertTrue(np.trace(P_after) < np.trace(P_before))

    def test_ukf_update(self):
        ukf = UKF(self.initial_state.copy(), self.P0.copy(), self.Q.copy(), self.R.copy())
        omega_meas = np.array([0, 0, 0])
        ukf.predict(omega_meas, self.dt)
        
        z_ref = np.array([1, 0, 0])
        z_meas = np.array([1, 0, 0])
        
        P_before = ukf.P.copy()
        ukf.update(z_meas, z_ref)
        P_after = ukf.P.copy()
        
        self.assertTrue(np.trace(P_after) < np.trace(P_before))

if __name__ == '__main__':
    unittest.main()
