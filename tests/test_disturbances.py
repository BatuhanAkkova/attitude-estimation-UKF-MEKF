import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.orbit import gravity_gradient_torque, drag_accel, srp_accel, aerodynamic_torque, srp_torque, RE, get_density
from src.utils import q_from_vectors, q_to_dgm

def test_gravity_gradient():
    print("Testing Gravity Gradient...")
    # Inertia
    I = np.diag([100, 100, 50])
    
    # Position: On X axis
    r = np.array([7000.0, 0, 0])
    
    # Attitude: Identity. Body aligned with ECI.
    q_ident = np.array([0, 0, 0, 1])
    
    tau = gravity_gradient_torque(r, I, q_ident)
    assert np.allclose(tau, 0), f"Expected 0 torque for aligned body, got {tau}"
    
    angle = np.pi/4
    q_rot = np.array([0, 0, np.sin(angle/2), np.cos(angle/2)])
    
    tau = gravity_gradient_torque(r, I, q_rot)
    assert np.allclose(tau, 0), f"Expected 0 torque for aligned body, got {tau}"
    
    tau = gravity_gradient_torque(r, I, q_rot)
    assert np.allclose(tau, 0), f"Expected 0 torque for symmetric XY inertia, got {tau}"
    
    # Change I
    I = np.diag([100, 50, 50])
    
    tau = gravity_gradient_torque(r, I, q_rot)
    assert not np.allclose(tau, 0), "Expected non-zero torque"
    assert np.abs(tau[2]) > 0, "Expected Z torque"
    print("Gravity Gradient Passed.")

def test_drag():
    print("Testing Drag...")
    r = np.array([RE + 500.0, 0, 0]) # 500 km altitude
    v = np.array([0, 7.6, 0]) # Circular velocity approx
    
    m = 100.0
    A = 1.0
    Cd = 2.2
    
    accel = drag_accel(r, v, m, A, Cd)
    
    rho = get_density(r)
    assert accel[1] < 0, "Drag should oppose velocity"
    
    # Torque
    # CoP at [1, 0, 0] body. CoM at [0, 0, 0].
    cp = np.array([1.0, 0.0, 0.0])
    cm = np.array([0.0, 0.0, 0.0])
    q = np.array([0., 0., 0., 1.])
    
    tau = aerodynamic_torque(r, v, q, A, Cd, cp, cm)
    assert tau[2] < 0, "Expected negative Z torque"
    print("Drag Passed.")

def test_srp():
    print("Testing SRP...")
    r = np.array([RE + 1000, 0, 0])
    sun = np.array([1e8, 0, 0]) # Sun at +X
    
    accel = srp_accel(r, sun, 100, 1.0, 1.8)
    
    assert accel[0] < 0, "SRP should push away from sun"
    
    # Shadow check
    r_shadow = np.array([-RE - 1000, 0, 0]) # Behind Earth
    accel_shadow = srp_accel(r_shadow, sun, 100, 1.0, 1.8)
    assert np.allclose(accel_shadow, 0), "Should be in shadow"
    
    print("SRP Passed.")

if __name__ == "__main__":
    test_gravity_gradient()
    test_drag()
    test_srp()
