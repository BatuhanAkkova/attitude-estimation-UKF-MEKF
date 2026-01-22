import numpy as np
from src.utils import q_mult, q_norm

def kinematics(state, omega):
    """
    Computes x_dot = f(x, omega)
    State x: [qx, qy, qz, qw, bx, by, bz] (7x1)
    Omega: Angular velocity (3x1)
    
    If used for truth: omega is true angular velocity.
        dbeta/dt = 0 (or random walk noise added separately)
    If used for filter: omega is (omega_meas - bias).
    """
    q = state[:4]
    
    # dq/dt = 0.5 * q * omega
    w_quat = np.array([omega[0], omega[1], omega[2], 0.0])
    dq_dt = 0.5 * q_mult(q, w_quat)
    
    # dbias/dt = 0 (random walk is discrete noise)
    d_bias = np.zeros(3)
    
    return np.concatenate([dq_dt, d_bias])

def rk4_step(state, omega, dt):
    """
    Runge-Kutta 4 integration step
    """
    k1 = kinematics(state, omega)
    k2 = kinematics(state + 0.5 * dt * k1, omega)
    k3 = kinematics(state + 0.5 * dt * k2, omega)
    k4 = kinematics(state + dt * k3, omega)
    
    new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Normalize quaternion
    new_state[:4] = q_norm(new_state[:4])
    
    return new_state
