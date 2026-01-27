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
    Runge-Kutta 4 integration step (Kinematics only)
    """
    k1 = kinematics(state, omega)
    k2 = kinematics(state + 0.5 * dt * k1, omega)
    k3 = kinematics(state + 0.5 * dt * k2, omega)
    k4 = kinematics(state + dt * k3, omega)
    
    new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Normalize quaternion
    new_state[:4] = q_norm(new_state[:4])
    
    return new_state

def rigid_body_derivative(state, torque, J):
    """
    Computes derivative for Rigid Body Dynamics (Attitude + Rate).
    State: [qx, qy, qz, qw, wx, wy, wz] (7x1)
    
    Args:
        state: [q(4), omega(3)]
        torque: External Torque [Nm] (Body Frame)
        J: Inertia Matrix [kg m^2]
        
    Returns:
        d/dt([q, w])
    """
    q = state[:4]
    w = state[4:]
    
    # Kinematics: dq = 0.5 * q * w
    w_quat = np.array([w[0], w[1], w[2], 0.0])
    dq_dt = 0.5 * q_mult(q, w_quat)
    
    # Dynamics: J dw = tau - w x (J w)
    # dw = J_inv * (tau - w x (J w))
    
    Jw = J @ w
    cross_term = np.cross(w, Jw)
    dw_dt = np.linalg.inv(J) @ (torque - cross_term)
    
    return np.concatenate([dq_dt, dw_dt])

def rk4_dynamics_step(state, dt, torque, J):
    """
    RK4 step for Rigid Body Dynamics.
    """
    k1 = rigid_body_derivative(state, torque, J)
    k2 = rigid_body_derivative(state + 0.5 * dt * k1, torque, J)
    k3 = rigid_body_derivative(state + 0.5 * dt * k2, torque, J)
    k4 = rigid_body_derivative(state + dt * k3, torque, J)
    
    new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Normalize q
    new_state[:4] = q_norm(new_state[:4])
    
    return new_state
