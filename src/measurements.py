import numpy as np
from src.utils import q_to_dgm

def vector_measurement_model(state, inertial_vec):
    """
    Generates expected body measurement given state and inertial vector.
    z = R(q)^T * v_inertial  (Passive rotation / Coordinate transform)
    """
    q = state[:4]
    dgm = q_to_dgm(q) 
    return dgm.T @ inertial_vec

def generate_measurement(state, inertial_vec, noise_std):
    """
    Simulates a noisy measurement.
    """
    true_meas = vector_measurement_model(state, inertial_vec)
    noise = np.random.normal(0, noise_std, 3)
    return true_meas + noise
