import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import q_norm, q_mult, q_inv, q_to_dgm, quat_to_mrp 
from src.dynamics import rk4_step, kinematics
from src.measurements import generate_measurement
from src.mekf import MEKF
from src.ukf import UKF

def run_simulation():
    # Simulation Parameters
    dt = 0.1
    t_end = 100.0
    steps = int(t_end / dt)
    time = np.linspace(0, t_end, steps)
    
    # Initial Truth
    true_q = np.array([0., 0., 0., 1.])
    true_bias = np.array([0.01, -0.02, 0.005]) # rad/s
    true_state = np.concatenate([true_q, true_bias])
    
    # True Angular Velocity (Sinusoidal or Constant)
    def get_true_omega(t):
        # 0.1 rad/s oscillating
        return np.array([0.1 * np.sin(0.1*t), 0.05 * np.cos(0.05*t), 0.02])

    # Sensors
    sun_inertial = np.array([1., 0., 0.])
    mag_inertial = np.array([0., 1., 0.])
    
    meas_noise_std = 0.01 
    gyro_noise_std = 0.001
    gyro_bias_walk_std = 1e-5
    
    # Estimators Initialization
    # Initial guess with some error
    init_q_est = np.array([0.1, 0., 0., 0.9])
    init_q_est = q_norm(init_q_est)
    init_bias_est = np.array([0., 0., 0.])
    init_state_est = np.concatenate([init_q_est, init_bias_est])
    
    P0 = np.eye(6) * 0.1
    Q = np.eye(6) * 1e-4
    Q[3:, 3:] = np.eye(3) * 1e-6 # Bias walk Q
    R = np.eye(3) * (meas_noise_std**2)
    
    mekf = MEKF(init_state_est.copy(), P0.copy(), Q.copy(), R.copy())
    ukf = UKF(init_state_est.copy(), P0.copy(), Q.copy(), R.copy())
    
    # Storage
    history = {
        'time': time,
        'true_q': [], 'true_bias': [],
        'mekf_q': [], 'mekf_bias': [], 'mekf_err': [], 'mekf_trace': [],
        'ukf_q': [], 'ukf_bias': [], 'ukf_err': [], 'ukf_trace': []
    }
    
    import time as pytime
    mekf_times = []
    ukf_times = []

    # Loop
    print(f"Starting simulation for {steps} steps...")
    
    for k in range(steps):
        t = time[k]
        omega_true_val = get_true_omega(t)
        
        # Truth propagation
        temp_state = np.concatenate([true_state[:4], np.zeros(3)])
        temp_next = rk4_step(temp_state, omega_true_val, dt)
        true_state[:4] = temp_next[:4] 
        true_state[4:] += np.random.normal(0, gyro_bias_walk_std, 3) * np.sqrt(dt) 
        
        omega_meas = omega_true_val + true_state[4:] + np.random.normal(0, gyro_noise_std, 3)
        z_sun = generate_measurement(true_state, sun_inertial, meas_noise_std)
        z_mag = generate_measurement(true_state, mag_inertial, meas_noise_std)
        
        # MEKF Filter
        t_start = pytime.perf_counter()
        mekf.predict(omega_meas, dt)
        mekf.update(z_sun, sun_inertial)
        mekf.update(z_mag, mag_inertial)
        t_end = pytime.perf_counter()
        mekf_times.append(t_end - t_start)
        
        # UKF Filter
        t_start = pytime.perf_counter()
        ukf.predict(omega_meas, dt)
        ukf.update(z_sun, sun_inertial)
        ukf.update(z_mag, mag_inertial)
        t_end = pytime.perf_counter()
        ukf_times.append(t_end - t_start)
        
        # Store
        history['true_q'].append(true_state[:4].copy())
        history['true_bias'].append(true_state[4:].copy())
        
        history['mekf_q'].append(mekf.state[:4].copy())
        history['mekf_bias'].append(mekf.state[4:].copy())
        history['mekf_trace'].append(np.trace(mekf.P))
        
        history['ukf_q'].append(ukf.state[:4].copy())
        history['ukf_bias'].append(ukf.state[4:].copy())
        history['ukf_trace'].append(np.trace(ukf.P))
        
        # Compute Errors
        q_err_mekf = q_mult(q_inv(mekf.state[:4]), true_state[:4])
        mrp_err_mekf = quat_to_mrp(q_err_mekf)
        history['mekf_err'].append(np.linalg.norm(mrp_err_mekf) * 4) 
        
        q_err_ukf = q_mult(q_inv(ukf.state[:4]), true_state[:4])
        mrp_err_ukf = quat_to_mrp(q_err_ukf)
        history['ukf_err'].append(np.linalg.norm(mrp_err_ukf) * 4)

    # Statistics
    avg_mekf_time = np.mean(mekf_times) * 1000 # ms
    avg_ukf_time = np.mean(ukf_times) * 1000 # ms
    
    stats = f"""Average Execution Time Monte Carlo (Predict + 2 Updates):
    MEKF: {avg_mekf_time:.3f} ms
    UKF:  {avg_ukf_time:.3f} ms
    Ratio (UKF/MEKF): {avg_ukf_time/avg_mekf_time:.2f}x

    Attitude Estimation RMS Error:
    MEKF: {np.sqrt(np.mean(history['mekf_err']**2)):.4f} rad
    UKF:  {np.sqrt(np.mean(history['ukf_err']**2)):.4f} rad

    Bias Estimation RMS Error:
    MEKF: {np.sqrt(np.mean(np.linalg.norm(history['mekf_bias'] - history['true_bias'], axis=1)**2)):.6f} rad/s
    UKF:  {np.sqrt(np.mean(np.linalg.norm(history['ukf_bias'] - history['true_bias'], axis=1)**2)):.6f} rad/s
    """
    print(stats)
    with open('performance_single_run.txt', 'w') as f:
        f.write(stats)

    # Convert to arrays
    for k in history:
        history[k] = np.array(history[k])
        
    print("Simulation complete. Plotting...")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(history['time'], history['mekf_err'], label='MEKF Error (rad)')
    plt.plot(history['time'], history['ukf_err'], label='UKF Error (rad)', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Attitude Error (rad)')
    plt.yscale('log')
    plt.title('Attitude Estimation Error (MEKF vs UKF)')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig('figures/attitude_error.png')
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['time'], history['true_bias'][:, 0], 'k-', label='True Bias X')
    plt.plot(history['time'], history['mekf_bias'][:, 0], 'r--', label='MEKF Bias X')
    plt.plot(history['time'], history['ukf_bias'][:, 0], 'b:', label='UKF Bias X')
    plt.xlabel('Time (s)')
    plt.ylabel('Bias (rad/s)')
    plt.title('Gyro Bias Estimation (X-axis)')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/bias_estimation.png')

    plt.figure(figsize=(10, 6))
    plt.plot(history['time'], history['mekf_trace'], label='MEKF Trace(P)')
    plt.plot(history['time'], history['ukf_trace'], label='UKF Trace(P)', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Trace of Covariance Matrix')
    plt.yscale('log')
    plt.title('Covariance Convergence (Trace(P))')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig('figures/covariance_trace.png')
    
    print("Plots saved to figures/")

if __name__ == '__main__':
    run_simulation()
