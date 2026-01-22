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

import time as pytime

def run_monte_carlo(num_runs=50):
    dt = 0.1
    t_end = 50.0
    steps = int(t_end / dt)
    time = np.linspace(0, t_end, steps)
    
    mekf_rms_errors = []
    ukf_rms_errors = []
    mekf_bias_rms_errors = []
    ukf_bias_rms_errors = []
    
    # Timing and trace storage
    all_mekf_times = []
    all_ukf_times = []
    mekf_trace_history = np.zeros((num_runs, steps))
    ukf_trace_history = np.zeros((num_runs, steps))

    # Bias estimation storage
    mekf_bias_history = np.zeros((num_runs, steps, 3))
    ukf_bias_history = np.zeros((num_runs, steps, 3))
    true_bias_history = np.zeros((num_runs, steps, 3))

    print(f"Starting Monte Carlo for {num_runs} runs...")
    
    for run in range(num_runs):
        if run % 10 == 0:
            print(f"Run {run}/{num_runs}")
            
        # Initial Truth
        true_q = q_norm(np.random.randn(4)) # Random orientation
        true_bias = np.random.uniform(-0.02, 0.02, 3) 
        true_state = np.concatenate([true_q, true_bias])
        
        # True Angular Velocity (Sinusoidal)
        phase = np.random.rand(3) * 2 * np.pi
        def get_true_omega(t):
            return np.array([
                0.1 * np.sin(0.1*t + phase[0]), 
                0.05 * np.cos(0.05*t + phase[1]), 
                0.02 * np.sin(0.02*t + phase[2])
            ])
            
        sun_inertial = np.array([1., 0., 0.])
        mag_inertial = np.array([0., 1., 0.])
        
        meas_noise_std = 0.01
        gyro_noise_std = 0.001
        gyro_bias_walk_std = 1e-5
        
        # Estimators Initialization (Randomized error)
        # Apply random small rotation to true q for initial estimate
        error_angle_axis = np.random.randn(3)
        error_angle_axis *= (10.0 * np.pi / 180.0) / np.linalg.norm(error_angle_axis) # 10 deg error
        
        th = np.linalg.norm(error_angle_axis)
        v = error_angle_axis / th
        q_err = np.array([v[0]*np.sin(th/2), v[1]*np.sin(th/2), v[2]*np.sin(th/2), np.cos(th/2)])
        
        init_q_est = q_mult(true_q, q_err)
        init_q_est = q_norm(init_q_est)
        init_bias_est = np.zeros(3) # Start with 0 bias guess
        init_state_est = np.concatenate([init_q_est, init_bias_est])
        
        P0 = np.eye(6) * 0.1
        Q = np.eye(6) * 1e-4
        Q[3:, 3:] = np.eye(3) * 1e-6
        R = np.eye(3) * (meas_noise_std**2)
        
        mekf = MEKF(init_state_est.copy(), P0.copy(), Q.copy(), R.copy())
        ukf = UKF(init_state_est.copy(), P0.copy(), Q.copy(), R.copy())
        
        run_mekf_sq_err = 0
        run_ukf_sq_err = 0
        run_mekf_bias_sq_err = 0
        run_ukf_bias_sq_err = 0

        # Timing for this run
        run_mekf_times = []
        run_ukf_times = []
        
        for k in range(steps):
            t = time[k]

            omega_true_val = get_true_omega(t)
            
            # Truth propagation
            temp_state = np.concatenate([true_state[:4], np.zeros(3)])
            temp_next = rk4_step(temp_state, omega_true_val, dt)
            true_state[:4] = temp_next[:4]
            true_state[4:] += np.random.normal(0, gyro_bias_walk_std, 3) * np.sqrt(dt)
            
            # Measurement
            omega_meas = omega_true_val + true_state[4:] + np.random.normal(0, gyro_noise_std, 3)
            z_sun = generate_measurement(true_state, sun_inertial, meas_noise_std)
            z_mag = generate_measurement(true_state, mag_inertial, meas_noise_std)
            
            # MEKF Filter
            t_start = pytime.perf_counter()
            mekf.predict(omega_meas, dt)
            mekf.update(z_sun, sun_inertial)
            mekf.update(z_mag, mag_inertial)
            t_end = pytime.perf_counter()
            run_mekf_times.append(t_end - t_start)
            
            # UKF Filter
            t_start = pytime.perf_counter()
            ukf.predict(omega_meas, dt)
            ukf.update(z_sun, sun_inertial)
            ukf.update(z_mag, mag_inertial)
            t_end = pytime.perf_counter()
            run_ukf_times.append(t_end - t_start)

            # Trace storage
            mekf_trace_history[run, k] = np.trace(mekf.P)
            ukf_trace_history[run, k] = np.trace(ukf.P)

            # Error accumulation
            # MEKF attitude error
            q_err_mekf = q_mult(q_inv(mekf.state[:4]), true_state[:4])
            mrp_err_mekf = quat_to_mrp(q_err_mekf)
            err_mekf = np.linalg.norm(mrp_err_mekf) * 4
            run_mekf_sq_err += err_mekf**2

            # MEKF bias error
            bias_err_mekf = np.linalg.norm(mekf.state[4:] - true_state[4:])
            run_mekf_bias_sq_err += bias_err_mekf**2
            
            # UKF attitude error
            q_err_ukf = q_mult(q_inv(ukf.state[:4]), true_state[:4])
            mrp_err_ukf = quat_to_mrp(q_err_ukf)
            err_ukf = np.linalg.norm(mrp_err_ukf) * 4
            run_ukf_sq_err += err_ukf**2

            # UKF bias error
            bias_err_ukf = np.linalg.norm(ukf.state[4:] - true_state[4:])
            run_ukf_bias_sq_err += bias_err_ukf**2

        # Store timing for this run
        all_mekf_times.extend(run_mekf_times)
        all_ukf_times.extend(run_ukf_times)
            
        mekf_rms = np.sqrt(run_mekf_sq_err / steps)
        ukf_rms = np.sqrt(run_ukf_sq_err / steps)
        mekf_bias_rms = np.sqrt(run_mekf_bias_sq_err / steps)
        ukf_bias_rms = np.sqrt(run_ukf_bias_sq_err / steps)
        
        mekf_rms_errors.append(mekf_rms)
        ukf_rms_errors.append(ukf_rms)
        mekf_bias_rms_errors.append(mekf_bias_rms)
        ukf_bias_rms_errors.append(ukf_bias_rms)
        
    mekf_rms_errors = np.array(mekf_rms_errors)
    ukf_rms_errors = np.array(ukf_rms_errors)
    mekf_bias_rms_errors = np.array(mekf_bias_rms_errors)
    ukf_bias_rms_errors = np.array(ukf_bias_rms_errors)
    
    print(f"\n=== Attitude Estimation ===")
    print(f"MEKF Mean RMS: {np.mean(mekf_rms_errors):.4f} rad")
    print(f"UKF Mean RMS: {np.mean(ukf_rms_errors):.4f} rad")
    
    print(f"\n=== Bias Estimation ===")
    print(f"MEKF Mean Bias RMS: {np.mean(mekf_bias_rms_errors):.6f} rad/s")
    print(f"UKF Mean Bias RMS: {np.mean(ukf_bias_rms_errors):.6f} rad/s")

    # Statistics
    avg_mekf_time = np.mean(all_mekf_times) * 1000 # ms
    avg_ukf_time = np.mean(all_ukf_times) * 1000 # ms
    
    stats = f"""Average Execution Time Monte Carlo (Predict + 2 Updates):
    MEKF: {avg_mekf_time:.3f} ms
    UKF:  {avg_ukf_time:.3f} ms
    Ratio (UKF/MEKF): {avg_ukf_time/avg_mekf_time:.2f}x

    Attitude Estimation RMS Error:
    MEKF: {np.mean(mekf_rms_errors):.4f} rad
    UKF:  {np.mean(ukf_rms_errors):.4f} rad

    Bias Estimation RMS Error:
    MEKF: {np.mean(mekf_bias_rms_errors):.6f} rad/s
    UKF:  {np.mean(ukf_bias_rms_errors):.6f} rad/s
    """
    print(stats)
    with open('performance_monte_carlo.txt', 'w') as f:
        f.write(stats)

    # Plot: RMS Attitude Error Histograms
    plt.figure(figsize=(10, 6))
    plt.hist(mekf_rms_errors, bins=20, alpha=0.5, label='MEKF RMS')
    plt.hist(ukf_rms_errors, bins=20, alpha=0.5, label='UKF RMS')
    plt.xlabel('RMS Attitude Error (rad)')
    plt.ylabel('Frequency')
    plt.title(f'Monte Carlo Results ({num_runs} runs)')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/monte_carlo_hist.png')
    print("Monte Carlo plot saved.")

    # Plot: RMS Bias Error Histograms
    plt.figure(figsize=(10, 6))
    plt.hist(mekf_bias_rms_errors, bins=20, alpha=0.5, label='MEKF Bias RMS', color='red')
    plt.hist(ukf_bias_rms_errors, bins=20, alpha=0.5, label='UKF Bias RMS', color='blue')
    plt.xlabel('RMS Bias Error (rad/s)')
    plt.ylabel('Frequency')
    plt.title(f'Monte Carlo Bias Estimation Error Results ({num_runs} runs)')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/monte_carlo_bias_hist.png')
    print("Monte Carlo bias histogram saved.")

    # Plot: Average Cov. Trace
    plt.figure(figsize=(10, 6))
    mekf_trace_mean = np.mean(mekf_trace_history, axis=0)
    ukf_trace_mean = np.mean(ukf_trace_history, axis=0)
    mekf_trace_std = np.std(mekf_trace_history, axis=0)
    ukf_trace_std = np.std(ukf_trace_history, axis=0)

    plt.plot(time, mekf_trace_mean, label='MEKF Trace(P) Mean', color='red')
    plt.fill_between(time, mekf_trace_mean - mekf_trace_std, mekf_trace_mean + mekf_trace_std, 
                     alpha=0.2, color='red')
    plt.plot(time, ukf_trace_mean, label='UKF Trace(P) Mean', linestyle='--', color='blue')
    plt.fill_between(time, ukf_trace_mean - ukf_trace_std, ukf_trace_mean + ukf_trace_std, 
                     alpha=0.2, color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Trace of Covariance Matrix')
    plt.yscale('log')
    plt.title(f'Covariance Convergence - Monte Carlo Average ({num_runs} runs)')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig('figures/monte_carlo_covariance_trace.png')
    print("Monte Carlo covariance trace plot saved.")

    # Plot: Bias Estimation Traj. (X-axis as example)
    plt.figure(figsize=(10, 6))
    mekf_bias_x_mean = np.mean(mekf_bias_history[:, :, 0], axis=0)
    ukf_bias_x_mean = np.mean(ukf_bias_history[:, :, 0], axis=0)
    true_bias_x_mean = np.mean(true_bias_history[:, :, 0], axis=0)
    
    mekf_bias_x_std = np.std(mekf_bias_history[:, :, 0], axis=0)
    ukf_bias_x_std = np.std(ukf_bias_history[:, :, 0], axis=0)
    
    plt.plot(time, true_bias_x_mean, 'k-', label='True Bias X (Mean)', linewidth=2)
    plt.plot(time, mekf_bias_x_mean, 'r--', label='MEKF Bias X (Mean)')
    plt.fill_between(time, mekf_bias_x_mean - mekf_bias_x_std, mekf_bias_x_mean + mekf_bias_x_std, 
                     alpha=0.2, color='red')
    plt.plot(time, ukf_bias_x_mean, 'b:', label='UKF Bias X (Mean)')
    plt.fill_between(time, ukf_bias_x_mean - ukf_bias_x_std, ukf_bias_x_mean + ukf_bias_x_std, 
                     alpha=0.2, color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Bias (rad/s)')
    plt.title(f'Gyro Bias Estimation (X-axis) - Monte Carlo Average ({num_runs} runs)')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/monte_carlo_bias_estimation.png')
    print("Monte Carlo bias estimation plot saved.")

if __name__ == '__main__':
    run_monte_carlo()
