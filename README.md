# Spacecraft Attitude Estimation Framework

## 1. Problem Statement
Accurate attitude estimation is critical for CubeSats and LEO satellites. This project simulates a high-fidelity attitude determination system (ADS) using low-cost sensors (Magnetometer, Sun Sensor, Gyroscope) to estimate spacecraft attitude quaternion ($q$) and gyroscope bias ($\beta$). I compared two industry-standard Kalman Filter formulations:
- **Multiplicative Extended Kalman Filter (MEKF)**: An error-state formulation that respects the unit quaternion constraint.
- **Unscented Kalman Filter (UKF/USQUE)**: A sigma-point filter that captures higher-order non-linearities without Jacobian linearization.

## 2. System Model

### 2.1 Coordinate Frames
- **ECI (GCRF)**: Earth-Centered Inertial, used for orbit propagation and environmental reference vectors.
- **Body Frame**: Fixed to the spacecraft.
- **LVLH**: Local-Vertical Local-Horizontal (Target frame for Earth Pointing).

### 2.2 Orbit and Dynamics
- **Orbit Model**: J2 Perturbed Two-Body Propagator (LEO, 500 km, 45 deg inclination). Now includes **Atmospheric Drag** and **Solar Radiation Pressure (SRP)** perturbations.
- **Attitude Dynamics**: Rigid body dynamics (Euler's equations). The simulation now accounts for:
    - **Initial Tumbling**: Realistic initial angular rates.
    - **Gravity Gradient Torque**: Standard 1/R^3 model.
    - **Aerodynamic Torque**: Based on center-of-pressure (CoP) offset and variable atmospheric density.
    - **SRP Torque**: Based on solar flux pressure and shadow effects.

### 2.3 Sensor Models
| Sensor | Model Type | Noise ($\sigma$) | Bias Stability |
|--------|------------|------------------|----------------|
| **Gyroscope** | Rate Integrating | $10^{-4}$ rad/s | Random Walk ($10^{-6} \text{ rad/s}/\sqrt{s}$) |
| **Magnetometer** | Tilted Dipole | 100 nT | N/A |
| **Sun Sensor** | Vector | 0.005 (Unitless) | Eclipse Handling |

## 3. Estimation Methods

### 3.1 MEKF Formulation
The MEKF estimates the error quaternion $\delta q$ relative to a reference quaternion.
- **State**: $\delta x = [\delta \theta^T, \delta \beta^T]^T$ (6x1).
- **Update**: Uses Jacobian $H$ computed from predicted measurements.
- **Reset**: Reference quaternion updated by $\delta q$ after every measurement.

### 3.2 UKF Formulation (USQUE)
The UKF uses the Unscented Transform to propagate means and covariances.
- **Sigma Points**: Generated from error state covariance $P$.
- **Propagation**: Runge-Kutta 4th order integration of sigma points.
- **Update**: Measurement update performed in sigma-space, avoiding explicit Jacobians.

## 4. Simulation Scenarios
The simulation integrates:
- **RK4 Orbit Propagation**: Evolves position $r$ and velocity $v$.
- **Environment Models**: Magnetic Field ($B_{eci}(r)$) and Sun Vector ($S_{eci}(t)$).
- **Sensor Simulation**: Generates noisy measurements in Body frame.

We provide a suite of scenarios to stress-test the filters:
- **[A] Nominal**: Small initial error (10 deg), standard noise. Baseline performance.
- **[B] Large Initial Error**: 120 deg initial error. Tests convergence from "lost in space" conditions.
- **[C] High Bias**: 10x Gyro Bias (0.1 rad/s). Tests bias estimation capacity.
- **[D] Eclipse**: Sun Sensor dropout (t=50s to 150s). Tests observability with only Magnetometer.

To run these:
```bash
python simulations/run_scenarios.py
```
Dependencies:
- `numpy`
- `matplotlib`
- `scipy` (for statistical analysis)
- `tqdm` (for parallel simulation progress bars)

## 5. Results & Analysis

### Computational Performance
Across all scenarios, the computational cost comparison is striking:
| Filter | Avg Run Time (s) | Efficiency |
|--------|------------------|------------|
| **MEKF** | **~1.7s** | **1.0x (Baseline)** |
| **UKF**  | **~14.5s** | **~8.5x Slower** |

The **MEKF** is the clear winner for real-time applications on constrained hardware. The **UKF** costs significantly more because it must integrate 13 sigma points (for a 6D error state) through the RK4 dynamics at every step.

### [A] Nominal Scenario
- **Performance**: Both filters converge rapidly (< 30s).
- **Accuracy**: Steady-state errors are matched at approx 0.1 deg.
- **Consistency**: NEES/NIS match theoretical bounds perfectly.

### [B] Large Initial Error (120 deg)
- **UKF**: Shows superior robustness and faster convergence from large angles due to its ability to capture non-linearities without linearization (Jacobians).
- **MEKF**: Converges but exhibits slightly higher transient errors initially.

### [C] High Bias & Disturbance
- **Robustness**: Both filters successfully estimate biases even under 10x nominal levels.
- **Disturbances**: Gravity gradient and drag torques introduce small periodic biases that the filters correctly track via the gyro bias state.

### [D] Sensor Dropout (Eclipse)
- **Behavior**: During eclipse (t=50-150s), Sun Sensor data is lost.
- **Covariance**: Covariance correctly inflates, and accuracy degrades gracefully as the system relies solely on the Magnetometer.
- **Recovery**: Both filters re-converge instantly upon exiting eclipse.

## 6. Consistency Analysis
- **NEES/NIS**: Multi-run Monte Carlo trials (N=50) confirm both filters are well-tuned. NEES remains within the 95% confidence intervals for the 6-DOF state.
- **Computational Cost**: UKF provides a safety margin for non-linear convergence at a high CPU cost (9x). MEKF is recommended for nominal operations.

## 8. References
- **For UKF (USQUE) implementation**: Crassidis, J. L., & Markley, F. L. (2003). Unscented Filtering for Spacecraft Attitude Estimation. Journal of Guidance, Control, and Dynamics, 26(4).
- **For MEKF implementation**: Lefferts, E. J., Markley, F. L., & Shuster, M. D. (1982). Kalman Filtering for Spacecraft Attitude Estimation. Journal of Guidance, Control, and Dynamics, 5(5).
- **For disturbance modelling and frame conversion**: Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications (4th ed.).
- **For sensor models**: Markley, F. L., & Crassidis, J. L. (2014). Fundamentals of Spacecraft Attitude Determination and Control. & Wertz, J. R. (Ed.). (1978). Spacecraft Attitude Determination and Control.