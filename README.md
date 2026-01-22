# Spacecraft Attitude Estimation (UKF vs MEKF)

This project implements and compares Unscented Kalman Filter (UKF) and Multiplicative Extended Kalman Filter (MEKF) for spacecraft attitude estimation using vector measurements (Sun, Magnetometer) and Gyroscope integration.

## Structure
- `src/`: Core libraries (dynamics, measurements, utilities) and estimators (MEKF, UKF).
- `simulations/`: Scripts to run simulations.
    - `single_run.py`: Runs a single simulation and plots attitude error, bias estimation and covariance trace.
    - `monte_carlo.py`: Runs multiple randomized simulations to compute formulation RMSE.
- `figures/`: Output plots.
- `tests/`: Unit tests.

## Dependencies
`numpy`, `scipy`, `matplotlib`
```bash
pip install -r requirements.txt
```

## Usage
### Run Single Simulation
```bash
python simulations/single_run.py
```
Outputs plots to `figures/` and performance statistics to `performance_single_run.txt`.

### Run Monte Carlo Analysis
```bash
python simulations/monte_carlo.py
```
Outputs histograms and covariance plots to `figures/`.
Outputs performance statistics to `performance_monte_carlo.txt`.

### Run Tests
```bash
python -m unittest discover tests
```

## Implementation Details

### Filters
- **MEKF**: Multiplicative Extended Kalman Filter (Error-State KF)
    - 6-state error vector: attitude error (3) + gyro bias (3)
    - Quaternion resets after each update for unit norm constraint.
- **UKF**: Unscented Quaternion Estimator (USQUE)
    - 7-state vector: quaternion (4) + gyro bias (3)
    - Sigma point propagation with MRP error parametrization
    - Unscented transform (alpha=0.001, beta=2, kappa=0)

### Dynamics and Propagation
- **Integration**: 4th-order Runge-Kutta for quaternion kinematics
- **Process Model**: Gyro integration with random walk bias.
    - Gyro noise: std_gyro = 0.001 rad/s
    - Bias random walk: std_bias = 1e-5 rad/s/sqrt(s)
    - Process noise Q: diag([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6])

### Measurements
- **Sensors**: Sun sensor + Magnetometer
- **Inertial Refs**: Sun [1,0,0], Mag field [0, 1, 0]
- **Measurement Noise**: std_meas = 0.01
- **Measurement model**: z = DCM(q) * v_inertial + noise

### Simulation Parameters
- Time step: 0.1 s
- Simulation Time: 100 s (single run), 50 s (Monte Carlo)
- Init attitude error: 10 deg (randomized in Monte Carlo)
- Init bias est: 0 (true bias randomized +- 0.02 rad/s)

## Results
Monte Carlo (50 runs) Average Execution Time (Predict + 2 Updates):
    MEKF: 0.187 ms
    UKF:  1.791 ms
    Ratio (UKF/MEKF): 9.56x

Monte Carlo Errors:
Attitude:
MEKF Mean RMS: 0.0111 rad (0.64 deg)
UKF Mean RMS: 0.0151 rad (0.87 deg)

Bias Estimation:
MEKF Mean RMS: 0.010864 rad/s
UKF Mean RMS: 0.006932 rad/s

Verdict: 
Both filters provide excellent estimation performance with < 1 deg attitude errors. 
While UKF has better bias estimation (36% better), MEKF has better attitude estimation (26% better). 
The computational cost of UKF is significantly higher than MEKF (~9.6x) due to sigma point propagation. 
This makes MEKF highly advantageous for combined computational efficiency and attitude performance.
