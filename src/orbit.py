import numpy as np
from src.utils import q_to_dgm

# Constants
MU = 398600.4418   # Earth gravitational parameter [km^3/s^2]
J2 = 1.0826267e-3  # J2 Zonal Harmonic
RE = 6378.137      # Earth Equatorial Radius [km]

def j2_accel(r_vec):
    """
    Computes acceleration due to Earth's gravity with J2 perturbation.
    
    Args:
        r_vec (np.array): Position vector [km] in ECI.
        
    Returns:
        np.array: Acceleration vector [km/s^2].
    """
    r_norm = np.linalg.norm(r_vec)
    x, y, z = r_vec
    
    # Common terms
    r2 = r_norm**2
    r3 = r2 * r_norm
    j2_factor = 1.5 * J2 * MU * (RE**2) / (r2 * r3)
    z2_r2 = (z / r_norm)**2
    
    # Two-Body acceleration
    a_2body = -MU * r_vec / r3
    
    # J2 Perturbation
    a_j2_x = j2_factor * x * (5 * z2_r2 - 1)
    a_j2_y = j2_factor * y * (5 * z2_r2 - 1)
    a_j2_z = j2_factor * z * (5 * z2_r2 - 3)
    
    a_j2 = np.array([a_j2_x, a_j2_y, a_j2_z])
    
    return a_2body + a_j2

def rk4_orbit_step(state, dt, disturbance_accel=None):
    """
    Runge-Kutta 4 integration step for orbit dynamics.
    
    Args:
        state (np.array): [x, y, z, vx, vy, vz] (km, km/s)
        dt (float): Time step (s)
        disturbance_accel (np.array): Constant disturbance acceleration [km/s^2].
        
    Returns:
        np.array: Next state.
    """
    r = state[:3]
    v = state[3:]
    
    if disturbance_accel is None:
        disturbance_accel = np.zeros(3)
    
    # k1
    k1_v = j2_accel(r) + disturbance_accel
    k1_r = v
    
    # k2
    k2_r = v + 0.5 * dt * k1_v
    k2_v = j2_accel(r + 0.5 * dt * k1_r) + disturbance_accel
    
    # k3
    k3_r = v + 0.5 * dt * k2_v
    k3_v = j2_accel(r + 0.5 * dt * k2_r) + disturbance_accel
    
    # k4
    k4_r = v + dt * k3_v
    k4_v = j2_accel(r + dt * k3_r) + disturbance_accel
    
    # Update
    next_r = r + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    next_v = v + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
    return np.concatenate([next_r, next_v])

def gravity_gradient_torque(r_eci, I, q):
    """
    Computes Gravity Gradient Torque in Body Frame.
    
    Args:
        r_eci (np.array): Position vector [km] in ECI.
        I (np.array): Inertia tensor [kg*m^2] (3x3).
        q (np.array): Attitude quaternion [x,y,z,w] (ECI to Body).
        
    Returns:
        np.array: Torque vector [N*m] in Body Frame.
    """
    # Convert r to meters for torque calculation (result in Nm)
    r_m = r_eci * 1000.0
    r_norm = np.linalg.norm(r_m)
    
    if r_norm < 1e-3:
        return np.zeros(3)
        
    # Rotate r to Body Frame
    dgm = q_to_dgm(q) # R_b2i
    r_body = dgm.T @ r_m
    r_hat = r_body / r_norm

    mu_m = MU * 1e9
    factor = 3 * mu_m / (r_norm**3)
    # tau = (3*mu/R^3) * (r_hat x (I @ r_hat))
    torque = factor * np.cross(r_hat, I @ r_hat)
    
    return torque

def get_density(r_eci):
    """
    Simple Exponential Atmospheric Density Model.
    Args:
        r_eci (np.array): Position vector [km].
    Returns:
        float: Density [kg/m^3].
    """
    r_norm = np.linalg.norm(r_eci)
    h_km = r_norm - RE
    
    if h_km < 0:
        return 0.0
        
    # Parameters for h=500km nominal
    # Using generic scale height approximation
    h0 = 500.0
    rho0 = 6.967e-13 # kg/m^3 (approx at 500km)
    H = 63.822 # km (Scale height)
    
    if h_km > 1000:
        return 0.0
        
    rho = rho0 * np.exp(-(h_km - h0) / H)
    
    return rho

def drag_accel(r_eci, v_eci, mass, area, Cd):
    """
    Computes Atmospheric Drag Acceleration in ECI.
    
    Args:
        r_eci (np.array): Position [km].
        v_eci (np.array): Velocity [km/s].
        mass (float): S/C Mass [kg].
        area (float): Cross-sectional Area [m^2].
        Cd (float): Drag Coefficient.
        
    Returns:
        np.array: Acceleration [km/s^2].
    """
    rho = get_density(r_eci) # kg/m^3
    
    # Relative velocity considering Earth rotation
    w_earth = np.array([0, 0, 7.292115e-5]) 
    v_atm = np.cross(w_earth, r_eci) # km/s
    v_rel = v_eci - v_atm
    
    v_mag = np.linalg.norm(v_rel) # km/s
    v_mag_m = v_mag * 1000.0 # m/s
    
    if mass <= 0: return np.zeros(3)
    
    b_factor = 0.5 * rho * (Cd * area / mass)
    a_mag_m_s2 = b_factor * (v_mag_m**2)
    
    if v_mag > 1e-6:
        v_unit = v_rel / v_mag
    else:
        v_unit = np.zeros(3)
        
    a_drag_vec = -a_mag_m_s2 * v_unit / 1000.0 # Convert back to km/s^2
    
    return a_drag_vec

def srp_accel(r_eci, sun_vec, mass, area, Cr):
    """
    Computes Solar Radiation Pressure Acceleration in ECI.
    
    Args:
        r_eci (np.array): Position [km].
        sun_vec (np.array): Vector from Earth to Sun.
        mass (float): Mass [kg].
        area (float): Area [m^2].
        Cr (float): Reflectivity Coefficient.
        
    Returns:
        np.array: Acceleration [km/s^2].
    """
    if mass <= 0: return np.zeros(3)
    
    P_srp = 4.56e-6 # N/m^2 (at 1 AU)
    
    sun_norm = np.linalg.norm(sun_vec)
    if sun_norm < 1e-6: return np.zeros(3)
    
    s_hat = sun_vec / sun_norm
    
    # Cylindrical Shadow Model
    d_proj = np.dot(r_eci, s_hat)
    
    nu = 1.0
    if d_proj < 0: # Behind Earth plane
        r_perp = r_eci - d_proj * s_hat
        if np.linalg.norm(r_perp) < RE:
            # In shadow
            nu = 0.0
            
    if nu == 0.0:
        return np.zeros(3)
        
    F_mag = P_srp * Cr * area # Newtons
    a_mag = F_mag / mass # m/s^2
    a_vec = -a_mag * s_hat / 1000.0 # km/s^2
    
    return a_vec

def aerodynamic_torque(r_eci, v_eci, q, area, Cd, cp_body, com_body):
    """
    Computes Aerodynamic Torque in Body Frame.
    
    Args:
        r_eci (np.array): Position [km].
        v_eci (np.array): Velocity [km/s].
        q (np.array): Attitude Quaternion [x,y,z,w].
        area (float): Area [m^2].
        Cd (float): Drag Coefficient.
        cp_body (np.array): Center of Pressure [m].
        com_body (np.array): Center of Mass [m].
        
    Returns:
        np.array: Torque [Nm].
    """
    rho = get_density(r_eci)
    w_earth = np.array([0, 0, 7.292115e-5]) 
    v_atm = np.cross(w_earth, r_eci)
    v_rel_eci = v_eci - v_atm
    v_mag_m = np.linalg.norm(v_rel_eci) * 1000.0
    
    if v_mag_m < 1e-3:
        return np.zeros(3)
    
    F_mag = 0.5 * rho * Cd * area * (v_mag_m**2) # Newtons

    v_unit_eci = v_rel_eci / np.linalg.norm(v_rel_eci)
    F_eci = -F_mag * v_unit_eci
    
    dgm = q_to_dgm(q) # R_b2i
    F_body = dgm.T @ F_eci
    
    r_arm = cp_body - com_body
    torque = np.cross(r_arm, F_body)
    
    return torque

def srp_torque(r_eci, sun_vec, q, area, Cr, cp_body, com_body):
    """
    Computes SRP Torque in Body Frame.
    
    Args:
        sun_vec (np.array): Vector to Sun in ECI.
    """
    # Check shadow
    sun_norm = np.linalg.norm(sun_vec)
    if sun_norm < 1e-6: return np.zeros(3)
    s_hat = sun_vec / sun_norm # Points to Sun
    
    d_proj = np.dot(r_eci, s_hat)
    nu = 1.0
    if d_proj < 0:
        r_perp = r_eci - d_proj * s_hat
        if np.linalg.norm(r_perp) < RE:
            nu = 0.0
            
    if nu == 0.0:
        return np.zeros(3)
        
    P_srp = 4.56e-6
    F_mag = P_srp * Cr * area
    
    dgm = q_to_dgm(q)
    F_body = dgm.T @ (-F_mag * s_hat)
    
    r_arm = cp_body - com_body
    torque = np.cross(r_arm, F_body)
    
    return torque
