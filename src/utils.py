import numpy as np

def q_mult(q1, q2):
    """
    Quaternion multiplication q_out = q1 * q2
    Format: [x, y, z, w]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def q_conj(q):
    """Quaternion conjugate"""
    return np.array([-q[0], -q[1], -q[2], q[3]])

def q_inv(q):
    """Quaternion inverse"""
    return q_conj(q) / (np.linalg.norm(q)**2)

def q_norm(q):
    """Normalize quaternion"""
    return q / np.linalg.norm(q)

def q_to_dgm(q):
    """
    Quaternion to Direction Cosine Matrix (Body to Inertial)
    Using standard conversion for scalar-last quaternion [x,y,z,w]
    """
    q = q_norm(q)
    x, y, z, w = q
    
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])

def mrp_to_quat(mrp):
    """Modified Rodrigues Parameters to Quaternion [x,y,z,w]"""
    m_sq = np.dot(mrp, mrp)
    scale = 1.0 / (1.0 + m_sq)
    return np.array([
        2*mrp[0]*scale,
        2*mrp[1]*scale,
        2*mrp[2]*scale,
        (1 - m_sq)*scale
    ])

def quat_to_mrp(q):
    """Quaternion [x,y,z,w] to MRP"""
    # simple standard conversion:
    den = 1.0 + q[3]
    return np.array([q[0], q[1], q[2]]) / den

def skew(v):
    """Skew symmetric matrix from vector"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
