"""
IK.py
Numerical Inverse Kinematics for UR5‑equivalent robot using SAME DH table as FK.py.

Strategy: Levenberg–Marquardt DLS IK
    q_{k+1} = q_k + (J^T(JJ^T + λ^2 I)) * (x_des - x_fk)

Features:
 - Uses the same DH table (user‑supplied)
 - Uses FK class from FK.py
 - Computes pose error using position + orientation (rotation vector)
 - Limits iterations and enforces convergence tolerance
 - Provides a test block at bottom

BEFORE USE: ensure FK.py is in same folder.
"""

import numpy as np
from numpy.linalg import norm, pinv
from FK import FK

# -----------------------------------------------------
# Utility: convert rotation matrix to axis‑angle vector
# -----------------------------------------------------
def rot_to_rvec(R):
    """Convert 3x3 rotation matrix to rotation vector (axis * angle)."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1, 1))
    if abs(angle) < 1e-9:
        return np.zeros(3)
    rx = (R[2,1] - R[1,2]) / (2*np.sin(angle))
    ry = (R[0,2] - R[2,0]) / (2*np.sin(angle))
    rz = (R[1,0] - R[0,1]) / (2*np.sin(angle))
    return angle * np.array([rx, ry, rz])

# -----------------------------------------------------
# Class: Inverse Kinematics
# -----------------------------------------------------
class IK:
    def __init__(self, dh, damping=0.01):
        self.fk = FK(dh)
        self.damping = damping

    # Jacobian (numeric differentiation)
    def jacobian(self, q, eps=1e-6):
        J = np.zeros((6,6))
        T0 = self.fk.forward_kinematics(q)
        p0 = T0[:3,3]
        R0 = T0[:3,:3]

        for i in range(6):
            dq = q.copy()
            dq[i] += eps
            Ti = self.fk.forward_kinematics(dq)
            pi = Ti[:3,3]
            Ri = Ti[:3,:3]

            dp = (pi - p0) / eps
            dR = Ri @ R0.T
            dr = rot_to_rvec(dR) / eps

            J[:,i] = np.hstack((dp, dr))
        return J

    def solve(self, q0, T_des, max_iter=75, tol=1e-4):
        q = q0.copy().astype(float)
        for _ in range(max_iter):
            T = self.fk.forward_kinematics(q)
            p = T[:3,3]
            R = T[:3,:3]
            p_des = T_des[:3,3]
            R_des = T_des[:3,:3]

            e_pos = p_des - p
            e_rot = rot_to_rvec(R_des @ R.T)
            e = np.hstack((e_pos, e_rot))

            if norm(e) < tol:
                return q, True

            J = self.jacobian(q)
            lambda2 = self.damping**2
            JJT = J @ J.T + lambda2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJT, e)
            q += dq

        return q, False


# -----------------------------------------------------
# Test code
# -----------------------------------------------------
if __name__ == "__main__":
    dh = np.array([
        [0, 0, 0, 'theta1'],
        [np.pi/2, 0, 0, 'theta2 + np.pi/2'],
        [0, 0.425, 0, 'theta3'],
        [0, 0.392, 0.133, 'theta4 - np.pi/2'],
        [-np.pi/2, 0, 0.100, 'theta5'],
        [np.pi/2, 0, 0, 'theta6']
    ], dtype=object)

    ik = IK(dh)
    fk = FK(dh)

    # Target: FK(q=0)
    T_target = fk.forward_kinematics(np.zeros(6))

    q_guess = np.radians([20, -30, 10, 50, -40, 25])
    q_sol, ok = ik.solve(q_guess, T_target)

    # Convergence
    print(f"Converged: {ok}")

    # q_sol in degrees with 2 decimals
    q_deg = np.degrees(q_sol)
    print("Solution q (deg):", [f"{x:.2f}" for x in q_deg])

    # Error norm with 2 decimals (or scientific notation if very small)
    err_norm = np.linalg.norm(fk.forward_kinematics(q_sol) - T_target)
    print(f"Error norm: {err_norm:.8f}")

