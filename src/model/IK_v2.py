# IK_improved.py
import numpy as np
from numpy.linalg import norm
from FK import FK

def wrap_angle(a):
    """Normalize angle to [-pi, pi]. Works on scalars or arrays."""
    return (a + np.pi) % (2 * np.pi) - np.pi

def pose_error_vec(T_des, T_cur):
    """6D error: [dp, rotation_vector], rotation vector from R_cur->R_des."""
    p_des = T_des[:3,3]
    p_cur = T_cur[:3,3]
    R_des = T_des[:3,:3]
    R_cur = T_cur[:3,:3]
    dp = p_des - p_cur
    # rotation from current to desired: R_err = R_des * R_cur^T
    R_err = R_des @ R_cur.T
    # convert to rotation vector (axis * angle)
    angle = np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
    if abs(angle) < 1e-8:
        rvec = np.zeros(3)
    else:
        rx = (R_err[2,1] - R_err[1,2]) / (2*np.sin(angle))
        ry = (R_err[0,2] - R_err[2,0]) / (2*np.sin(angle))
        rz = (R_err[1,0] - R_err[0,1]) / (2*np.sin(angle))
        rvec = angle * np.array([rx, ry, rz])
    return np.hstack((dp, rvec))

class IKImproved:
    def __init__(self, dh, joint_limits=None, damping=0.01, max_iter=75, tol=1e-4):
        """
        joint_limits: list-of-(min,max) in radians [ (min1,max1), ..., (min6,max6) ]
                      if None, no limit checking is done.
        """
        self.fk = FK(dh)
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.joint_limits = joint_limits

    def numeric_jacobian(self, q, eps=1e-6):
        J = np.zeros((6,6))
        T0 = self.fk.forward_kinematics(q)
        p0 = T0[:3,3]
        R0 = T0[:3,:3]
        for i in range(6):
            dq = q.copy()
            dq[i] += eps
            Ti = self.fk.forward_kinematics(dq)
            dp = (Ti[:3,3] - p0) / eps
            dR = Ti[:3,:3] @ R0.T
            # small-angle approx to get dr vector
            angle = np.arccos(np.clip((np.trace(dR) - 1.0)/2.0, -1.0, 1.0))
            if abs(angle) < 1e-9:
                dr = np.zeros(3)
            else:
                rx = (dR[2,1] - dR[1,2]) / (2*np.sin(angle))
                ry = (dR[0,2] - dR[2,0]) / (2*np.sin(angle))
                rz = (dR[1,0] - dR[0,1]) / (2*np.sin(angle))
                dr = angle * np.array([rx, ry, rz]) / eps
            J[:,i] = np.hstack((dp, dr))
        return J

    def enforce_limits(self, q):
        if self.joint_limits is None:
            return q, True
        q_new = q.copy()
        for i in range(6):
            lo, hi = self.joint_limits[i]
            # wrap q to principal value then clip
            q_new[i] = wrap_angle(q_new[i])
            if q_new[i] < lo or q_new[i] > hi:
                # Hard reject: return false
                return q_new, False
        return q_new, True

    def solve_from_seed(self, q_seed, T_des):
        q = np.array(q_seed, dtype=float)
        for it in range(self.max_iter):
            T_cur = self.fk.forward_kinematics(q)
            e = pose_error_vec(T_des, T_cur)
            err_norm = norm(e)
            if err_norm < self.tol:
                q_wrapped = wrap_angle(q)
                q_wrapped, ok = self.enforce_limits(q_wrapped)
                return q_wrapped, True, err_norm
            J = self.numeric_jacobian(q)
            # Damped least squares step
            JJt = J @ J.T
            lam2 = self.damping**2
            try:
                dq = J.T @ np.linalg.solve(JJt + lam2*np.eye(6), e)
            except np.linalg.LinAlgError:
                dq = J.T @ np.linalg.pinv(JJt + lam2*np.eye(6)) @ e
            q += dq
            # keep angles reasonable during iteration
            q = wrap_angle(q)
        # final check
        T_cur = self.fk.forward_kinematics(q)
        err_norm = norm(pose_error_vec(T_des, T_cur))
        q_wrapped = wrap_angle(q)
        q_wrapped, oklim = self.enforce_limits(q_wrapped)
        return q_wrapped, False, err_norm

    def solve(self, T_des, seeds=None, try_multiple_seeds=True):
        """
        T_des: desired 4x4 pose
        seeds: list of q seeds. If None, use several default seeds.
        Returns best solution (q, success, err_norm, seed_index)
        """
        if seeds is None:
            # try a set of diverse seeds to get different branches
            seeds = [
                np.zeros(6),
                np.radians([10,-10,10,0,0,0]),
                np.radians([45,-30,20,10,0,0]),
                np.radians([-45,30,-20,-10,0,0])
            ]
        best = None
        for idx, s in enumerate(seeds):
            q_sol, ok, err = self.solve_from_seed(np.array(s,dtype=float), T_des)
            # score: lower err and smaller deviation from seed preferred
            score = err + 0.01 * norm(wrap_angle(q_sol - s))
            if best is None or score < best['score']:
                best = {'q': q_sol, 'ok': ok, 'err': err, 'score': score, 'seed_idx': idx}
            if ok and not try_multiple_seeds:
                break
        return best['q'], best['ok'], best['err'], best['seed_idx']


# -------------------------
# Quick test (run as script)
# -------------------------
if __name__ == '__main__':
    # Use same DH as your FK script
    dh = np.array([
        [0, 0, 0, 'theta1'],
        [np.pi/2, 0, 0, 'theta2 + np.pi/2'],
        [0, 0.425, 0, 'theta3'],
        [0, 0.392, 0.133, 'theta4 - np.pi/2'],
        [-np.pi/2, 0, 0.100, 'theta5'],
        [np.pi/2, 0, 0, 'theta6']
    ], dtype=object)

    # Optional joint limits (example wide limits)
    joint_limits = [(-2*np.pi, 2*np.pi) for _ in range(6)]

    # create FK & IK
    from FK import FK  # ensures consistency
    ik = IKImproved(dh, joint_limits=joint_limits, damping=0.01, max_iter=100, tol=1e-5)

    fk = FK(dh)
    q_target = np.radians([20, -25, 5, 0, 0, 0])
    T_target = fk.forward_kinematics(q_target)

    q_guess = np.radians([30, -20, 0, 0, 0, 0])
    q_sol, ok, err, seed_idx = ik.solve(T_target, seeds=[q_guess], try_multiple_seeds=False)

    print(f"Converged? {ok}")
    print(f"Err norm: {err:.8f}")   # scientific notation, 2 decimals

    # For q_sol (convert to degrees, then print each with 2 decimals)
    q_deg = np.degrees(q_sol)
    formatted_q = [f"{x:.2f}" for x in q_deg]

    print("q_sol (deg):", formatted_q)