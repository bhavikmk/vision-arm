import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# ==========================================
# 1. MATH UTILS
# ==========================================
def wrap_angle(a):
    """Normalize angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi

def pose_error_vec(T_des, T_cur):
    """6D error vector (3 translation + 3 rotation)."""
    p_des, p_cur = T_des[:3, 3], T_cur[:3, 3]
    R_des, R_cur = T_des[:3, :3], T_cur[:3, :3]
    
    dp = p_des - p_cur
    R_err = R_des @ R_cur.T
    
    # Trace method for angle-axis
    tr = np.trace(R_err)
    angle = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    
    if abs(angle) < 1e-6:
        rvec = np.zeros(3)
    else:
        # Standard axis extraction
        rx = (R_err[2, 1] - R_err[1, 2]) / (2 * np.sin(angle))
        ry = (R_err[0, 2] - R_err[2, 0]) / (2 * np.sin(angle))
        rz = (R_err[1, 0] - R_err[0, 1]) / (2 * np.sin(angle))
        rvec = angle * np.array([rx, ry, rz])
        
    return np.hstack((dp, rvec))

# ==========================================
# 2. FORWARD KINEMATICS (Real Implementation)
# ==========================================
class FK:
    def __init__(self, dh_params):
        self.dh = dh_params

    def get_transform(self, a, alpha, d, theta):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d],
            [0,   0,      0,     1]
        ])

    def forward_kinematics(self, q):
        T = np.eye(4)
        for i, row in enumerate(self.dh):
            # Evaluate theta expression (e.g., "theta1 + pi/2")
            # We assume the row is [alpha, a, d, offset] 
            # OR standard DH: [theta_offset, d, a, alpha]
            # Based on your previous inputs, I'll use standard numeric DH + q[i]
            
            # Unpacking your specific table structure:
            # row = [alpha, a, d, theta_string_or_offset]
            alpha, a, d = float(row[0]), float(row[1]), float(row[2])
            
            # Simple offset handling for this script
            offset = 0.0
            if "np.pi" in str(row[3]):
                if "+" in str(row[3]): offset = np.pi/2
                elif "-" in str(row[3]): offset = -np.pi/2
            
            theta = q[i] + offset
            
            Ti = self.get_transform(a, alpha, d, theta)
            T = T @ Ti
        return T

# ==========================================
# 3. INVERSE KINEMATICS (Optimized DLS)
# ==========================================
class IKImproved:
    def __init__(self, dh, joint_limits=None, damping=0.15, max_iter=200, tol=1e-4):
        self.fk = FK(dh)
        self.limits = joint_limits
        self.lam = damping
        self.max_iter = max_iter
        self.tol = tol

    def numeric_jacobian(self, q, eps=1e-6):
        J = np.zeros((6, 6))
        T0 = self.fk.forward_kinematics(q)
        for i in range(6):
            q_perturbed = q.copy()
            q_perturbed[i] += eps
            Ti = self.fk.forward_kinematics(q_perturbed)
            
            # 6D numerical derivative
            e_vec = pose_error_vec(Ti, T0) 
            J[:, i] = e_vec / eps 
        return J

    def solve_from_seed(self, q_seed, T_des):
        q = q_seed.copy()
        for _ in range(self.max_iter):
            T_curr = self.fk.forward_kinematics(q)
            err = pose_error_vec(T_des, T_curr)
            
            if norm(err) < self.tol:
                return wrap_angle(q), True, norm(err)
            
            J = self.numeric_jacobian(q)
            
            # Damped Least Squares: dq = J^T * (J*J^T + lam^2*I)^-1 * err
            # Using linalg.solve is faster/stable than inversion
            A = J @ J.T + (self.lam**2) * np.eye(6)
            dq = J.T @ np.linalg.solve(A, err)
            
            q += dq
            q = wrap_angle(q)
            
        # Final check
        T_final = self.fk.forward_kinematics(q)
        final_err = norm(pose_error_vec(T_des, T_final))
        return q, (final_err < self.tol), final_err

# ==========================================
# 4. MOTION PLANNER
# ==========================================
def interpolate_pose(T_start, T_end, alpha):
    """LERP Position + SLERP Rotation"""
    p_s, p_e = T_start[:3, 3], T_end[:3, 3]
    p_interp = p_s + (p_e - p_s) * alpha
    
    R_s = R.from_matrix(T_start[:3, :3])
    R_e = R.from_matrix(T_end[:3, :3])
    
    # Correct SLERP syntax
    key_rots = R.concatenate([R_s, R_e])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    R_interp = slerp([alpha])[0].as_matrix()
    
    T = np.eye(4)
    T[:3, :3] = R_interp
    T[:3, 3] = p_interp
    return T

class MotionPlanner:
    def __init__(self, ik_solver, max_joint_change_deg=5.0):
        self.ik = ik_solver
        self.max_dq = np.radians(max_joint_change_deg)

    def plan_straight_line(self, q_start, T_goal, step_size_m=0.01):
        """
        Plans path.
        step_size_m: Desired euclidean distance per step (in meters).
                     Used to auto-calculate number of steps.
        """
        T_start = self.ik.fk.forward_kinematics(q_start)
        
        # Auto-calculate steps based on distance
        dist = norm(T_goal[:3, 3] - T_start[:3, 3])
        # Ensure at least 50 steps, or 1 step per cm
        steps = int(max(50, dist / step_size_m))
        print(f"Planning distance: {dist:.3f}m | Steps: {steps}")

        Q_traj = [q_start]
        q_curr = q_start.copy()

        for i in range(1, steps + 1):
            alpha = i / steps
            T_des = interpolate_pose(T_start, T_goal, alpha)
            
            # Using q_curr as seed is CRITICAL for path following
            q_sol, ok, err = self.ik.solve_from_seed(q_curr, T_des)
            
            if not ok:
                return np.array(Q_traj), False, f"IK Diverged at step {i} (Err: {err:.4e})"
            
            # Check for physical jumps (singularity branch switching)
            dq = wrap_angle(q_sol - q_curr)
            max_jump = np.max(np.abs(dq))
            
            if max_jump > self.max_dq:
                return np.array(Q_traj), False, f"Jump detected: {np.degrees(max_jump):.2f} deg"
            
            Q_traj.append(q_sol)
            q_curr = q_sol
            
        return np.array(Q_traj), True, "Path Found"

# ==========================================
# 5. MAIN TEST
# ==========================================
# ==========================================
# 5. MAIN TEST (Corrected)
# ==========================================
if __name__ == '__main__':
    # 6-DOF DH Parameters
    dh = np.array([
        [0,       0,     0,     'theta1'],
        [np.pi/2, 0,     0,     'theta2 + pi/2'],
        [0,       0.425, 0,     'theta3'],
        [0,       0.392, 0.133, 'theta4 - pi/2'],
        [-np.pi/2,0,     0.100, 'theta5'],
        [np.pi/2, 0,     0,     'theta6']
    ], dtype=object)

    # 1. ADJUST SOLVER PARAMETERS
    # Lower damping (0.15 -> 0.05) helps it move faster out of difficult spots.
    ik = IKImproved(dh, damping=0.05, max_iter=500, tol=1)
    planner = MotionPlanner(ik, max_joint_change_deg=5.0)

    # 2. FIX START CONFIGURATION (Avoid Singularity)
    # Changed q5 from 0 to 10 degrees (0.17 rad) to prevent wrist lock
    q_start = np.radians([10, -10, 5, 0, 10, 0]) 
    
    # Define Goal Config
    q_goal_targ = np.radians([20, -25, 5, 50, 80, 40])
    T_goal = ik.fk.forward_kinematics(q_goal_targ)

    print("Starting Motion Plan...")
    # 3. LET PLANNER AUTO-CALCULATE STEPS
    # Pass step_size_m to control density (e.g., 1 step every 5mm)
    Q, success, msg = planner.plan_straight_line(q_start, T_goal, step_size_m=0.005)

    print(f"\nResult: {success}")
    print(f"Log: {msg}")

    if success:
        # Plot Trajectory
        plt.figure(figsize=(10, 6))
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for i in range(6):
            plt.plot(np.degrees(Q[:, i]), color=colors[i], label=f'J{i+1}')
        plt.title('Joint Trajectory')
        plt.xlabel('Step')
        plt.ylabel('Angle (Deg)')
        plt.legend()
        plt.grid(True)
        plt.show()