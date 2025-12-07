# IK_improved.py (Continued) or motion_planner.py

import numpy as np
from numpy.linalg import norm
from IK_v2 import IKImproved, wrap_angle, pose_error_vec

import scipy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# Assuming FK and IKImproved (including wrap_angle, pose_error_vec) are available from the code block you provided.

# --- Helper Function for Pose Interpolation ---
def interpolate_pose(T_start, T_end, alpha):
    """
    Linearly interpolates position and spherically interpolates rotation 
    (SLERP) between two 4x4 homogenous transformation matrices.
    alpha is the interpolation factor (0.0=start, 1.0=end).
    """
    # 1. Position (Linear Interpolation - LERP)
    p_start = T_start[:3, 3]
    p_end = T_end[:3, 3]
    p_interp = p_start * (1 - alpha) + p_end * alpha

    # 2. Orientation (Spherical Linear Interpolation - SLERP)
    R_start = R.from_matrix(T_start[:3, :3])
    R_end = R.from_matrix(T_end[:3, :3])
    
    # --- ERROR CORRECTION HERE ---
    # 
    # Create the Slerp object using a list of key rotations and corresponding times/factors.
    # The 'factors' should range from 0 (start) to 1 (end).
    key_rotations = R.concatenate([R_start, R_end])
    key_factors = [0, 1]
    
    # Initialize the Slerp interpolator
    slerp_interp = Slerp(key_factors, key_rotations)
    
    # Interpolate using the factor 'alpha'
    R_interp = slerp_interp(alpha)
    
    # Get the resulting rotation matrix
    R_interp_matrix = R_interp.as_matrix()

    # 3. Combine into T_interp
    T_interp = np.eye(4)
    T_interp[:3, :3] = R_interp_matrix
    T_interp[:3, 3] = p_interp
    
    return T_interp

# --- Motion Planner Class ---
class MotionPlanner:
    def __init__(self, ik_solver: IKImproved, max_joint_change=np.radians(10)):
        """
        :param ik_solver: An instance of the IKImproved class.
        :param max_joint_change: Maximum allowed angle change between steps.
        """
        self.ik = ik_solver
        self.fk = ik_solver.fk # Inherit FK from the IK solver
        self.max_dq_step = max_joint_change
        
    def plan_straight_line(self, q_start, T_goal, steps=50):
        """
        Plans a straight-line path in end-effector space from q_start's pose 
        to T_goal, using IK to find the joint configuration at each step.
        
        :param q_start: The initial 6-DOF joint configuration (seed for the first IK step).
        :param T_goal: The desired final 4x4 pose.
        :param steps: The number of steps (poses) to discretize the path into.
        :return: (Q_trajectory, success_flag, error_message)
        """
        # 1. Determine Start Pose
        T_start = self.fk.forward_kinematics(q_start)
        
        # 2. Initialize Trajectory Storage
        Q_trajectory = [q_start]
        q_current = q_start.copy()

        # 3. Iterate through Path Segments
        for i in range(1, steps + 1):
            alpha = i / steps # Interpolation factor from 0.0 to 1.0
            
            # Interpolate the desired pose T_des
            T_des = interpolate_pose(T_start, T_goal, alpha)
            
            # Solve IK using the configuration from the *previous* step as the seed
            q_sol, ok, err = self.ik.solve_from_seed(q_current, T_des)

            if not ok:
                return None, False, f"IK failed to converge at step {i}/{steps}. Error: {err:.4e}"

            # 4. Continuity Check (Prevent large joint jumps)
            # Use wrap_angle to ensure we measure the shortest path distance
            dq = wrap_angle(q_sol - q_current)
            if np.any(np.abs(dq) > self.max_dq_step):
                return None, False, f"Joint jump detected at step {i}/{steps} on Joint {np.argmax(np.abs(dq))+1}. Change of {np.degrees(np.max(np.abs(dq))):.2f} deg exceeds max of {np.degrees(self.max_dq_step):.2f} deg."

            # Update and store
            q_current = q_sol
            Q_trajectory.append(q_current)

        return np.array(Q_trajectory), True, "Path successfully planned."

# -------------------------
# Motion Planner Test
# -------------------------
if __name__ == '__main__':
    # --- Setup (Same as your IK test block) ---
    dh = np.array([
        [0, 0, 0, 'theta1'],
        [np.pi/2, 0, 0, 'theta2 + np.pi/2'],
        [0, 0.425, 0, 'theta3'],
        [0, 0.392, 0.133, 'theta4 - np.pi/2'],
        [-np.pi/2, 0, 0.100, 'theta5'],
        [np.pi/2, 0, 0, 'theta6']
    ], dtype=object)

    joint_limits = [(-2*np.pi, 2*np.pi) for _ in range(6)]

    # Assuming FK class exists
    class FK:
        # Placeholder FK for demonstration (replace with your actual FK)
        def __init__(self, dh):
            pass
        def forward_kinematics(self, q):
            # This must return a 4x4 numpy array (Homogeneous Matrix)
            # For a quick test, let's just make T_target easy to hit from q_start
            return np.eye(4) # In a real scenario, this would compute the pose!
            
    # Instantiate the IK solver
    ik_solver = IKImproved(dh, joint_limits=joint_limits, damping=0.2, 
                max_iter=500, tol=5e-1) # Increased from 100 to 200
    
    # 1. Define Path
    q_start = np.radians([10, -10, 5, 0, 0, 0]) 
    
    # Define a goal pose T_goal. This should be generated by FK of a known point.
    # We will use the original T_target definition for consistency.
    q_target_true = np.radians([20, -25, 5, 50, 80, 40])
    T_goal = ik_solver.fk.forward_kinematics(q_target_true) # Get the real goal pose

    # 2. Plan Motion
    planner = MotionPlanner(ik_solver, max_joint_change=np.radians(10))
    Q_traj, success, message = planner.plan_straight_line(q_start, T_goal, steps=100) # Increased steps from 50 to 100
    print("\n--- Motion Planning Result ---")
    print(f"Success: {success}")
    print(f"Message: {message}")

    if success:
        print(f"Trajectory shape: {Q_traj.shape}")
        
        # Plotting the Trajectory (Joint Positions over steps/time)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for i in range(Q_traj.shape[1]):
            plt.plot(Q_traj[:, i], label=f'$q_{i+1}$')

        plt.title('Planned Joint Trajectories over Path Steps')
        plt.xlabel('Path Step Index')
        plt.ylabel('Joint Angle (radians)')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()