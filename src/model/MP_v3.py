# motion_planner_advanced.py (Integrate this with your existing FK/IK/Utils)

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import CubicSpline # New Import for advanced smoothing

# Assuming IKImproved, FK, wrap_angle, pose_error_vec, interpolate_pose are defined above

# --- Motion Planner Class ---
class MotionPlannerAdvanced:
    def __init__(self, ik_solver, max_joint_change_deg=5.0):
        """
        :param ik_solver: An instance of the IKImproved class.
        :param max_joint_change_deg: Max allowed joint jump (for continuity check).
        """
        self.ik = ik_solver
        self.max_dq = np.radians(max_joint_change_deg)
        
    def plan_spline_path(self, q_start, T_goals, time_duration=4.0, steps_per_second=50):
        """
        Plans a smooth Cartesian path through a sequence of target poses using Cubic Spline interpolation.
        
        :param q_start: The initial 6-DOF joint configuration.
        :param T_goals: A LIST of 4x4 poses (T_goal1, T_goal2, ...).
        :param time_duration: The total time (in seconds) allocated for the entire path.
        :param steps_per_second: The sampling rate (Hz) for discretizing the path.
        :return: (Q_trajectory, success_flag, error_message)
        """
        if not T_goals or len(T_goals) < 1:
            return None, False, "T_goals must be a list of at least one target pose."

        # 1. Define Waypoints and Times
        T_start = self.ik.fk.forward_kinematics(q_start)
        all_poses = [T_start] + T_goals

        # Calculate cumulative distance to assign time segments proportional to path length
        positions = np.array([T[:3, 3] for T in all_poses])
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        
        # Total distance and time
        total_dist = np.sum(distances)
        times = np.hstack(([0.0], np.cumsum(distances / total_dist * time_duration)))
        total_time = times[-1]
        
        # 2. Interpolate Position and Orientation separately
        
        # Extract Position Waypoints and create Cubic Spline
        # CubicSpline creates a smooth, C2-continuous function through the points.
        x_spline = CubicSpline(times, positions[:, 0])
        y_spline = CubicSpline(times, positions[:, 1])
        z_spline = CubicSpline(times, positions[:, 2])

        # Extract Rotation Waypoints (for SLERP interpolation)
        rotations = R.from_matrix([T[:3, :3] for T in all_poses])
        slerp_interp = Slerp(times, rotations)

        # 3. Discretize Path and Solve IK
        
        total_steps = int(total_time * steps_per_second)
        time_vector = np.linspace(0, total_time, total_steps, endpoint=True)

        Q_trajectory = [q_start]
        q_curr = q_start.copy()

        print(f"Planning {len(T_goals)} segments over {total_time:.2f}s with {total_steps} steps.")

        for i, t in enumerate(time_vector[1:], start=1):
            
            # 3a. Generate Interpolated Pose T_des
            p_des = np.array([x_spline(t), y_spline(t), z_spline(t)])
            R_des_matrix = slerp_interp(t).as_matrix()
            
            T_des = np.eye(4)
            T_des[:3, :3] = R_des_matrix
            T_des[:3, 3] = p_des
            
            # 3b. Solve IK
            q_sol, ok, err = self.ik.solve_from_seed(q_curr, T_des)

            if not ok:
                return np.array(Q_trajectory), False, f"IK Diverged at time {t:.2f}s (Step {i}/{total_steps}). Error: {err:.4e}"

            # 3c. Continuity Check
            dq = wrap_angle(q_sol - q_curr)
            max_jump = np.max(np.abs(dq))
            
            if max_jump > self.max_dq:
                return np.array(Q_trajectory), False, f"Joint jump detected: {np.degrees(max_jump):.2f} deg exceeds max of {np.degrees(self.max_dq):.2f} deg."
            
            Q_trajectory.append(q_sol)
            q_curr = q_sol
            
        return np.array(Q_trajectory), True, f"Path found with {total_steps} steps."

# -------------------------
# Advanced Motion Planner Test (Requires actual FK to run)
# -------------------------
if __name__ == '__main__':
    # --- Setup (Same DH as before) ---
    dh = np.array([
        [0, 0, 0, 'theta1'],
        [np.pi/2, 0, 0, 'theta2 + np.pi/2'],
        [0, 0.425, 0, 'theta3'],
        [0, 0.392, 0.133, 'theta4 - np.pi/2'],
        [-np.pi/2, 0, 0.100, 'theta5'],
        [np.pi/2, 0, 0, 'theta6']
    ], dtype=object)

    # Use a real FK class instance (assuming your original FK and IK are available)
    try:
        from IK_v2 import IKImproved, wrap_angle, pose_error_vec
        # Instantiate the IK solver with optimized parameters
        ik = IKImproved(dh, damping=0.5, max_iter=500, tol=1e-2)
    except ImportError:
        # Placeholder for testing without running the full codebase
        print("Note: Running with placeholder FK/IK.")
        class PlaceholderFK:
            def forward_kinematics(self, q): return np.eye(4)
        class PlaceholderIK:
            def __init__(self, dh, **kwargs): self.fk = PlaceholderFK()
            def solve_from_seed(self, q_seed, T_des): return q_seed, True, 1e-5
        
        ik = PlaceholderIK(dh)


    # Define Start Config (non-singular)
    q_start = np.radians([10, -10, 5, 0, 10, 0]) 
    
    # Define Waypoint Joint Configurations
    q_wp1 = np.radians([20, -25, 5, 50, 80, 40])
    q_wp2 = np.radians([-10, -40, 20, -30, -10, 0])

    # Calculate Goal Poses (Waypoints)
    T_wp1 = ik.fk.forward_kinematics(q_wp1)
    T_wp2 = ik.fk.forward_kinematics(q_wp2)
    
    T_goals = [T_wp1, T_wp2] # Path goes: Start -> WP1 -> WP2

    # 2. Plan Motion
    planner = MotionPlannerAdvanced(ik, max_joint_change_deg=5.0)
    Q_traj, success, message = planner.plan_spline_path(q_start, T_goals, time_duration=5.0, steps_per_second=100)

    print("\n--- Advanced Motion Planning Result ---")
    print(f"Success: {success}")
    print(f"Message: {message}")

    if success:
        # Plotting the Trajectory (Joint Positions over steps)
        # (Plotting logic omitted for brevity, but same as previous)
        print(f"Trajectory shape: {Q_traj.shape}")