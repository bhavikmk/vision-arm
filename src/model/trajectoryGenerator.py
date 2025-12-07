import numpy as np

class TrajectoryGenerator:
    """
    Joint-space trajectory generator for 6-axis robots.
    Supports:
        - Cubic polynomial (pos/vel boundary)
        - Quintic polynomial (pos/vel/acc boundary)
    """

    def __init__(self, dof=6):
        self.dof = dof

    # ----------------------------------------------------------------------
    # Polynomial helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def cubic(q0, qf, t0, tf):
        """
        Cubic polynomial trajectory (pos/vel boundary).
        vel(t0)=vel(tf)=0
        """
        T = tf - t0
        a0 = q0
        a1 = 0.0
        a2 = 3*(qf - q0) / (T**2)
        a3 = -2*(qf - q0) / (T**3)
        return a0, a1, a2, a3

    @staticmethod
    def quintic(q0, qf, t0, tf):
        """
        Quintic polynomial trajectory (pos/vel/acc = 0 at start/end)
        """
        T = tf - t0
        a0 = q0
        a1 = 0.0
        a2 = 0.0
        a3 = 10*(qf - q0)/T**3
        a4 = -15*(qf - q0)/T**4
        a5 = 6*(qf - q0)/T**5
        return a0, a1, a2, a3, a4, a5

    # ----------------------------------------------------------------------
    # Single joint evaluation
    # ----------------------------------------------------------------------
    @staticmethod
    def eval_cubic(coeff, t):
        a0, a1, a2, a3 = coeff
        q  = a0 + a1*t + a2*t**2 + a3*t**3
        dq = a1 + 2*a2*t + 3*a3*t**2
        ddq = 2*a2 + 6*a3*t
        return q, dq, ddq

    @staticmethod
    def eval_quintic(coeff, t):
        a0, a1, a2, a3, a4, a5 = coeff
        q  = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
        dq = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
        ddq = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
        return q, dq, ddq

    # ----------------------------------------------------------------------
    # Multi-joint full trajectory generator
    # ----------------------------------------------------------------------
    def plan(self, q_start, q_goal, T=2.0, steps=200, mode="quintic"):
        """
        Generate a trajectory for all joints.
        mode = 'cubic' or 'quintic'
        """
        q_start = np.array(q_start)
        q_goal = np.array(q_goal)

        coeffs = []
        for i in range(self.dof):
            if mode == "cubic":
                coeffs.append(self.cubic(q_start[i], q_goal[i], 0, T))
            elif mode == "quintic":
                coeffs.append(self.quintic(q_start[i], q_goal[i], 0, T))
            else:
                raise ValueError("Invalid mode")

        ts = np.linspace(0, T, steps)
        Q = np.zeros((steps, self.dof))
        dQ = np.zeros_like(Q)
        ddQ = np.zeros_like(Q)

        for k, t in enumerate(ts):
            for j in range(self.dof):
                if mode == "cubic":
                    Q[k, j], dQ[k, j], ddQ[k, j] = self.eval_cubic(coeffs[j], t)
                else:
                    Q[k, j], dQ[k, j], ddQ[k, j] = self.eval_quintic(coeffs[j], t)

        return ts, Q, dQ, ddQ

if __name__ == "__main__":
    traj = TrajectoryGenerator()

    q0 = np.zeros(6)
    qf = np.radians([45, -30, 20, 10, 0, 15])

    ts, Q, dQ, ddQ = traj.plan(q0, qf, T=3.0, steps=36, mode="quintic")

    print("\nTrajectory Test:")
    print("------------------------")
    print("Start q (deg):", [f"{x:.2f}" for x in np.degrees(q0)])
    print("Goal  q (deg):", [f"{x:.2f}" for x in np.degrees(qf)])
    print("\nSample points:")

    for i in range(len(ts)):
        q_deg_formatted = [f"{x:.2f}" for x in np.degrees(Q[i])]
        print(f"t={ts[i]:.2f} s  |  q_deg={q_deg_formatted}")    # Plotting example (requires matplotlib)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))

        # Plot positions
        plt.subplot(3, 1, 1)
        plt.plot(ts, np.degrees(Q))
        plt.title('Joint Positions')
        plt.ylabel('Angle (deg)')
        plt.grid(True)
        plt.legend([f'Joint {i+1}' for i in range(traj.dof)], loc='upper left')

        # Plot velocities
        plt.subplot(3, 1, 2)
        plt.plot(ts, np.degrees(dQ))
        plt.title('Joint Velocities')
        plt.ylabel('Velocity (deg/s)')
        plt.grid(True)
        plt.legend([f'Joint {i+1}' for i in range(traj.dof)], loc='upper left')

        # Plot accelerations
        plt.subplot(3, 1, 3)
        plt.plot(ts, np.degrees(ddQ))
        plt.title('Joint Accelerations')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (deg/s^2)')
        plt.grid(True)
        plt.legend([f'Joint {i+1}' for i in range(traj.dof)], loc='upper left')

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\nMatplotlib not found. Skipping plot generation.")
