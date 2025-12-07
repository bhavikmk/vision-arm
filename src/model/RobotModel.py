import numpy as np



dh = np.array([
    [0,0,0,theta1],
    [np.pi/2, 0,0, theta2 + np.pi/2],
    [0, 0.425,0, theta3],
    [0, 0.392, 0.133, theta4 - np.pi/2],
    [-np.pi/2, 0, 0.100, theta5],
    [np.pi/2, 0, 0 , theta6]
])

class Robot:
    def __init__(self):
        # -----------------------------
        # 1. DH PARAMETERS (UR5)
        # -----------------------------
        # a, alpha, d, theta (theta is variable for revolute joints)
        # Units: meters / radians
        self.dh = np.array([
            [0.000,      np.pi/2,  0.089159, 0],   # Joint 1
            [-0.425,     0,        0,        0],   # Joint 2
            [-0.39225,   0,        0,        0],   # Joint 3
            [0.000,      np.pi/2,  0.10915,  0],   # Joint 4
            [0.000,     -np.pi/2,  0.09465,  0],   # Joint 5
            [0.000,      0,        0.0823,   0]    # Joint 6
        ])

        # -----------------------------
        # 2. JOINT LIMITS (radians)
        # -----------------------------
        self.joint_limits = {
            "min": np.radians([-360, -360, -360, -360, -360, -360]),
            "max": np.radians([360,  360,  360,  360,  360,  360])
        }

        # -----------------------------
        # 3. DYNAMICS — MASS PROPERTIES
        # -----------------------------
        # From UR5 inertia model (approx)
        self.link_masses = np.array([
            3.7,
            8.393,
            2.275,
            1.219,
            1.219,
            0.1879
        ])

        # Center of Mass (CoM) per link (meters)
        self.link_com = np.array([
            [0.0, -0.02561,  0.00193],
            [0.0,  0.21250,  0.11336],
            [0.0,  0.11993,  0.0265 ],
            [0.0, -0.0018,   0.01634],
            [0.0,  0.0018,   0.01634],
            [0.0,  0.0,      -0.001159]
        ])

        # 6 inertia matrices (3x3 each)
        self.link_inertia = [

            # Link 1 inertia (kg*m²)
            np.array([
                [0.010267,  0,         0        ],
                [0,         0.010267,  0        ],
                [0,         0,         0.00666  ]
            ]),

            # Link 2
            np.array([
                [0.22689,   0,         0        ],
                [0,         0.22689,   0        ],
                [0,         0,         0.015107 ]
            ]),

            # Link 3
            np.array([
                [0.049443,  0,         0        ],
                [0,         0.049443,  0        ],
                [0,         0,         0.004095 ]
            ]),

            # Link 4
            np.array([
                [0.111172,  0,         0        ],
                [0,         0.111172,  0        ],
                [0,         0,         0.21942  ]
            ]),

            # Link 5
            np.array([
                [0.111172,  0,         0        ],
                [0,         0.111172,  0        ],
                [0,         0,         0.21942  ]
            ]),

            # Link 6
            np.array([
                [0.017136,  0,         0        ],
                [0,         0.017136,  0        ],
                [0,         0,         0.033822 ]
            ])
        ]

        # -----------------------------
        # 4. MOTOR & TRANSMISSION PARAMETERS
        # -----------------------------
        self.motor = {
            "torque_constant": 0.12,      # Nm/A
            "gear_ratio": 100,            # assume (UR5 uses harmonic drives ~100:1)
            "max_current": 20.0,          # A
            "max_torque": 150.0           # Nm (post gearbox)
        }

        # -----------------------------
        # 5. CONTROLLER PARAMETERS
        # -----------------------------
        # You will tune these later — realistic defaults here.
        self.control = {
            "position": {
                "Kp": np.array([200, 200, 200, 80, 60, 40]),
                "Kd": np.array([20,  20,  20,  10,  8,  5])
            },
            "velocity": {
                "Kp": np.array([10, 10, 10, 6, 4, 4]),
                "Ki": np.array([0.5, 0.5, 0.5, 0.3, 0.3, 0.3])
            }
        }

        # -----------------------------
        # 6. SAFETY LIMITS
        # -----------------------------
        self.safety = {
            "max_velocity": np.radians([180, 180, 180, 360, 360, 360]),
            "max_accel":    np.radians([300, 300, 300, 400, 400, 400]),
            "max_torque":   np.array([150, 150, 150, 28, 28, 28]),  # Nm
            "max_current":  20.0
        }

    # --------------------------------
    # Optional helper accessors
    # --------------------------------
    def get_dh(self):
        return self.dh

    def get_joint_limits(self):
        return self.joint_limits

    def get_mass_properties(self):
        return self.link_masses, self.link_com, self.link_inertia

    def get_controller_gains(self):
        return self.control

    def get_motor_params(self):
        return self.motor

    def get_safety_limits(self):
        return self.safety
