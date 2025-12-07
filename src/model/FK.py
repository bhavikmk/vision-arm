"""
FK.py
Robust Forward Kinematics module that accepts a user-supplied DH table.

Expected DH format: a (alpha), a (link length), d (link offset), theta (either:
  - numeric offset (float) representing theta = q[i] + offset
  - a Python expression string using names theta1..theta6 and numpy (e.g. "theta2 + np.pi/2")

Example DH the user provided (this is valid for the run example):
    dh = np.array([
        [0,0,0,'theta1'],
        [np.pi/2, 0,0, 'theta2 + np.pi/2'],
        [0, 0.425,0, 'theta3'],
        [0, 0.392, 0.133, 'theta4 - np.pi/2'],
        [-np.pi/2, 0, 0.100, 'theta5'],
        [np.pi/2, 0, 0 , 'theta6']
    ], dtype=object)

This module will:
 - parse DH table rows
 - accept joint vector q (6,) in radians
 - evaluate theta expressions (if strings) using q
 - compute homogeneous transforms for each link
 - return end-effector transform and optionally all intermediate transforms
 - provide a small CLI-run test using the user's DH example
"""

import numpy as np
import math
from typing import Tuple, List, Union

# ---------------------------
# Helper: DH transform
# ---------------------------

def dh_transform(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """Return 4x4 homogeneous transform for given DH params."""
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    ct = math.cos(theta)
    st = math.sin(theta)

    T = np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,      ca,       d],
        [0.0,     0.0,     0.0,     1.0]
    ], dtype=float)
    return T


# ---------------------------
# Parser for theta entries
# ---------------------------

def eval_theta_entry(entry: Union[str, float, int], q: np.ndarray) -> float:
    """
    Evaluate the theta entry. Supported types:
      - numeric (float/int): interpreted as an absolute theta offset and result theta = q[i] + offset
      - string: Python expression using theta1..theta6 and 'np' or 'math' namespace.

    The convention used here: if entry is a string that directly equals 'thetaN',
    it will be substituted with the corresponding q value. For expressions, the
    expression is evaluated with theta1..theta6 bound to q[0]..q[5].

    Returns a numeric theta (radians).
    """
    if isinstance(entry, (float, int)):
        # Numeric offset: caller must provide which joint index to add to.
        # This function alone doesn't know the joint index; the caller will add q[i].
        return float(entry)

    if isinstance(entry, str):
        # Prepare namespace
        ns = { 'np': np, 'math': math }
        # bind theta1..theta6
        for i in range(6):
            ns[f'theta{i+1}'] = float(q[i])
        try:
            val = eval(entry, ns)
        except Exception as e:
            raise ValueError(f"Failed to eval theta expression '{entry}': {e}")
        return float(val)

    raise TypeError(f"Unsupported theta entry type: {type(entry)}")


# ---------------------------
# FK class
# ---------------------------
class FK:
    def __init__(self, dh: np.ndarray):
        """
        dh: numpy array of shape (6,4) where each row is [alpha, a, d, theta_entry]
            - alpha, a, d are numeric
            - theta_entry either numeric (offset) or string expression
        """
        dh = np.asarray(dh, dtype=object)
        if dh.shape != (6,4):
            raise ValueError(f"DH must be shape (6,4). Got {dh.shape}")
        self.dh = dh

    def forward_kinematics(self, q: Union[np.ndarray, List[float]], return_all: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Compute forward kinematics.
        q: iterable of 6 joint angles in radians.

        return_all: if True, return (T_end, [T1, T2, ..., T6])
        else return T_end
        """
        q = np.asarray(q, dtype=float).flatten()
        if q.shape != (6,):
            raise ValueError("q must be length-6 vector")

        T = np.eye(4, dtype=float)
        frames: List[np.ndarray] = []

        for i in range(6):
            alpha = float(self.dh[i,0])
            a = float(self.dh[i,1])
            d = float(self.dh[i,2])
            theta_entry = self.dh[i,3]

            # Evaluate theta entry
            theta_val = eval_theta_entry(theta_entry, q)

            # If the entry was numeric, we assume it is an offset and add q[i]
            if isinstance(theta_entry, (float, int)):
                theta = q[i] + theta_val
            else:
                # For string expressions the expression should already include the q parts
                # (e.g. 'theta2 + np.pi/2' -> evaluated to q[1] + pi/2)
                theta = theta_val

            Ti = dh_transform(alpha, a, d, theta)
            T = T @ Ti
            frames.append(T.copy())

        if return_all:
            return T, frames
        return T


# ---------------------------
# Small CLI-run test using user's DH example
# ---------------------------
if __name__ == '__main__':
    # User-provided DH (with symbolic entries as strings)
    dh = np.array([
        [0, 0, 0, 'theta1'],
        [np.pi/2, 0, 0, 'theta2 + np.pi/2'],
        [0, 0.425, 0, 'theta3'],
        [0, 0.392, 0.133, 'theta4 - np.pi/2'],
        [-np.pi/2, 0, 0.100, 'theta5'],
        [np.pi/2, 0, 0, 'theta6']
    ], dtype=object)

    fk = FK(dh)

    # Example joint vector (radians)
    q_test = np.radians([0, 0, 0, 0, 0, 0])

    T_end, frames = fk.forward_kinematics(q_test, return_all=True)

    np.set_printoptions(precision=6, suppress=True)
    print("End-effector transform (4x4):")
    print(T_end)
    print('Intermediate frames:')
    for i, f in enumerate(frames, start=1):
        print(f"T_{i}:")
        print(f)
        print()
