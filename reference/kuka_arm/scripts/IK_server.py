#!/usr/bin/env python3

# Copyright (C) 2017 Udacity Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya

# import modules
import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import *
import numpy as np

def rot_x(q):
    R_x = Matrix([[ 1,              0,        0],
                  [ 0,         cos(q),  -sin(q)],
                  [ 0,         sin(q),  cos(q)]])
    return R_x

def rot_y(q):
    R_y = Matrix([[ cos(q),        0,  sin(q)],
                  [      0,        1,       0],
                  [-sin(q),        0, cos(q)]])
    return R_y

def rot_z(q):
    R_z = Matrix([[ cos(q),  -sin(q),       0],
                  [ sin(q),   cos(q),       0],
                  [      0,        0,       1]])
    return R_z

def rtd(r):
    return r*180/pi

def dtr(d):
    return d*pi/180

def TransMatrix(alpha, a, d, q):
    T = Matrix([[            cos(q),           -sin(q),           0,             a],
                [ sin(q)*cos(alpha), cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                [ sin(q)*sin(alpha), cos(q)*sin(alpha),  cos(alpha),  cos(alpha)*d],
                [                 0,                 0,           0,             1]])
    return T

def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:

        ### Your FK code here
        
        # Create symbols
        q1,q2,q3,q4,q5,q6,q7 = symbols('q1:8')
        alpha0, alpha1,alpha2,alpha3,alpha4,alpha5,alpha6 = symbols('alpha0:7')
        a0, a1,a2,a3,a4,a5,a6 = symbols('a0:7')
        d1,d2,d3,d4,d5,d6,d7 = symbols('d1:8')
        #
        #
        # Create Modified DH parameters
        DH_table = {alpha0:        0, a0:      0, d1:  0.75, q1: q1,
             alpha1: rad(-90), a1:   0.35, d2:     0, q2: q2-rad(90),
             alpha2:        0, a2:   1.25, d3:     0, q3: q3,
             alpha3: rad(-90), a3: -0.054, d4:  1.50, q4: q4,
             alpha4:  rad(90), a4:      0, d5:     0, q5: q5,
             alpha5: rad(-90), a5:      0, d6:     0, q6: q6,
             alpha6:        0, a6:      0, d7: 0.303, q7: 0
        } 
        #
        #
        # Define Modified DH Transformation matrix
        T0_1 = TransMatrix(alpha0, a0, d1, q1).subs(DH_table)
        T1_2 = TransMatrix(alpha1, a1, d2, q2).subs(DH_table)
        T2_3 = TransMatrix(alpha2, a2, d3, q3).subs(DH_table)
        T3_4 = TransMatrix(alpha3, a3, d4, q4).subs(DH_table)
        T4_5 = TransMatrix(alpha4, a4, d5, q5).subs(DH_table)
        T5_6 = TransMatrix(alpha5, a5, d6, q6).subs(DH_table)
        T6_EE = TransMatrix(alpha6, a6, d7, q7).subs(DH_table)

        T0_EE = simplify(T0_1 * T1_2 * T2_3 * T3_4 * T4_5 * T5_6 * T6_EE)
        #
        #
        # Extract rotation matrices from the transformation matrices
        #
        #
            ###

        # Initialize service response
        joint_trajectory_list = []
        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()

            # Extract end-effector position and orientation from request
            # px,py,pz = end-effector position
            # roll, pitch, yaw = end-effector orientation

            px = req.poses[x].position.x
            py = req.poses[x].position.y
            pz = req.poses[x].position.z

            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                    req.poses[x].orientation.z, req.poses[x].orientation.w])

            ### Your IK code here
            r, p, y = symbols('r p y')

            R_x = rot_x(r)
            R_y = rot_y(p)
            R_z = rot_z(y)

            R_EE = R_z * R_y * R_x
            R_EE, WC, theta1, theta2, theta3 = calculate_123(R_EE, px, py, pz, roll, pitch, yaw)

            R0_3 = T0_1[0:3,0:3] * T1_2[0:3,0:3] * T2_3[0:3,0:3]
            R0_3 = R0_3.evalf(subs={q1:theta1, q2:theta2, q3:theta3})
            
            R3_6 = R0_3.inv("LU") * R_EE

            theta6 = atan2(-R3_6[1,1], R3_6[1,0])# +0.45370228
            sq5 = -R3_6[1,1]/sin(theta6)
            cq5 = R3_6[1,2]
            theta5 = atan2(sq5, cq5)
            sq4 = R3_6[2,2]/sin(theta5)
            cq4 = -R3_6[0,2]/sin(theta5)
            theta4 = atan2(sq4, cq4)

            if x >= len(req.poses):
                theta5 = theta5_fin
                theta6 = theta6_fin

            theta1_fin = theta1
            theta2_fin = theta2
            theta3_fin = theta3
            theta4_fin = theta4
            theta5_fin = theta5
            theta6_fin = theta6
            # Compensate for rotation discrepancy between DH parameters and Gazebo
            #
            #
            # Calculate joint angles using Geometric IK method
            #
            #
            ###

            # Populate response for the IK request
            # In the next line replace theta1,theta2...,theta6 by your joint angle variables
	        joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
	        joint_trajectory_list.append(joint_trajectory_point)

        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()

if __name__ == "__main__":
    IK_server()
