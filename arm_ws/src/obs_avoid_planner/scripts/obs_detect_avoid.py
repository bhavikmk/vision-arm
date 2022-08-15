from collections import deque
import imp
from re import sub
import numpy as np
from math import sqrt, atan2, pi, cos, sin
from sympy import symbols, Matrix
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class DetectAvoid:
    
    def __init__(self):

        depth_sub = rospy.Subscriber("rgbd_camera/depth/image_raw", self.depth_cb)
        rgb_sub = rospy.Subscriber('/rgbd_camera/rgb/image_raw/compressed', Image, self.rgb_cb)
        joints_sub = rospy.Subscriber('/joint_states', JointTrajectory, self.joint_cb)
    
        joint_pub1 = rospy.Publisher("seven_dof_arm/joint1_position_controller/command", Float64, queue_size=1)
        joint_pub2 = rospy.Publisher("seven_dof_arm/joint2_position_controller/command", Float64, queue_size=1)
        joint_pub3 = rospy.Publisher("seven_dof_arm/joint3_position_controller/command", Float64, queue_size=1)
        joint_pub4 = rospy.Publisher("seven_dof_arm/joint4_position_controller/command", Float64, queue_size=1)
        joint_pub5 = rospy.Publisher("seven_dof_arm/joint5_position_controller/command", Float64, queue_size=1)
        joint_pub6 = rospy.Publisher("seven_dof_arm/joint6_position_controller/command", Float64, queue_size=1)
        joint_pub7 = rospy.Publisher("seven_dof_arm/joint7_position_controller/command", Float64, queue_size=1)
    
        self.joint_state = JointState()
        self.depth_img = Image()
        self.rgb_img = Image()
        self.end_effector_pose = Pose()

    def joint_cb(self, msg):
        self.joint_state = msg

    def depth_cb(self, msg):
        self.depth_img = msg

    def rgb_cb(self, msg):
        self.rgb_img = msg
    
    def detect_obstacles(self):

        """ function to detect obstacles from RGB and depth image """

        config_space = self.depth_img.data
        discrete_graph = self.voronoi_diagram(config_space)
        path = self.A_star(discrete_graph)

        return path

    def avoid_obstacles(self):
        """ function to plan trajectory for avoiding obstacles """
        path = self.detect_obstacles()
        traj = self.path_to_traj(path)
        traj.header.stamp = rospy.Time.now()
        self.traj_to_joints(traj)

    def A_star(self, graph):
        """ function to find shortest path from start to goal in graph using A* algorithm """
        q = deque()
        q.append(self.start)
        visited = set()
        visited.add(self.start)
        parent = {}
        parent[self.start] = None
        while len(q) > 0:
            curr = q.popleft()
            if curr == self.goal:
                break
            for next in graph[curr]:
                if next not in visited:
                    visited.add(next)
                    q.append(next)
                    parent[next] = curr
        path = []
        while curr != self.start:
            path.append(curr)
            curr = parent[curr]
        path.append(self.start)
        path.reverse()
        return path

    def voronoi_diagram(self, config_space):
        """ function to create graph from configuration space """

        graph = []
        for i in range(len(config_space)):
            pass            
        
        return graph

    def T_matrix(self, alpha, a, d, theta):
        """ function to calculate transformation matrix from base to end effector """

        T = Matrix([[cos(theta), -sin(theta), 0, a],
                    [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                    [sin(theta)*sin(alpha), cos(theta)*sin(alpha), cos(alpha), cos(alpha)*d],
                    [0, 0, 0, 1]])
        return T

    def ik(self, pose):
        """ function to calculate joint angles from cartesian position using inverse kinematics """

        joint_traj_point = JointTrajectoryPoint()

        # Define DH parameter symbols for joint 1, 2, 3, 4, 5, 6, 7
        d1, d2, d3, d4, d5, d6, d7, d8 = symbols('d1:9')
        a0, a1, a2, a3, a4, a5, a6, a7 = symbols('a0:8')
        alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7 = symbols('alpha0:8')
        q1, q2, q3, q4, q5, q6, q7, q8 = symbols('q1:9')

        # Create Modified DH parameters
        s = {alpha0: 0, a0: 0, d1: 0.75, q1: q1,
            alpha1: -pi/2, a1: 0.35, d2: 0, q2: q2-pi/2,
            alpha2: 0, a2: 1.25, d3: 0, q3: q3,
            alpha3: -pi/2, a3: -0.054, d4: 1.5, q4: q4,
            alpha4: pi/2, a4: 0, d5: 0, q5: q5,
            alpha5: -pi/2, a5: 0, d6: 0, q6: q6,
            alpha6: 0, a6: 0, d7: 0.303, q7: q7,
            alpha7: 0, a7: 0, d8: 0.303, q8: 0}
    
        # Create individual transformation matrices
        T0_1 = self.T_matrix(alpha0, a0, d1, q1)
        T1_2 = self.T_matrix(alpha1, a1, d2, q2)
        T2_3 = self.T_matrix(alpha2, a2, d3, q3)
        T3_4 = self.T_matrix(alpha3, a3, d4, q4)
        T4_5 = self.T_matrix(alpha4, a4, d5, q5)
        T5_6 = self.T_matrix(alpha5, a5, d6, q6)
        T6_7 = self.T_matrix(alpha6, a6, d7, q7)
        T7_8 = self.T_matrix(alpha7, a7, d8, q8)

        T_0_7 = T0_1 * T1_2 * T2_3 * T3_4 * T4_5 * T5_6 * T6_7 * T7_8

        # Extract rotation matrices from the transformation matrices
        R_corr = Matrix([[0, 0, 1, 0],
                        [1, 0, 0, 0],   
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])
        
        R_0_7 = T_0_7[0:3, 0:3]

        # Extract end effector position and orientation from msg
        px,py,pz = self.end_effector_pose.position.x, self.end_effector_pose.position.y, self.end_effector_pose.position.z
        roll, pitch, yaw = self.end_effector_pose.orientation.x, self.end_effector_pose.orientation.y, self.end_effector_pose.orientation.z

        # Create Rotation Matrix
        R_roll = Matrix([[1, 0, 0],
                        [0, cos(roll), -sin(roll)], 
                        [0, sin(roll), cos(roll)]])
        R_pitch = Matrix([[cos(pitch), 0, sin(pitch)],
                        [0, 1, 0],  
                        [-sin(pitch), 0, cos(pitch)]])
        R_yaw = Matrix([[cos(yaw), -sin(yaw), 0],
                        [sin(yaw), cos(yaw), 0],                
                        [0, 0, 1]])
        R_corr = R_roll * R_pitch * R_yaw

        # TODO: Compute all joint angles
        
        # Add theta values to the joint_traj_point

        joint_traj_point.positions = [q1, q2, q3, q4, q5, q6, q7, q8]

        return joint_traj_point 

    def path_to_traj(self, path):
        """ function to calculate joint angles from cartesian position using inverse kinematics """
        
        traj = JointTrajectory()
        for i in range(len(path)):
            point = JointTrajectoryPoint()
            point.positions = self.ik(path[i])
            traj.points.append(point)

        return traj

    def traj_to_joints(self, traj):

        """ function to execute planned path """

        for i in traj.points:
            self.joint_pub1.publish(i.positions[0])
            self.joint_pub2.publish(i.positions[1])
            self.joint_pub3.publish(i.positions[2])
            self.joint_pub4.publish(i.positions[3])
            self.joint_pub5.publish(i.positions[4])
            self.joint_pub6.publish(i.positions[5])
            self.joint_pub7.publish(i.positions[6])
            rospy.sleep(0.1)

rospy.init_node('obstacle_detect_avoid')
detect_avoid = DetectAvoid()
rospy.spin()