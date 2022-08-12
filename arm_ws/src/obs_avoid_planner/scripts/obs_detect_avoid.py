import imp
from re import sub
import numpy as np
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
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

    def joint_cb(self, msg):
        self.joint_state = msg

    def depth_cb(self, msg):
        self.depth_img = msg

    def rgb_cb(self, msg):
        self.rgb_img = msg

    def detect_obstacles(self):
        """ function to detect obstacles from RGB and depth image """

        config_space = self.depth_img.data

        # TODO: Search through config space 

        pass

    def joint_state_from_xyz(self, position):
        """ function to calculate joint angles from cartesian position using inverse kinematics """
        joint_val = JointState()

        return joint_val

    def avoid_obstacles(self):
        
        """ function to plan trajectory for avoiding obstacles """
        
        traj = JointTrajectory()
        traj.header.stamp = rospy.Time.now()


        self.traj_to_joints(traj)

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