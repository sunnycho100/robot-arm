#!/usr/bin/env python3

# ROS node to command joint angles in the "xArm 1S" through a publisher 
# Peter Adamczyk, University of Wisconsin - Madison
# Updated 2025-07-15
 
import rclpy
from rclpy.node import Node
import tkinter as tk
import threading
import traceback 
import numpy as np
# IMPORT the messages: 
from sensor_msgs.msg import JointState
# from xarmrob_interfaces.msg import ME439JointCommand



class AngleSliders(Node): 
    def __init__(self): 
        super().__init__('angle_sliders')

        # =============================================================================
        #   # Publisher for the joint angles. 
        # =============================================================================
        self.pub_joint_angles_desired = self.create_publisher(JointState,'/joint_angles_desired',1)
        
        # Create the message, with a nominal pose
        self.joint_neutral_angs_base_to_tip = self.declare_parameter('joint_neutral_angs_base_to_tip', [0., -1.5707, 1.5707, 0., 0., 0., 0.]).value
        self.ang_all = self.joint_neutral_angs_base_to_tip
        self.joint_angles_desired_msg = JointState()
        self.joint_angles_desired_msg.name = ['base_joint', 'shoulder_joint', 'elbow_joint', 'forearm_joint', 'wrist_joint', 'fingers_joint', 'gripper'];
        self.joint_angles_desired_msg.position = self.ang_all      # upright neutral position

        # command frequency parameter: how often we expect it to be updated    
        self.command_frequency = self.declare_parameter('command_frequency',5).value
        self.movement_time_ms = round(1000/self.command_frequency)  # milliseconds for movement time. 

        # Ranges of Angles calibrated for each joint
        self.map_ang_rad_01 = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_joint_01',[-90,0,90]).value))
        self.map_ang_rad_12 = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_joint_12',[-180,-90,0]).value))
        self.map_ang_rad_23 = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_joint_23',[0,90,180]).value))
        self.map_ang_rad_34 = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_joint_34',[-112,-90,0,90,112]).value))
        self.map_ang_rad_45 = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_joint_45',[-112,-90,0,90,112]).value))
        self.map_ang_rad_56 = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_joint_56',[-112,-90,0,90,112]).value))
        self.map_ang_rad_gripper = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_gripper',[0, 90]).value))
        self.ang_all = list(map(int,self.joint_neutral_angs_base_to_tip))  # List of neutral values for each bus servo
        
        # Set up a timer to send the commands at the specified rate. 
        self.timer = self.create_timer(self.movement_time_ms/1000, self.send_joint_angles_desired)
        
    
    def send_joint_angles_desired(self):
        self.joint_angles_desired_msg.position = self.ang_all
        self.joint_angles_desired_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_joint_angles_desired.publish(self.joint_angles_desired_msg)
    
    # Specific functions for the specific Servos/Sliders       
    def move_joint_00(self, ang):
        self.ang_all[0] = float(ang)

    def move_joint_01(self, ang):
        self.ang_all[1] = float(ang)
        
    def move_joint_02(self, ang):
        self.ang_all[2] = float(ang)
        
    def move_joint_03(self, ang):
        self.ang_all[3] = float(ang)
        
    def move_joint_04(self, ang):
        self.ang_all[4] = float(ang)
        
    def move_joint_05(self, ang):
        self.ang_all[5] = float(ang)
        
    def move_joint_06(self, ang):
        self.ang_all[6] = float(ang)


    #%% Section to set up a nice Tkinter GUI with sliders. 
    def tk_gui(self): 
        # set up GUI
        root = tk.Tk()
        root.title("Manual Robot XArm Joint Angle Control (Radians)")
        
        # draw a big slider for servo 0 position
        min_ang = min(self.map_ang_rad_01)
        max_ang = max(self.map_ang_rad_01)
        mid_ang = self.joint_neutral_angs_base_to_tip[0]
        scale0 = tk.Scale(root,
            from_ = min_ang,
            to = max_ang,
            resolution = (max_ang-min_ang)/1000,
            command = self.move_joint_00,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Joint_00 BaseJoint: Current calibration valid from {:.3f} to {:.3f}'.format(min_ang,max_ang))
        scale0.set(mid_ang)
        scale0.pack(anchor = tk.CENTER)
        
        # draw a big slider for servo 1 position
        min_ang = min(self.map_ang_rad_12)
        max_ang = max(self.map_ang_rad_12)
        mid_ang = self.joint_neutral_angs_base_to_tip[1]
        scale1 = tk.Scale(root,
            from_ = min_ang,
            to = max_ang,
            resolution = (max_ang-min_ang)/1000,
            command = self.move_joint_01,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Joint_01 ShoulderJoint: Current calibration valid from {:.3f} to {:.3f}'.format(min_ang,max_ang))
        scale1.set(mid_ang)
        scale1.pack(anchor = tk.CENTER)
        
        # draw a big slider for servo 2 position
        min_ang = min(self.map_ang_rad_23)
        max_ang = max(self.map_ang_rad_23)
        mid_ang = self.joint_neutral_angs_base_to_tip[2]
        scale2 = tk.Scale(root,
            from_ = min_ang,
            to = max_ang,
            resolution = (max_ang-min_ang)/1000,
            command = self.move_joint_02,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Joint_02 ElbowJoint: Current calibration valid from {:.3f} to {:.3f}'.format(min_ang,max_ang))
        scale2.set(mid_ang)
        scale2.pack(anchor = tk.CENTER)
        
        # draw a big slider for servo 3 position
        min_ang = min(self.map_ang_rad_34)
        max_ang = max(self.map_ang_rad_34)
        mid_ang = self.joint_neutral_angs_base_to_tip[3]
        scale3 = tk.Scale(root,
            from_ = min_ang,
            to = max_ang,
            resolution = (max_ang-min_ang)/1000,
            command = self.move_joint_03,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Joint_03 ForearmJoint: Current calibration valid from {:.3f} to {:.3f}'.format(min_ang,max_ang))
        scale3.set(mid_ang)
        scale3.pack(anchor = tk.CENTER)
        
        # draw a big slider for servo 4 position
        min_ang = min(self.map_ang_rad_45)
        max_ang = max(self.map_ang_rad_45)
        mid_ang = self.joint_neutral_angs_base_to_tip[4]
        scale4 = tk.Scale(root,
            from_ = min_ang,
            to = max_ang,
            resolution = (max_ang-min_ang)/1000,
            command = self.move_joint_04,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Joint_04 WristJoint: Current calibration valid from {:.3f} to {:.3f}'.format(min_ang,max_ang))
        scale4.set(mid_ang)
        scale4.pack(anchor = tk.CENTER)
        
        # draw a big slider for servo 5 position
        min_ang = min(self.map_ang_rad_56)
        max_ang = max(self.map_ang_rad_56)
        mid_ang = self.joint_neutral_angs_base_to_tip[5]
        scale5 = tk.Scale(root,
            from_ = min_ang,
            to = max_ang,
            resolution = (max_ang-min_ang)/1000,
            command = self.move_joint_05,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Joint_05 FingerJoint: Current calibration valid from {:.3f} to {:.3f}'.format(min_ang,max_ang))
        scale5.set(mid_ang)
        scale5.pack(anchor = tk.CENTER)
        
        # draw a big slider for servo 6 position
        min_ang = min(self.map_ang_rad_gripper)
        max_ang = max(self.map_ang_rad_gripper)
        mid_ang = self.joint_neutral_angs_base_to_tip[6]
        scale6 = tk.Scale(root,
            from_ = min_ang,
            to = max_ang,
            resolution = (max_ang-min_ang)/1000,
            command = self.move_joint_06,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Joint_06 Gripper: Current calibration valid from {:.3f} to {:.3f}'.format(min_ang,max_ang))
        scale6.set(mid_ang)
        scale6.pack(anchor = tk.CENTER)
        

        # run Tk event loop
        root.mainloop()
        



def main(args=None):
    try: 
        rclpy.init(args=args)
        angle_sliders_instance = AngleSliders()  
        
        # Run the GUI in a separate Thread so it does not block the ROS functionality. 
        thrd = threading.Thread(target=angle_sliders_instance.tk_gui )
        thrd.start()
        # "Spin" the node so that the timer callback will execute. 
        rclpy.spin(angle_sliders_instance)

        
    except: 
        traceback.print_exc()
        


if __name__ == '__main__':
    main()
