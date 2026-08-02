#!/usr/bin/env python3

# ROS node to command an Endpoint to a HiWonder xArm 1S using Sliders. 
# Peter Adamczyk, University of Wisconsin - Madison
# Updated 2024-11-14
 
import rclpy
from rclpy.node import Node
import tkinter as tk
import threading
import traceback 
import numpy as np
from example_interfaces.msg import Float32, Float32MultiArray

## Define a temporary function using Python "lambda" functionality to print colored text
# see https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal/3332860#3332860
# search that page for "CircuitSacul" to find that answer
coloredtext = lambda r, g, b, text: f'\033[38;2;{r};{g};{b}m{text}\033[38;2;255;255;255m'



class PoseSliders(Node): 
    def __init__(self): 
        super().__init__('endpoint_sliders')
        
        self.pose_goal = [0.0, 0.0, 0.0] # Zero rotation goal
        self.gripper_goal = 0.

        # =============================================================================
        #   # Publisher for the Endpoint goal. 
        # =============================================================================
        self.pub_pose_desired = self.create_publisher(Float32MultiArray,'/pose_desired',1)
        # Create the message, with a nominal pose
        self.pose_desired_msg = Float32MultiArray()
        self.pose_desired_msg.data = self.pose_goal 

        # =============================================================================
        #   # Publisher for the Gripper goal. 
        # =============================================================================
        self.pub_gripper_desired = self.create_publisher(Float32,'/gripper_desired',1)
        # Create the message, with a nominal pose
        self.gripper_desired_msg = Float32()
        self.gripper_desired_msg.data = self.gripper_goal 

        # command frequency parameter: how often we expect it to be updated    
        self.command_frequency = self.declare_parameter('command_frequency',5).value
        self.movement_time_ms = round(1000/self.command_frequency)  # milliseconds for movement time. 
        
        # Set up a timer to send the commands at the specified rate. 
        self.timer = self.create_timer(self.movement_time_ms/1000, self.send_pose_desired)

    # Callback to publish the endpoint at the specified rate. 
    def send_pose_desired(self):
        self.pose_desired_msg.data = self.pose_goal 
        self.pub_pose_desired.publish(self.pose_desired_msg)
        # Also send the gripper data. 
        self.send_gripper_desired()

    # Callback to publish the endpoint at the specified rate. 
    def send_gripper_desired(self):
        self.gripper_desired_msg.data = self.gripper_goal 
        self.pub_gripper_desired.publish(self.gripper_desired_msg)

    
    #%% GUI callback functions for the specific RPY (XYZ rotation) directions       
    def move_roll(self, roll):
        self.pose_goal[0] = float(roll)

    def move_pitch(self, pitch):
        self.pose_goal[1] = float(pitch)
        
    def move_yaw(self, yaw):
        self.pose_goal[2] = float(yaw)

    def move_gripper(self, gripper):
        self.gripper_goal = float(gripper)
        

    #%% Section to set up a nice Tkinter GUI with sliders. 
    def tk_gui(self): 
        # set up GUI
        root = tk.Tk()
        root.title("Manual Robot XArm Pose Control (Euler Angles RPY)")
        
        # draw a big slider for X position
        min_val = -np.pi
        max_val = np.pi
        mid_val = 0.0
        scale0 = tk.Scale(root,
            from_ = min_val,
            to = max_val,
            resolution = (max_val-min_val)/1000,
            command = self.move_roll,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Roll (radians)')
        scale0.set(mid_val)
        scale0.pack(anchor = tk.CENTER)
        
        # draw a big slider for Y position
        min_val = -np.pi
        max_val = np.pi
        mid_val = 0.0
        scale1 = tk.Scale(root,
            from_ = min_val,
            to = max_val,
            resolution = (max_val-min_val)/1000,
            command = self.move_pitch,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Pitch (radians))')
        scale1.set(mid_val)
        scale1.pack(anchor = tk.CENTER)
        
        # draw a big slider for Z position
        min_val = -np.pi
        max_val = np.pi
        mid_val = 0.0
        scale2 = tk.Scale(root,
            from_ = min_val,
            to = max_val,
            resolution = (max_val-min_val)/1000,
            command = self.move_yaw,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Yaw (radians)')
        scale2.set(mid_val)
        scale2.pack(anchor = tk.CENTER)
        
        
        # draw a big slider for Gripper
        min_val = 0.0
        max_val = np.pi/2
        mid_val = np.pi/4
        scale3 = tk.Scale(root,
            from_ = min_val,
            to = max_val,
            resolution = (max_val-min_val)/1000,
            command = self.move_gripper,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Gripper (radians)')
        scale3.set(min_val)     # Set this one to default OPEN
        scale3.pack(anchor = tk.CENTER)
        
        
        # run Tk event loop
        root.mainloop()
        



def main(args=None):
    try: 
        rclpy.init(args=args)
        pose_sliders_instance = PoseSliders()  
        
        # Run the GUI in a separate Thread so it does not block the ROS functionality. 
        thrd = threading.Thread(target=pose_sliders_instance.tk_gui )
        thrd.start()
        
        # "Spin" the node so that the timer callback will execute. 
        rclpy.spin(pose_sliders_instance)
        
    except: 
        traceback.print_exc()
        


if __name__ == '__main__':
    main()