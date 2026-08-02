#!/usr/bin/env python3

# ROS node to command an Endpoint to a HiWonder xArm 1S using Sliders. 
# Peter Adamczyk, University of Wisconsin - Madison
# Updated 2026-07-16
 
import rclpy
from rclpy.node import Node
import tkinter as tk
import threading
import traceback 
import numpy as np
from xarmrob_interfaces.msg import ME439PointXYZ
from example_interfaces.msg import Float32 

## Define a temporary function using Python "lambda" functionality to print colored text
# see https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal/3332860#3332860
# search that page for "CircuitSacul" to find that answer
coloredtext = lambda r, g, b, text: f'\033[38;2;{r};{g};{b}m{text}\033[38;2;255;255;255m'



class EndpointSliders(Node): 
    def __init__(self): 
        super().__init__('endpoint_sliders')
        
        self.xyz_goal = [0.165, 0.0, 0.155] # roughly upright neutral with wrist at 45 degrees. Formally: [0.1646718829870224, 0.0, 0.1546700894832611]
        self.gripper = float(0.)

        # =============================================================================
        #   # Publisher for the Endpoint goal. 
        # =============================================================================
        self.pub_endpoint_desired = self.create_publisher(ME439PointXYZ,'/endpoint_desired',1)
        # Create the message, with a nominal pose
        self.endpoint_desired_msg = ME439PointXYZ()
        self.endpoint_desired_msg.xyz = self.xyz_goal 
        
        # =============================================================================
        #   # Publisher for the Gripper angle 
        # =============================================================================
        self.pub_gripper = self.create_publisher(Float32, '/gripper_desired',1)
        self.gripper_msg = Float32()
        self.gripper_msg.data = self.gripper 

        # command frequency parameter: how often we expect it to be updated    
        self.command_frequency = self.declare_parameter('command_frequency',5).value
        self.movement_time_ms = round(1000/self.command_frequency)  # milliseconds for movement time. 
        
        # Set up a timer to send the commands at the specified rate. 
        self.timer = self.create_timer(self.movement_time_ms/1000, self.send_endpoint_desired)

    # Callback to publish the endpoint and gripper angle at the specified rate. 
    def send_endpoint_desired(self):
        self.endpoint_desired_msg.xyz = self.xyz_goal 
        self.pub_endpoint_desired.publish(self.endpoint_desired_msg)
        self.gripper_msg.data = self.gripper 
        self.pub_gripper.publish(self.gripper_msg)
    
    #%% GUI callback functions for the specific XYZ directions       
    def move_x(self, x):
        self.xyz_goal[0] = float(x)

    def move_y(self, y):
        self.xyz_goal[1] = float(y)
        
    def move_z(self, z):
        self.xyz_goal[2] = float(z)
        
    def move_gripper(self, g): 
        self.gripper = float(g)
        

    #%% Section to set up a nice Tkinter GUI with sliders. 
    def tk_gui(self): 
        # set up GUI
        root = tk.Tk()
        root.title("Manual Robot XArm Endpoint Control (Meters)")
        
        # draw a big slider for X position
        min_val = -0.3
        max_val = 0.4
        mid_val = 0.2
        scale0 = tk.Scale(root,
            from_ = min_val,
            to = max_val,
            resolution = (max_val-min_val)/1000,
            command = self.move_x,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'X position (m)')
        scale0.set(mid_val)
        scale0.pack(anchor = tk.CENTER)
        
        # draw a big slider for Y position
        min_val = -0.4
        max_val = 0.4
        mid_val = 0.0
        scale1 = tk.Scale(root,
            from_ = min_val,
            to = max_val,
            resolution = (max_val-min_val)/1000,
            command = self.move_y,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Y position (m)')
        scale1.set(mid_val)
        scale1.pack(anchor = tk.CENTER)
        
        # draw a big slider for Z position
        min_val = 0.0
        max_val = 0.4
        mid_val = 0.1
        scale2 = tk.Scale(root,
            from_ = min_val,
            to = max_val,
            resolution = (max_val-min_val)/1000,
            command = self.move_z,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Z position (m)')
        scale2.set(mid_val)
        scale2.pack(anchor = tk.CENTER)
        
        # draw a big slider for Gripper
        min_val = 0.0
        max_val = 1.571
        default_val = 0.0
        scale2 = tk.Scale(root,
            from_ = min_val,
            to = max_val,
            resolution = (max_val-min_val)/1000,
            command = self.move_gripper,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Gripper (rad)')
        scale2.set(default_val)
        scale2.pack(anchor = tk.CENTER)
        
        
        # run Tk event loop
        root.mainloop()
        



def main(args=None):
    try: 
        rclpy.init(args=args)
        endpoint_sliders_instance = EndpointSliders()  
        
        # Run the GUI in a separate Thread so it does not block the ROS functionality. 
        thrd = threading.Thread(target=endpoint_sliders_instance.tk_gui )
        thrd.start()
        
        # "Spin" the node so that the timer callback will execute. 
        rclpy.spin(endpoint_sliders_instance)
        
    except: 
        traceback.print_exc()
        


if __name__ == '__main__':
    main()
