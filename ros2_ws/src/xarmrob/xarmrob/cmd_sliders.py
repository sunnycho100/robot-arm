#!/usr/bin/env python3

# ROS node to command a set of HiWonder Bus servos through the "xArm 1S" controller using Sliders. 
# Peter Adamczyk, University of Wisconsin - Madison
# Updated 2025-07-15
 
import rclpy
from rclpy.node import Node
import tkinter as tk
import threading
import traceback 
import numpy as np
# IMPORT the messages: 
# from sensor_msgs.msg import JointState
from xarmrob_interfaces.msg import ME439JointCommand



class CmdSliders(Node): 
    def __init__(self): 
        super().__init__('cmd_sliders')

        # =============================================================================
        #   # Publisher for the digital servo motor commands. 
        # =============================================================================
        self.pub_bus_servo_commands = self.create_publisher(ME439JointCommand,'/bus_servo_commands',1)
        self.bus_servo_commands_msg = ME439JointCommand()
        self.bus_servo_commands_msg.name = ['cmd00','cmd01','cmd02','cmd03','cmd04','cmd05','cmd06']
        self.bus_servo_commands_msg.enable = True

        # Matched Lists of Servo Indices and ID Numbers
        self.bus_servo_indices_base_to_tip = self.declare_parameter('bus_servo_indices_base_to_tip',[0,1,2,3,4,5,6]).value
        self.bus_servo_IDs_base_to_tip = self.declare_parameter('bus_servo_IDs_base_to_tip',[6,5,4,7,3,2,1]).value    
        self.bus_servo_neutral_cmds_base_to_tip = self.declare_parameter('bus_servo_neutral_cmds_base_to_tip', [500,500,500,500,500,500,500]).value

        # command frequency parameter: how often we expect it to be updated    
        self.command_frequency = self.declare_parameter('command_frequency',5).value
        self.movement_time_ms = round(1000/self.command_frequency)  # milliseconds for movement time. 

        # Ranges of Commands currently calibrated
        self.map_cmd_01 = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_joint_01',[120, 500, 880]).value)
        self.map_cmd_12 = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_joint_12',[970,600,220]).value)
        self.map_cmd_23 = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_joint_23',[140,500,880]).value)
        self.map_cmd_34 = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_joint_34', [1000,890,505,140,0]).value)
        self.map_cmd_45 = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_joint_45',[0,120,490,880,1000]).value)
        self.map_cmd_56 = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_joint_56',[0,120,500,880,1000]).value)
        self.map_cmd_gripper = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_gripper',[90,610]).value)

        self.cmd_all = list(map(int,self.bus_servo_neutral_cmds_base_to_tip))  # List of neutral values for each bus servo
        
        # Set up a timer to send the commands at the specified rate. 
        self.timer = self.create_timer(self.movement_time_ms/1000, self.send_command)

    
    def send_command(self):
        self.bus_servo_commands_msg.command = self.cmd_all
        self.bus_servo_commands_msg.enable = True
        self.bus_servo_commands_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_bus_servo_commands.publish(self.bus_servo_commands_msg)
    
    # Specific functions for the specific Servos/Sliders       
    def move_servo_00(self, cmd):
        self.cmd_all[0] = int(cmd)

    def move_servo_01(self, cmd):
        self.cmd_all[1] = int(cmd)
        
    def move_servo_02(self, cmd):
        self.cmd_all[2] = int(cmd)
        
    def move_servo_03(self, cmd):
        self.cmd_all[3] = int(cmd)
        
    def move_servo_04(self, cmd):
        self.cmd_all[4] = int(cmd)
        
    def move_servo_05(self, cmd):
        self.cmd_all[5] = int(cmd)
        
    def move_servo_06(self, cmd):
        self.cmd_all[6] = int(cmd)

    def shutdown_servos(self):
        # Send the "servoOff" command - turn "enable" to False
        self.bus_servo_commands_msg.command = self.cmd_all
        self.bus_servo_commands_msg.enable = False
        self.bus_servo_commands_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_bus_servo_commands.publish(self.bus_servo_commands_msg)


    #%% Section to set up a nice Tkinter GUI with sliders. 
    def tk_gui(self): 
        # set up GUI
        root = tk.Tk()
        root.title("Manual Robot XArm Bus Servo Control (Commands 0-1000)")
        
        # draw a big slider for servo 0 position
        min_cmd = min(self.map_cmd_01)
        max_cmd = max(self.map_cmd_01)
        mid_cmd = self.bus_servo_neutral_cmds_base_to_tip[0]
        scale0 = tk.Scale(root,
            from_ = 0,
            to = 1000,
            command = self.move_servo_00,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Servo_00: Current calibration valid from {0} to {1}'.format(min_cmd,max_cmd))
        scale0.set(mid_cmd)
        scale0.pack(anchor = tk.CENTER)
        
        # draw a big slider for servo 1 position
        min_cmd = min(self.map_cmd_12)
        max_cmd = max(self.map_cmd_12)
        mid_cmd = self.bus_servo_neutral_cmds_base_to_tip[1]
        scale1 = tk.Scale(root,
            from_ = 0,
            to = 1000,
            command = self.move_servo_01,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Servo_01: Current calibration valid from {0} to {1}'.format(min_cmd,max_cmd))
        scale1.set(mid_cmd)
        scale1.pack(anchor = tk.CENTER)
        
        # draw a big slider for servo 2 position
        min_cmd = min(self.map_cmd_23)
        max_cmd = max(self.map_cmd_23)
        mid_cmd = self.bus_servo_neutral_cmds_base_to_tip[2]
        scale2 = tk.Scale(root,
            from_ = 0,
            to = 1000,
            command = self.move_servo_02,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Servo_02: Current calibration valid from {0} to {1}'.format(min_cmd,max_cmd))
        scale2.set(mid_cmd)
        scale2.pack(anchor = tk.CENTER)
        
        # draw a big slider for servo 3 position
        min_cmd = min(self.map_cmd_34)
        max_cmd = max(self.map_cmd_34)
        mid_cmd = self.bus_servo_neutral_cmds_base_to_tip[3]
        scale3 = tk.Scale(root,
            from_ = 0,
            to = 1000,
            command = self.move_servo_03,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Servo_03: Current calibration valid from {0} to {1}'.format(min_cmd,max_cmd))
        scale3.set(mid_cmd)
        scale3.pack(anchor = tk.CENTER)
        
        # draw a big slider for servo 4 position
        min_cmd = min(self.map_cmd_45)
        max_cmd = max(self.map_cmd_45)
        mid_cmd = self.bus_servo_neutral_cmds_base_to_tip[4]
        scale4 = tk.Scale(root,
            from_ = 0,
            to = 1000,
            command = self.move_servo_04,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Servo_04: Current calibration valid from {0} to {1}'.format(min_cmd,max_cmd))
        scale4.set(mid_cmd)
        scale4.pack(anchor = tk.CENTER)
        
        # draw a big slider for servo 5 position
        min_cmd = min(self.map_cmd_56)
        max_cmd = max(self.map_cmd_56)
        mid_cmd = self.bus_servo_neutral_cmds_base_to_tip[5]
        scale5 = tk.Scale(root,
            from_ = 0,
            to = 1000,
            command = self.move_servo_05,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Servo_05: Current calibration valid from {0} to {1}'.format(min_cmd,max_cmd))
        scale5.set(mid_cmd)
        scale5.pack(anchor = tk.CENTER)
        
        # draw a big slider for servo 6 position
        min_cmd = min(self.map_cmd_gripper)
        max_cmd = max(self.map_cmd_gripper)
        mid_cmd = self.bus_servo_neutral_cmds_base_to_tip[6]
        scale6 = tk.Scale(root,
            from_ = 0,
            to = 1000,
            command = self.move_servo_06,
            orient = tk.HORIZONTAL,
            length = 1000,
            label = 'Servo_06: Current calibration valid from {0} to {1}'.format(min_cmd,max_cmd))
        scale6.set(mid_cmd)
        scale6.pack(anchor = tk.CENTER)
        

        # run Tk event loop
        root.mainloop()
        



def main(args=None):
    try: 
        rclpy.init(args=args)
        cmd_sliders_instance = CmdSliders()  
        
        # Run the GUI in a separate Thread so it does not block the ROS functionality. 
        thrd = threading.Thread(target=cmd_sliders_instance.tk_gui )
        thrd.start()
        # "Spin" the node so that the timer callback will execute. 
        rclpy.spin(cmd_sliders_instance)
        
        # When done, shut down the servos. 
        cmd_sliders_instance.shutdown_servos()
        
    except: 
        traceback.print_exc()
        


if __name__ == '__main__':
    main()