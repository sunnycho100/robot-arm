#!/usr/bin/env python3

# ROS node to command a set of HiWonder Bus servos through the "xArm 1S" controller using the keyboard. 
# Peter Adamczyk, University of Wisconsin - Madison
# Updated 2025-07-15
 
import rclpy
from rclpy.node import Node
import traceback 
import numpy as np
# IMPORT the messages: 
# from sensor_msgs.msg import JointState
from xarmrob_interfaces.msg import ME439JointCommand



class CmdKeyboard(Node): 
    def __init__(self): 
        super().__init__('cmd_keyboard')

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

        self.cmd_all = list(map(int,self.bus_servo_neutral_cmds_base_to_tip))  # List of neutral values for each bus servo
        
        # command frequency parameter: how often we expect it to be updated    
        self.command_frequency = self.declare_parameter('command_frequency',5).value
        self.movement_time_ms = round(1000/self.command_frequency)  # milliseconds for movement time. 

        # Run a loop that gets input from the user
        while True:
        
            try:
                in_string = input('Enter the 7 bus servo commands (int 0-1000), formatted as a list: [a, b, c, d, e, f, g]. Press "Enter" to quit.\n')
                in_ints = list(map(int, in_string.strip('[]').split(',')))
                assert len(in_ints) == 7
            except: 
                retry = input('Bad Input. Try again? [y]/n: ')
                if (retry=='y') or (retry=='Y') or (retry==''):
                    continue
                else: 
                    return 
                    
            prnttmpl = 'Input was [' + '{}, '*6 + '{}]\n'
            print(prnttmpl.format(*in_ints))

            # if len(in_ints) == 7:
            self.cmd_all = np.clip(in_ints, a_min=0, a_max=1000)
            
            ####    CODE HERE
            ## For continuous motion, put the 'send' command below inside another While loop  
            ## that moves to the new commands gradually. 
            ## Use 'self.movement_time_ms' to regulate the timing. 
            ####    END CODE
            
            prnttmpl = 'Moving to [' + '{}, '*6 + '{}]\n'
            self.get_logger().info(prnttmpl.format(*self.cmd_all))
            self.send_command()
            # else: 
                # print('Bad list of commands')            
          
    def send_command(self):
        self.bus_servo_commands_msg.command = self.cmd_all
        self.bus_servo_commands_msg.enable = True
        self.bus_servo_commands_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_bus_servo_commands.publish(self.bus_servo_commands_msg)
    

    def shutdown_servos(self):

        # Send the "servoOff" command - turn "enable" to False
        self.bus_servo_commands_msg.command = self.cmd_all
        self.bus_servo_commands_msg.enable = False
        self.bus_servo_commands_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_bus_servo_commands.publish(self.bus_servo_commands_msg)


def main(args=None):
    try: 
        rclpy.init(args=args)
        cmd_keyboard_instance = CmdKeyboard()  
        # No need to "rclpy.spin(cmd_keyboard_instance)" here because there's a while() loop blocking and keeping it alive. 
        cmd_keyboard_instance.shutdown_servos()
        
    except: 
        traceback.print_exc()
        


if __name__ == '__main__':
    main()
