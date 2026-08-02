#!/usr/bin/env python3

# ROS node to command the "xArm 1S" using manually typed joint angles.  
# Peter Adamczyk, University of Wisconsin - Madison
# Updated 2025-07-15
 
import rclpy
from rclpy.node import Node
import traceback 
import numpy as np
# IMPORT the messages: 
from sensor_msgs.msg import JointState
# from xarmrob_interfaces.msg import ME439JointCommand




class AngleKeyboard(Node): 
    def __init__(self): 
        super().__init__('angle_keyboard')

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

        
        # limits for each of the joints
        rotlim_01 = np.radians(np.array(self.declare_parameter('rotational_limits_joint_01',[-150,150]).value))
        rotlim_12 = np.radians(np.array(self.declare_parameter('rotational_limits_joint_12',[-180,0]).value))
        rotlim_23 = np.radians(np.array(self.declare_parameter('rotational_limits_joint_23',[0,180]).value))
        rotlim_34 = np.radians(np.array(self.declare_parameter('rotational_limits_joint_34',[-110,110]).value))
        rotlim_45 = np.radians(np.array(self.declare_parameter('rotational_limits_joint_45',[-100,100]).value))
        rotlim_56 = np.radians(np.array(self.declare_parameter('rotational_limits_joint_56',[-110,111]).value))
        rotlim_gripper = np.radians(np.array(self.declare_parameter('gripper_limits',[0,90]).value))

        # Run a loop that gets input from the user
        while True:
            
            try: 
                in_string = input('Enter the 7 joint angles (float in Radians), formatted as a list:\n  [alpha0, beta1, beta2, gamma3, beta4, gamma5, gripper]   Enter to exit.\n')
                in_floats = list(map(float, in_string.strip('[]').split(',')))
                assert len(in_floats) == 7
            except: 
                retry = input('Bad Input. Try again? [y]/n: ')
                if (retry=='y') or (retry=='Y') or (retry==''):
                    continue
                else: 
                    return
                    
            prnttmpl = 'Input was [' + '{:.2f}, '*6 + '{:.2f}]\n'
            print(prnttmpl.format(*in_floats))
            self.ang_all = in_floats
            # if len(in_floats) == 7:
            self.ang_all[0] = np.clip(in_floats[0], np.min(rotlim_01), np.max(rotlim_01))
            self.ang_all[1] = np.clip(in_floats[1], np.min(rotlim_12), np.max(rotlim_12))
            self.ang_all[2] = np.clip(in_floats[2], np.min(rotlim_23), np.max(rotlim_23))
            self.ang_all[3] = np.clip(in_floats[3], np.min(rotlim_34), np.max(rotlim_34))
            self.ang_all[4] = np.clip(in_floats[4], np.min(rotlim_45), np.max(rotlim_45))
            self.ang_all[5] = np.clip(in_floats[5], np.min(rotlim_56), np.max(rotlim_56))
            self.ang_all[6] = np.clip(in_floats[6], np.min(rotlim_gripper), np.max(rotlim_gripper))
            
            ####    CODE HERE
            ## For continuous motion, put the 'send' command below inside another While loop  
            ## that moves to the new angles gradually. 
            ## Use 'self.movement_time_ms' to regulate the timing. 
            ####    END CODE
            
            prnttmpl = 'Moving to [' + '{:.2f}, '*6 + '{:.2f}]\n'
            self.get_logger().info(prnttmpl.format(*self.ang_all))

            self.send_joint_angles_desired()
            # else: 
                # print('Bad list of joint angles')            
                
    
    def send_joint_angles_desired(self):
        self.joint_angles_desired_msg.position = self.ang_all
        self.joint_angles_desired_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_joint_angles_desired.publish(self.joint_angles_desired_msg)



def main(args=None):
    try: 
        rclpy.init(args=args)
        angle_keyboard_instance = AngleKeyboard()  
        # No need to "rclpy.spin(angle_keyboard_instance)" here because there's a while() loop blocking and keeping it alive. 
        
    except: 
        traceback.print_exc()
        


if __name__ == '__main__':
    main()


