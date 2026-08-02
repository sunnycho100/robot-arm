#!/usr/bin/env python3

# ROS node to command an Endpoint to a HiWonder xArm 1S 
# Peter Adamczyk, University of Wisconsin - Madison
# Updated 2024-11-12

import rclpy
from rclpy.node import Node 
import numpy as np
import traceback
from sensor_msgs.msg import JointState
from xarmrob_interfaces.srv import ME439XArmInverseKinematics #, ME439XArmForwardKinematics


## Define a temporary function using Python "lambda" functionality to print colored text
# see https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal/3332860#3332860
# search that page for "CircuitSacul" to find that answer
coloredtext = lambda r, g, b, text: f'\033[38;2;{r};{g};{b}m{text}\033[38;2;255;255;255m'



class EndpointKeyboard(Node): 
    def __init__(self): 
        super().__init__('endpoint_keyboard')
        
        # =============================================================================
        #   # Publisher for the Endpoint goal as a set of joint angles. 
        # =============================================================================
        self.pub_joint_angles_desired = self.create_publisher(JointState,'/joint_angles_desired',1)
        # Create the message, with a nominal pose
        self.joint_neutral_angs_base_to_tip = self.declare_parameter('joint_neutral_angs_base_to_tip', [0., -1.5707, 1.5707, 0., 0., 0., 0.]).value
        self.ang_all = self.joint_neutral_angs_base_to_tip
        self.joint_angles_desired_msg = JointState()
        self.joint_angles_desired_msg.name = ['base_joint', 'shoulder_joint', 'elbow_joint', 'forearm_joint', 'wrist_joint', 'fingers_joint', 'gripper'];
        self.joint_angles_desired_msg.position = self.ang_all      # upright neutral position


        # =============================================================================
        #  Client for the Inverse Dynamics service
        # =============================================================================
        self.cli_inverse_kinematics = self.create_client(ME439XArmInverseKinematics, 'xarm_inverse_kinematics')
        while not self.cli_inverse_kinematics.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting...')
        # The Request message for the service. 
        self.request_inverse_kinematics = ME439XArmInverseKinematics.Request()
        
        
    def endpoint_requests(self):
        # Run a loop that gets input from the user
        while True:
            
            try: 
                in_string = input('Enter Endpoint Location as a list (in meters): [x, y, z] \n    Ctrl-C and Enter to exit.\n')   
                in_floats = list(map(float, in_string.strip('[]').split(',')))
                assert len(in_floats) == 3
            except: 
                retry = input('Bad Input. Try again? [y]/n: ')
                if (retry=='y') or (retry=='Y') or (retry==''):
                    continue
                else: 
                    return
                    
            prnttmpl = coloredtext(50,255,50,'\n\tEndpoint Goal Input was [' + '{:.3f}, '*2 + '{:.3f}]')
            self.get_logger().info(prnttmpl.format(*in_floats))
            xyz_goal = in_floats
            
            # Compute Inverse Kinematics
            self.request_inverse_kinematics.endpoint = xyz_goal
            future = self.cli_inverse_kinematics.call_async(self.request_inverse_kinematics)
            rclpy.spin_until_future_complete(self, future)
            
            response = future.result()
            endpoint = response.endpoint
            
            # Check if the response is what you asked for, or something else. 
            if response.modified:
                self.get_logger().info(coloredtext(255,0,0,'Unreachable Endpoint!'))
                if not any(np.isnan(endpoint)):
                    go_anyway = input(coloredtext(50,255,50,'Move to nearest point [' + '{:.3f}, '*2 + '{:.3f}] - [y]/n?\n').format(*endpoint))
                    if len(go_anyway)==0:
                        go_anyway='y'
                else:
                    go_anyway = 'n'
                
                if not (go_anyway[0].upper() == 'Y') : 
                    self.get_logger().info(coloredtext(255,0,0,'Not moving - try again.'))
                    continue
                
            # If the program gets here it is okay to go ahead. 
            # Move to endpoint. 
            self.ang_all = response.joint_angles            
            
            self.get_logger().info(coloredtext(50,255,50,'\n\tMoving to [' + '{:.3f}, '*2 + '{:.3f}]').format(*endpoint))
            self.get_logger().info(coloredtext(25,255,75,'\n\tAngles [' + '{:.3f}, '*5 + '{:.3f}]').format(*self.ang_all))
            self.send_joint_angles_desired()
                

    def send_joint_angles_desired(self):
        self.joint_angles_desired_msg.position = np.append(self.ang_all,0.)
        self.joint_angles_desired_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_joint_angles_desired.publish(self.joint_angles_desired_msg)



def main(args=None):
    try: 
        rclpy.init(args=args)
        endpoint_keyboard_instance = EndpointKeyboard()  
        endpoint_keyboard_instance.endpoint_requests()
        # No need to "rclpy.spin(endpoint_keyboard_instance)" here because there's a while() loop blocking and keeping it alive. 
        
    except: 
        traceback.print_exc()
        


if __name__ == '__main__':
    main()


