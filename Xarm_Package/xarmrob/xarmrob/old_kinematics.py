#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 00:35:02 2026

@author: pi
"""

#!/usr/bin/env python3
 
# ROS node to receive an Endpoint for a HiWonder xArm 1S 
# and convert it to Joint Angles, which are published 
# Peter Adamczyk, University of Wisconsin - Madison
# Updated 2026-07-16

import rclpy
from rclpy.node import Node 
import numpy as np
import traceback
# Import the service types we will need
from xarmrob_interfaces.srv import ME439XArmForwardKinematics, ME439XArmInverseKinematics
# Import the message types we will need
from xarmrob_interfaces.msg import ME439PointXYZ
from sensor_msgs.msg import JointState
from example_interfaces.msg import Float32

## Define a temporary function using Python "lambda" functionality to print colored text
# see https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal/3332860#3332860
# search that page for "CircuitSacul" to find that answer
coloredtext = lambda r, g, b, text: f'\033[38;2;{r};{g};{b}m{text}\033[38;2;255;255;255m'




class XArmKinematics(Node): 
    def __init__(self):
        super().__init__('xarm_kinematics')
        
        # Create Services 
        self.srv_FK = self.create_service(ME439XArmForwardKinematics, 'xarm_forward_kinematics', self.compute_FK)
        self.srv_IK = self.create_service(ME439XArmInverseKinematics, 'xarm_inverse_kinematics', self.compute_IK)
        # Create Subscribers and Publishers
        self.sub_endpoint = self.create_subscription(ME439PointXYZ, '/endpoint_desired', self.compute_IK_pub_JTANG,1)
        self.sub_gripper = self.create_subscription(Float32, '/gripper_desired', self.set_gripper,1)
        self.pub_JTANG = self.create_publisher(JointState, '/joint_angles_desired',1)
        
        # Create the message, with a nominal pose        
        self.joint_neutral_angs_base_to_tip = self.declare_parameter('joint_neutral_angs_base_to_tip', [0., -1.5707, 1.5707, 0., 0., 0., 0.]).value
        self.ang_all = self.joint_neutral_angs_base_to_tip
        self.joint_angles_desired_msg = JointState()
        self.joint_angles_desired_msg.name = ['base_joint', 'shoulder_joint', 'elbow_joint', 'forearm_joint', 'wrist_joint', 'fingers_joint', 'gripper'];
        self.joint_angles_desired_msg.position = self.ang_all 

        # Load parameters for the functions below: 
        # Matched lists of angles and microsecond commands
        self.map_ang_rad_01 = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_joint_01',[-90,0,90]).value))
        # self.map_cmd_01 = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_joint_01',[120, 500, 880]).value)
        self.map_ang_rad_12 = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_joint_12',[-180,-90,0]).value))
        # self.map_cmd_12 = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_joint_12',[870,500,120]).value)
        self.map_ang_rad_23 = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_joint_23',[0,90,180]).value))
        # self.map_cmd_23 = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_joint_23',[140,500,880]).value)
        self.map_ang_rad_34 = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_joint_34',[-112,-90,0,90,112]).value))
        # self.map_cmd_34 = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_joint_34', [1000,890,505,140,0]).value)
        self.map_ang_rad_45 = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_joint_45',[-112,-90,0,90,112]).value))
        # self.map_cmd_45 = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_joint_45',[0,120,490,880,1000]).value)
        self.map_ang_rad_56 = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_joint_56',[-112,-90,0,90,112]).value))
        # self.map_cmd_56 = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_joint_56',[0,120,500,880,1000]).value)
        self.map_ang_rad_gripper = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_gripper',[0, 90]).value))
        # self.map_cmd_gripper = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_gripper',[90,610]).value)
        
        # limits for each of the joints
        self.rotlim_01 = np.radians(np.array(self.declare_parameter('rotational_limits_joint_01',[-150,150]).value))
        self.rotlim_12 = np.radians(np.array(self.declare_parameter('rotational_limits_joint_12',[-180,0]).value))
        self.rotlim_23 = np.radians(np.array(self.declare_parameter('rotational_limits_joint_23',[0,180]).value))
        self.rotlim_34 = np.radians(np.array(self.declare_parameter('rotational_limits_joint_34',[-110,110]).value))
        self.rotlim_45 = np.radians(np.array(self.declare_parameter('rotational_limits_joint_45',[-100,100]).value))
        self.rotlim_56 = np.radians(np.array(self.declare_parameter('rotational_limits_joint_56',[-110,111]).value))
        self.rotlim_gripper = np.radians(np.array(self.declare_parameter('gripper_limits',[0,90]).value))
        
        # Sign of 'positive' rotations w.r.t. the y axis
        self.y_rotation_sign = np.sign(self.declare_parameter('y_rotation_sign',1).value)
        # Vectors from each frame origin to the next frame origin, in the proximal
        self.r_01 = np.column_stack(self.declare_parameter('frame_offset_01',[0., 0., 0.074]).value).transpose()
        self.r_12 = np.column_stack(self.declare_parameter('frame_offset_12',[0.010, 0., 0.]).value).transpose()
        self.r_23 = np.column_stack(self.declare_parameter('frame_offset_23',[0.101, 0., 0.]).value).transpose()
        self.r_34 = np.column_stack(self.declare_parameter('frame_offset_34',[0.0627, 0., 0.0758]).value).transpose()        
        self.r_45 = np.column_stack(self.declare_parameter('frame_offset_45',[0., 0., 0.]).value).transpose()
        self.r_56 = np.column_stack(self.declare_parameter('frame_offset_56',[0., 0., 0.]).value).transpose()
        self.r_6end = np.column_stack(self.declare_parameter('endpoint_offset_in_frame_6',[0.133, 0., -0.003]).value).transpose()  # Endpoint is the center of the gripper pincer when OPEN 
        self.r_6end_open = self.r_6end
        self.gripper_arm_length = self.declare_parameter('gripper_arm_length',0.030).value
        
        # States for Gripper (open = 0 rad)
        self.gripper = 0.



    def set_gripper(self,msg_in): 
        self.gripper = msg_in.data
        self.r_6end = self.r_6end_open + np.column_stack([self.gripper_arm_length*np.sin(self.gripper), 0, 0]).transpose()
        
            
        
    # Service for Forward Kinematics    
    def compute_FK(self, request_FK, response_FK):
        ang = request_FK.joint_angles
        
        # Limit angles to the allowed ranges based on limit Parameters
        ang_lim = self.limit_joint_angles(ang)
        
        pos_endpoint, pos_all, T_06 = self.fwdkin(ang_lim)        
        
        # Pack the response
        response_FK.joint_angles = ang_lim
        response_FK.endpoint = pos_endpoint
        if np.allclose(ang_lim,ang):
            response_FK.modified = False
        else: 
            response_FK.modified = True
            
        return response_FK
        
    # Function to compute endpoint location from joint angles. 
    def fwdkin(self, joint_angles):
        # Unpack joint angles.
        alpha0 = joint_angles[0]
        beta1 = joint_angles[1] * self.y_rotation_sign
        beta2 = joint_angles[2] * self.y_rotation_sign
        gamma3 = joint_angles[3]
        beta4 = joint_angles[4] * self.y_rotation_sign
        gamma5 = joint_angles[5]    
        
        # =============================================================================
        # # Transformation from frame 0 to 1 (+Z axis rotation)
        # =============================================================================
        # "Rotation matrix of frame 1 in frame 0's coordinates" (columns are unit vectors of Frame 1 in Frame 0 coordinates)
        R_01 = np.array([ [np.cos(alpha0), -np.sin(alpha0), 0.], 
                             [np.sin(alpha0), np.cos(alpha0), 0.],
                             [       0.,           0.,  1.] ])
        # "Homogeneous Transform of Frame 1 in Frame 0's Coordinates"
        T_01 = np.vstack( (np.column_stack( (R_01, self.r_01) ) , [0., 0., 0., 1.]) )
        
        # =============================================================================
        # # Transformation from frame 1 to 2 (+Y axis rotation)
        # =============================================================================
        R_12 = np.array([ [ np.cos(beta1), 0., np.sin(beta1)], 
                           [       0. ,     1.,        0.    ],
                           [-np.sin(beta1), 0., np.cos(beta1)] ])
        T_12 = np.vstack( (np.column_stack( (R_12, self.r_12) ) , [0., 0., 0., 1.]) )
            
        # =============================================================================
        # # Transformation from frame 2 to 3 (+Y axis rotation)
        # =============================================================================
        R_23 = np.array([ [ np.cos(beta2), 0., np.sin(beta2)], 
                           [       0. ,     1.,        0.    ],
                           [-np.sin(beta2), 0., np.cos(beta2)] ])
        T_23 = np.vstack( (np.column_stack( (R_23, self.r_23) ) , [0., 0., 0., 1.]) )
            
        # =============================================================================
        # # Transformation from frame 3 to 4 (+X axis rotation)
        # =============================================================================
        R_34 = np.array([ [ 1. ,        0.     ,        0.      ], 
                           [ 0. , np.cos(gamma3), -np.sin(gamma3)], 
                           [ 0. , np.sin(gamma3),  np.cos(gamma3)] ])
        T_34 = np.vstack( (np.column_stack( (R_34, self.r_34) ) , [0., 0., 0., 1.]) )
                
        # =============================================================================
        # # Transformation from frame 4 to 5 (+Y axis rotation)
        # =============================================================================
        R_45 = np.array([ [ np.cos(beta4), 0., np.sin(beta4)], 
                           [       0. ,     1.,        0.    ],
                           [-np.sin(beta4), 0., np.cos(beta4)] ])
        T_45 = np.vstack( (np.column_stack( (R_45, self.r_45) ) , [0., 0., 0., 1.]) )
                
        # =============================================================================
        # # Transformation from frame 5 to 6 (+X axis rotation)
        # =============================================================================
        R_56 = np.array([ [ 1. ,        0.     ,        0.      ], 
                           [ 0. , np.cos(gamma5), -np.sin(gamma5)], 
                           [ 0. , np.sin(gamma5),  np.cos(gamma5)] ])
        T_56 = np.vstack( (np.column_stack( (R_56, self.r_56) ) , [0., 0., 0., 1.]) )
        
        # return T_01, T_12, T_23, T_34, T_45, T_56         

        # Vector of Zero from the frame origin in question, augmented with a 1 so it can be used with the Homogeneous Transform
        zerovec = np.column_stack(np.array([0.,0.,0.,1.])).transpose()     
        
        pos_0 = zerovec[0:3,0] # base link location: 0
        pos_1 = (T_01@zerovec)[0:3,0]
        T_02 = T_01@T_12
        pos_2 = (T_02@zerovec)[0:3,0]
        T_03 = T_02@T_23
        pos_3 = (T_03@zerovec)[0:3,0]
        T_04 = T_03@T_34
        pos_4 = (T_04@zerovec)[0:3,0]
        T_05 = T_04@T_45
        pos_5 = (T_05@zerovec)[0:3,0]
        T_06 = T_05@T_56
        pos_6 = (T_06@zerovec)[0:3,0]
        
        pos_endpoint = (T_06@np.vstack((self.r_6end,1)) )[0:3,0]
        pos_all = np.column_stack( (pos_0, pos_1, pos_2, pos_3, pos_4, pos_5, pos_6, pos_endpoint) )     
        
        return pos_endpoint, pos_all, T_06
        
    

    # =============================================================================
    # # Function to compute joint angles to reach the target endpoint
    # =============================================================================
    def compute_IK_pub_JTANG(self, msg_endpoint):  
        self.endpoint_goal = np.array(msg_endpoint.xyz)

        
        # effector_pose = request_IK.effector_pose
        # if not all(np.array(effector_pose)==0.):
        #     self.get_logger().info(coloredtext(255,0,0,'Warning: Custom End Effector Pose is not yet implemented!!')) 
        
        # First compute the inverse kinematics for perfect endpoint positioning
        ang = self.invkin(self.endpoint_goal)
        
        # Then Limit the angle at each joint to its achievable range
        ang_lim = self.limit_joint_angles(ang)
        
        # Compute the endpoint achieved by the limited angles
        pos_endpoint, pos_all, T_06 = self.fwdkin(ang_lim)   
        
        # Pack the response
        joint_angles = ang_lim
        endpoint = pos_endpoint.flatten()

        # Report if the solution is Mismatched from the original. 
        if np.allclose(pos_endpoint.flatten(),self.endpoint_goal.flatten()):
            modified = False
        else: 
            modified = True
            self.get_logger().info(coloredtext(50,255,50,'\n\tMoving to [' + '{:.3f}, '*2 + '{:.3f}]').format(*endpoint))
            
        self.joint_angles_desired_msg.position = np.append(ang_lim,self.gripper) # 0 is a placeholder for the gripper.
        self.joint_angles_desired_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_JTANG.publish(self.joint_angles_desired_msg)
    
    # =============================================================================
    # # Function to compute joint angles to reach the target endpoint
    # =============================================================================
    def compute_IK(self, request_IK, response_IK):  
        endpoint = np.array(request_IK.endpoint)
        
        effector_pose = request_IK.effector_pose
        if not all(np.array(effector_pose)==0.):
            self.get_logger().info(coloredtext(255,0,0,'Warning: Custom End Effector Pose is not yet implemented!!')) 
        
        # First compute the inverse kinematics for perfect endpoint positioning
        ang = self.invkin(endpoint)
        
        # Then Limit the angle at each joint to its achievable range
        ang_lim = self.limit_joint_angles(ang)
        
        # Compute the endpoint achieved by the limited angles
        pos_endpoint, pos_all, T_06 = self.fwdkin(ang_lim)   
        
        # Pack the response
        response_IK.joint_angles = ang_lim
        response_IK.endpoint = pos_endpoint.flatten()

        # Report if the solution is Mismatched from the original. 
        if np.allclose(pos_endpoint.flatten(),endpoint.flatten()):
            response_IK.modified = False
        else: 
            response_IK.modified = True

        return response_IK    
    
        
    def invkin(self, endpoint):
        xyz = endpoint
        
        # Compute base rotation plus 2-link IK... 
        # Assuming that the "forearm" and "fingers" do not rotate 
        gamma3 = 0
        gamma5 = 0
        
        # # Gripper Assumption option 0: set the gripper to point directly down
        # Rgrip = np.array([ [np.cos(np.pi/2), 0, np.sin(np.pi/2)], [0,1,0], [-np.sin(np.pi/2), 0, np.cos(np.pi/2)]])
    
        # Gripper Assumption option 1; set the Gripper is at a 45 degree angle downward in the RTZ world frame. 
        #gripper_angle = np.pi/4
        #gripper_angle = np.pi/2
        gripper_angle = np.pi/2
        
        # # Gripper Assumption option 2: set the gripper angle to be adjusted by height
        # z_end = xyz[2]
        # gripper_angle = z_end/.4334*-np.pi + np.pi/2
        
        # Wrist to Gripper in the plane: 
        Rgrip = np.array([ [np.cos(gripper_angle), 0, np.sin(gripper_angle)], [0,1,0], [-np.sin(gripper_angle), 0, np.cos(gripper_angle)]])
        gripper_offset_RTZ = Rgrip.dot(self.r_6end)
        
        # First the out-of-plane rotation
        alpha0 = np.arctan2(xyz[1], xyz[0])
        
        # Now compute the radial and vertical distances spanned by the two links of the arm
        R = np.linalg.norm(xyz[0:2])   # Remember that this means "start at 0, stop BEFORE 2"
        dR = R - gripper_offset_RTZ[0] - self.r_12[0]         # subtract off the x of all the links that are not part of the 2-link kinematic solution. NEW 2024: Use matrix form of the gripper offset. 
        
        dz = xyz[2] - gripper_offset_RTZ[2] - self.r_01[2] - self.r_12[2]    # subtract off the Z of all the links that are not part of the 2-link kinematic solution. NEW 2024: Use matrix form of the gripper offset.
        
        # Now compute the "overall elevation" angle from the "shoulder" to the "wrist" 
        # NOTE this assumes rotations about the +y axis (positive rotations push the wrist down)
        psi = -np.arctan2(dz, dR)  # use negative because of the positive-rotations-down convention. 
        # Now the difference between the actual shoulder angle and the overall elevation angle
        # ... being aware that there are two solutions and we want the "elbow up" configuration. 
        L1 = np.linalg.norm(self.r_23)  # vector magnitude of the link that spans from shoulder to elbow ("upper arm")
        L2 = np.linalg.norm(self.r_34)  # vector magnitude of the link that spans from elbow to wrist ("lower arm")
        H = np.linalg.norm(np.array((dz,dR))) # vector magnitude of the vector from shoulder to wrist. (H = hypotenuse)
        try:
            phi = np.arccos( (L2**2 - L1**2 - H**2)/(-2*L1*H) )  # arccos will always return a positive value. 
        
        
            # Compute the "elbow up" solution for beta1
            beta1 = psi - phi   #  phi is always positive (from arccos function) so "-phi" is the elbow pose in a more negative position (elbow up for the +y axis rotations) 
            
            # Compute the corresponding solution for beta2VL (VL = "virtual link" direct from joint 2 to joint 3 (elbow to wrist)
            # Use the ArcTangent (two quadrant)
            beta2VL = np.arctan2(H*np.sin(phi), H*np.cos(phi)-L1)
        #    print(beta2VL)
            
            # Compute the offset in angle between  the VL (virtual link straight from joint 3 to joint 4) and the true link axis. 
            # True link should be more positive by this amount. 
            beta2_offset_from_VL = np.arctan2(self.r_34[2], self.r_34[0])  
        
        except: 
            print('NAN in solution')
            
        # Real-world beta2, assuming +y axis rotations
        beta2 = beta2VL + beta2_offset_from_VL 
           
        # Depending on the sign of positive rotations, give back the rotations. 
        beta1 = beta1 * self.y_rotation_sign
        beta2 = beta2 * self.y_rotation_sign
        
        # Compute beta4 to cancel out beta1 and beta2 (works regardless of the sign) 
        beta4 = -(beta1+beta2) + gripper_angle
        
        joint_angles = np.asfarray(list(map(float,[alpha0, beta1, beta2, gamma3, beta4, gamma5])))
        
        return joint_angles
    
        
    def limit_joint_angles(self, angles):
        angles_limited = angles
        
        # Clip (saturate) the angles at the achievable limits. 
        angles_limited[0] = np.clip(angles_limited[0], np.min(self.rotlim_01), np.max(self.rotlim_01))
        angles_limited[1] = np.clip(angles_limited[1], np.min(self.rotlim_12), np.max(self.rotlim_12))
        angles_limited[2] = np.clip(angles_limited[2], np.min(self.rotlim_23), np.max(self.rotlim_23))
        angles_limited[3] = np.clip(angles_limited[3], np.min(self.rotlim_34), np.max(self.rotlim_34))
        angles_limited[4] = np.clip(angles_limited[4], np.min(self.rotlim_45), np.max(self.rotlim_45))
        angles_limited[5] = np.clip(angles_limited[5], np.min(self.rotlim_56), np.max(self.rotlim_56))
    
        return angles_limited





def main(args=None):
    try: 
        rclpy.init(args=args)
        xarm_kinematics_instance = XArmKinematics()  
        rclpy.spin(xarm_kinematics_instance) 
        
    except: 
        traceback.print_exc()
        


if __name__ == '__main__':
    main()


