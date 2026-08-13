#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:18:06 2026

@author: pi
"""

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from example_interfaces.msg import String
from xarmrob_interfaces.msg import ME439PointXYZ
import numpy as np
import cv2

class PresetController(Node):
    def __init__(self):
        super().__init__('preset_controller')
        self.sub_aruco = self.create_subscription(String, '/aruco', self.aruco_callback, 1)
        self.pub_endpoint = self.create_publisher(ME439PointXYZ, '/endpoint_desired', 1)
        
        self.ROBOT_BASE_ID = 4  # The "Little Tag"
        self.AMP_ID = 2         # The "Big Tag"
        self.IN2M = 0.0254
        self.command_sent = False

        # ==========================================================
        # SCALING CORRECTION
        # Scale = (Actual width of Little Tag) / (Actual width of Big Tag)
        # ==========================================================
        self.LITTLE_TAG_SCALE_FACTOR = 0.5  # <--- MUST UPDATE AFTER MEASURING!
        
        # Fixed Amp Measurements (Big Tag -> Knobs)
        # Big Tag Axes: +X (Right), -Y (Forward), 0.0 (Z-hover)
        self.knob_offsets_inches = {
            'volume': np.array([2.0, -3.0, 1.0]),
            'treble': np.array([3.25, -3.0, 1.0]),
            'high mid': np.array([4.375, -3.0, 1.0]),
            'low mid': np.array([5.5, -3.0, 1.0]),
            'bass': np.array([6.625, -3.0, 1.0]),
            'dist lev': np.array([7.875, -3.0, 1.0]),
            'drive': np.array([9.0, -3.0, 1.0]),
            'gain': np.array([10.875, -3.0, 1.0])
        }

    def create_transform_matrix(self, rvec, tvec):
        """Converts rotation and translation vectors into a 4x4 homogenous matrix."""
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = tvec
        return T

    def aruco_callback(self, msg):
        if self.command_sent:
            return

        raw_data = msg.data.strip().split(" ")
        T_cam_to_little = None
        T_cam_to_big = None
        
        for tag in raw_data:
            if not tag: continue
            parts = tag.split(",")
            try:
                tag_id = int(parts[0].split(":")[1])
                tvec = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                rvec = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
                
                if tag_id == self.ROBOT_BASE_ID:
                    # Fix the distance scaling for the Little Tag!
                    tvec_scaled = tvec * self.LITTLE_TAG_SCALE_FACTOR
                    T_cam_to_little = self.create_transform_matrix(rvec, tvec_scaled)
                elif tag_id == self.AMP_ID:
                    T_cam_to_big = self.create_transform_matrix(rvec, tvec)
            except Exception:
                pass 

        if T_cam_to_little is not None and T_cam_to_big is not None:
            
            # =========================================================
            # Eq (i): r_robot_base = r_little_tag + r_little_tag_to_base
            # =========================================================
            # Little Tag +Y (Green) points forward toward the true origin
            v_little_to_base_local = np.array([0.0, 3.375 * self.IN2M, 0.0, 1.0])
            r_robot_base_cam = np.dot(T_cam_to_little, v_little_to_base_local)
            
            # =========================================================
            # Eq (ii): r_knob = r_big_tag + r_big_tag_to_knob
            # =========================================================
            target = 'gain'
            knob_offset = self.knob_offsets_inches[target].copy() * self.IN2M
            v_big_to_knob_local = np.array([knob_offset[0], knob_offset[1], knob_offset[2], 1.0])
            r_knob_cam = np.dot(T_cam_to_big, v_big_to_knob_local)
            
            # =========================================================
            # Eq (iii): r_robot_base_to_knob = r_knob - r_robot_base
            # =========================================================
            # This gives the vector in the CAMERA's coordinate frame
            v_base_to_knob_cam = r_knob_cam[0:3] - r_robot_base_cam[0:3]
            
            # =========================================================
            # Final Step: Rotate the vector into the Robot's frame
            # =========================================================
            # Robot axes relative to Little Tag:
            # Robot +X (Forward) = Little Tag +Y
            # Robot +Y (Left) = Little Tag -X
            # Robot +Z (Up) = Little Tag +Z
            R_little_to_robot = np.array([
                [ 0.0,  1.0,  0.0],
                [-1.0,  0.0,  0.0],
                [ 0.0,  0.0,  1.0]
            ])
            
            # Get the rotation matrix of the Little Tag in the camera frame
            R_cam_to_little = T_cam_to_little[0:3, 0:3]
            
            # Find the rotation of the Robot Base in the camera frame
            R_cam_to_robot = np.dot(R_cam_to_little, R_little_to_robot)
            
            # To move a vector from the camera frame to the robot frame, 
            # we multiply by the INVERSE of the robot's rotation.
            R_robot_to_cam = np.linalg.inv(R_cam_to_robot)
            final_coords = np.dot(R_robot_to_cam, v_base_to_knob_cam)
            
            final_x = final_coords[0]
            final_y = final_coords[1]
            final_z = final_coords[2]
            
            self.get_logger().info(
                f"Knob Aquired\n"
                f"Commanding arm to {target.upper()} (Meters): X={final_x:.4f}, Y={final_y:.4f}, Z={final_z:.4f}"
            )
            
            msg_out = ME439PointXYZ()
            msg_out.xyz = [float(final_x), float(final_y), float(final_z)]
            self.pub_endpoint.publish(msg_out)
            
            self.command_sent = True

def main(args=None):
    rclpy.init(args=args)
    preset_controller = PresetController()
    rclpy.spin(preset_controller)
    preset_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()