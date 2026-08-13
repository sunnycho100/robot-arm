#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        
        # Fixed Amp Measurements (Big Tag -> Knobs)
        # CALIBRATED OFFSETS (+0.125 X, +0.5 Y)
        self.knob_offsets_inches = {
            'volume': np.array([2.125, -5.2, 2.0]),
            'treble': np.array([3.375, -5.2, 2.0]),
            'high mid': np.array([4.5, -5.2, 2.0]),
            'low mid': np.array([5.625, -5.2, 2.0]),
            'bass': np.array([6.75, -5.2, 2.0]),
            'dist lev': np.array([8.0, -5.2, 2.0]),
            'drive': np.array([9.125, -5.2, 2.0]),
            'gain': np.array([11.0, -5.2, 2.0]),
            'initial': np.array([6.5, -5.2, 2.0]) 
        }

        # The sequence the arm will follow
        self.sequence = [
            'initial', 'volume', 
            'initial', 'treble', 
            'initial', 'high mid', 
            'initial', 'low mid', 
            'initial', 'bass', 
            'initial', 'dist lev', 
            'initial', 'drive', 
            'initial', 'gain', 
            'initial'
        ]
        
        self.step_index = 0
        
        # These store your locked-in camera matrices
        self.locked_T_cam_to_little = None
        self.locked_T_cam_to_big = None
        
        # The timer fires every 4 seconds to execute your math
        self.timer = self.create_timer(4.0, self.timer_callback)

    def create_transform_matrix(self, rvec, tvec):
        """Converts rotation and translation vectors into a 4x4 homogenous matrix."""
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = tvec
        return T

    def aruco_callback(self, msg):
        # CALIBRATION LOCK: If we already locked the matrices, ignore the camera!
        if self.locked_T_cam_to_little is not None and self.locked_T_cam_to_big is not None:
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
                    T_cam_to_little = self.create_transform_matrix(rvec, tvec)
                elif tag_id == self.AMP_ID:
                    T_cam_to_big = self.create_transform_matrix(rvec, tvec)
            except Exception:
                pass 

        # If both tags are seen in the exact same frame, lock them in forever!
        if T_cam_to_little is not None and T_cam_to_big is not None:
            self.locked_T_cam_to_little = T_cam_to_little
            self.locked_T_cam_to_big = T_cam_to_big
            self.get_logger().info("CALIBRATION LOCKED! Starting sequence...")

    def timer_callback(self):
        # 1. Wait until ArUco callback has locked the matrices
        if self.locked_T_cam_to_little is None or self.locked_T_cam_to_big is None:
            self.get_logger().info('Waiting to see both tags...', throttle_duration_sec=4.0)
            return 
            
        # 2. Stop when the sequence is done
        if self.step_index >= len(self.sequence):
            self.get_logger().info('Sequence Complete!', throttle_duration_sec=4.0)
            return 
            
        # 3. Get the current target from the sequence list
        target = self.sequence[self.step_index]

        # =========================================================
        # YOUR EXACT MATH (Unchanged)
        # =========================================================
        v_little_to_base_local = np.array([0.0, 2.4375 * self.IN2M, 0.0, 1.0])
        r_robot_base_cam = np.dot(self.locked_T_cam_to_little, v_little_to_base_local)
        
        knob_offset = self.knob_offsets_inches[target].copy() * self.IN2M
        v_big_to_knob_local = np.array([knob_offset[0], knob_offset[1], knob_offset[2], 1.0])
        r_knob_cam = np.dot(self.locked_T_cam_to_big, v_big_to_knob_local)
        
        v_base_to_knob_cam = r_knob_cam[0:3] - r_robot_base_cam[0:3]
        
        R_little_to_robot = np.array([
            [ 0.0, -1.0,  0.0],
            [ -1.0,  0.0,  0.0],
            [ 0.0,  0.0,  1.0]
        ])
        
        R_cam_to_little = self.locked_T_cam_to_little[0:3, 0:3]
        R_cam_to_robot = np.dot(R_cam_to_little, R_little_to_robot)
        R_robot_to_cam = np.linalg.inv(R_cam_to_robot)
        final_coords = np.dot(R_robot_to_cam, v_base_to_knob_cam)
        # =========================================================

        # Publish the final coordinates
        msg_out = ME439PointXYZ()
        msg_out.xyz = [float(final_coords[0]), float(final_coords[1]), float(final_coords[2])]
        self.pub_endpoint.publish(msg_out)
        
        self.get_logger().info(f"Commanding arm to {target.upper()}")
        
        # Advance to the next knob for the next 4-second loop
        self.step_index += 1

def main(args=None):
    rclpy.init(args=args)
    preset_controller = PresetController()
    rclpy.spin(preset_controller)
    preset_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()