#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 17:52:21 2026

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
        
        self.ROBOT_BASE_ID = 4  
        self.AMP_ID = 2         
        self.IN2M = 0.0254
        self.command_sent = False

        self.LITTLE_TAG_SCALE_FACTOR = 0.5  
        
        # ==========================================================
        # CUSTOM OFFSET VALUES (From your Spyder IDE)
        # ==========================================================
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

        # ==========================================================
        # TRAJECTORY GENERATION SETTINGS
        # ==========================================================
        self.trajectory_active = False
        self.trajectory_steps = 100  # Number of waypoints
        self.current_step = 0
        
        # Start hovering 15cm forward and 15cm in the air
        self.start_xyz = np.array([0.15, 0.0, 0.15]) 
        self.target_xyz = np.array([0.0, 0.0, 0.0])
        
        # Publish a new waypoint every 0.05 seconds (5-second total move)
        self.timer = self.create_timer(0.05, self.trajectory_timer_callback)

    def trajectory_timer_callback(self):
        if self.trajectory_active and self.current_step <= self.trajectory_steps:
            
            frac = self.current_step / self.trajectory_steps
            
            # Linear Interpolation
            current_xyz = self.start_xyz + (self.target_xyz - self.start_xyz) * frac
            
            msg_out = ME439PointXYZ()
            msg_out.xyz = [float(current_xyz[0]), float(current_xyz[1]), float(current_xyz[2])]
            self.pub_endpoint.publish(msg_out)
            
            self.current_step += 1
            
            if self.current_step > self.trajectory_steps:
                self.get_logger().info("Target Reached. Trajectory Complete.")

    def create_transform_matrix(self, rvec, tvec):
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
                    tvec_scaled = tvec * self.LITTLE_TAG_SCALE_FACTOR
                    T_cam_to_little = self.create_transform_matrix(rvec, tvec_scaled)
                elif tag_id == self.AMP_ID:
                    T_cam_to_big = self.create_transform_matrix(rvec, tvec)
            except Exception:
                pass 

        if T_cam_to_little is not None and T_cam_to_big is not None:
            
            v_little_to_base_local = np.array([0.0, 3.375 * self.IN2M, 0.0, 1.0])
            r_robot_base_cam = np.dot(T_cam_to_little, v_little_to_base_local)
            
            target = 'gain' # Set to your currently tested knob
            knob_offset = self.knob_offsets_inches[target].copy() * self.IN2M
            v_big_to_knob_local = np.array([knob_offset[0], knob_offset[1], knob_offset[2], 1.0])
            r_knob_cam = np.dot(T_cam_to_big, v_big_to_knob_local)
            
            v_base_to_knob_cam = r_knob_cam[0:3] - r_robot_base_cam[0:3]
            
            R_little_to_robot = np.array([
                [ 0.0,  1.0,  0.0],
                [-1.0,  0.0,  0.0],
                [ 0.0,  0.0,  1.0]
            ])
            
            R_cam_to_little = T_cam_to_little[0:3, 0:3]
            R_cam_to_robot = np.dot(R_cam_to_little, R_little_to_robot)
            
            R_robot_to_cam = np.linalg.inv(R_cam_to_robot)
            final_coords = np.dot(R_robot_to_cam, v_base_to_knob_cam)
            
            # TRIGGER THE TRAJECTORY
            self.target_xyz = np.array([final_coords[0], final_coords[1], final_coords[2]])
            self.command_sent = True
            self.trajectory_active = True
            
            self.get_logger().info(
                f"Target Acquired!\n"
                f"Beginning 5-second slow approach to {target.upper()} (Meters): X={self.target_xyz[0]:.4f}, Y={self.target_xyz[1]:.4f}, Z={self.target_xyz[2]:.4f}"
            )

def main(args=None):
    rclpy.init(args=args)
    preset_controller = PresetController()
    rclpy.spin(preset_controller)
    preset_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()