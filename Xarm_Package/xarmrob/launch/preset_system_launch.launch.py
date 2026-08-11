#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 08:24:45 2026

@author: pi
"""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Start the Arm Controller
        Node(
            package='xarmrob',
            executable='command_xarm',
            name='command_xarm',
            output='screen'
        ),
        # Start the Kinematics Engine
        Node(
            package='xarmrob',
            executable='xarm_kinematics',
            name='xarm_kinematics',
            output='screen'
        ),
        # Start the ArUco Vision Tracker
        Node(
            package='mobrob',
            executable='aruco',
            name='aruco_tracker',
            output='screen'
        ),
        # Start the Preset Controller Brain
        Node(
            package='xarmrob',
            executable='preset_controller',
            name='preset_controller',
            output='screen'
        )
    ])