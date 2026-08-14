#!/usr/bin/env python3
"""His preset_system_launch.launch.py with one node removed.

The three that stay are reactive: they compute IK and drive servos when a
message arrives and do nothing otherwise. The one that goes, preset_controller,
is his sequencer, which starts driving all eight knobs the moment the tags
lock. Our brain publishes the same three topics it did, so nothing below this
line knows the difference.

His launch file is untouched and still runs his original demo.
"""
import os

from launch import LaunchDescription
from launch_ros.actions import Node


def _aruco_env():
    """What his aruco node needs to survive on this Pi.

    GSTREAMER off: his bare cv2.VideoCapture(0) lets OpenCV try GStreamer
    first, which deadlocks against pipewire in a futex at import. The process
    starts, never registers a node, and /aruco never publishes, which reads as
    a tag problem and is not one. Without GStreamer the same call falls through
    to V4L2 and opens in 0.2 s.

    Qt offscreen, but only when there is no display: he draws the detected axes
    and outlines with cv2.imshow, so run headless the node publishes exactly
    one message and then dies on "could not connect to display". The lock is
    one-shot, so the demo appears to work while the amp-moved check has quietly
    stopped updating. Offscreen keeps it alive. When a display IS there, this
    is left alone so his debug window still opens.
    """
    env = {'OPENCV_VIDEOIO_PRIORITY_GSTREAMER': '0'}
    if not os.environ.get('DISPLAY'):
        env['QT_QPA_PLATFORM'] = 'offscreen'
    return env


def generate_launch_description():
    return LaunchDescription([
        Node(package='xarmrob', executable='command_xarm',
             name='command_xarm', output='screen'),
        Node(package='xarmrob', executable='xarm_kinematics',
             name='xarm_kinematics', output='screen'),
        # See _aruco_env() above: two environment settings, no edit to his file.
        Node(package='mobrob', executable='aruco',
             name='aruco_tracker', output='screen',
             additional_env=_aruco_env()),
        # preset_controller deliberately absent: knobbrain replaces it.
    ])
