from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    params_file_name = 'robot_xarm_info.yaml'
    params_file = os.path.join(
      get_package_share_directory('xarmrob'),
      'config',
      params_file_name
      )

    launch_descr = LaunchDescription([
        Node(
            package='xarmrob',
            executable='command_xarm',
            name='command_xarm',
            parameters=[params_file]
        )
    ## NOTE this file is set up to only run the "command_arm" code.
    ## THE USER must also run "cmd_manual" or some other command node to interface with it.
    ])
    
    return launch_descr
    

