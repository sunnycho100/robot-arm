from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # Find the URDF file for the robot and get a robot description from it. 
    urdf_file_name = 'robot-xarm.urdf'
    urdf = os.path.join(
        get_package_share_directory('xarmrob'),
        'urdf',
        urdf_file_name
        )
    with open(urdf, 'r') as infp:
        robot_desc = infp.read()
        
    # Find the YAML file of parameters
    params_file_name = 'robot_xarm_info.yaml'
    params_file = os.path.join(
        get_package_share_directory('xarmrob'),
        'config',
        params_file_name
        )
    
    # Create the Launch Description
    launch_descr = LaunchDescription([
        Node(
            package='xarmrob',
            executable='command_xarm',
            name='command_xarm',
            parameters=[params_file]
        ),
        Node(
            package='xarmrob',
            executable='xarm_kinematics',
            name='xarm_kinematics',
            parameters=[params_file]
        ),
        Node(
            package='xarmrob',
            executable='endpoint_sliders',
            name='endpoint_sliders',
            parameters=[params_file]
        ),
        
        # This adds "robot_state_publisher" to publish Transforms through the "tf2" mechanism. It is a stock ROS package, and it uses the URDF file and "joint_states" topic to work. 
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc}],
            arguments=[urdf]
        )
    ])
    
    return launch_descr
    

