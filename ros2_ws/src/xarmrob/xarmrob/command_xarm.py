#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup 
from rclpy.executors import MultiThreadedExecutor
import traceback 
import numpy as np
from scipy import interpolate
# IMPORT the messages: 
from sensor_msgs.msg import JointState
from xarmrob_interfaces.msg import ME439JointCommand
# IMPORT the arm controller
import xarm


class CommandXArm(Node): 
    def __init__(self): 
        super().__init__('command_arm')
                    
        # =============================================================================
        #   Subscribe to "/joint_angles_desired" to get joint angle values. 
        # =============================================================================
        ## This will come from joint angle sliders or anything else that specifies Angles rather than direct commands. 
        self.sub_joint_angles = self.create_subscription(JointState, '/joint_angles_desired', self.compute_commands, 1, callback_group=ReentrantCallbackGroup())
        #   NOTE the Callback to the "self.compute_commands" method, which will compute the commands that should be sent to each specific motor. 
        
        # =============================================================================
        #   Subscribe to "/bus_servo_commands" to get commands for direct use with the digital bus servo motors. 
        # =============================================================================
        ## This will come from command sliders or anything else that specifies Commands directly rather than joint angles.
        ## Note the message type "JointState" -- this maintains the structure of a JointState object even though the position[] field will be specified by command value rather than angle. 
        self.sub_bus_servo_commands = self.create_subscription(ME439JointCommand, '/bus_servo_commands', self.set_servos, 1, callback_group=ReentrantCallbackGroup())
        #   NOTE the Callback to the "self.move_servos_and_set_joint_state" method, which will command the motors directly and also compute what joint angle the system model thinks that is. 
        
        # =============================================================================
        #   # Publisher for the digital servo motor commands. 
        # =============================================================================
        self.pub_bus_servo_commands = self.create_publisher(ME439JointCommand,'/bus_servo_commands',1)
        self.bus_servo_commands_msg = ME439JointCommand()

        # =============================================================================
        #   # Publisher for the Joint States. 
        # =============================================================================
        self.pub_joint_states = self.create_publisher(JointState,'/joint_states',1)
        self.joint_state_msg = JointState()
        

        
        # Load parameters to keep handy for the functions below: 

        # Matched Lists of Servo Indices and ID Numbers
        self.bus_servo_indices_base_to_tip = self.declare_parameter('bus_servo_indices_base_to_tip',[0,1,2,3,4,5,6]).value
        self.bus_servo_IDs_base_to_tip = self.declare_parameter('bus_servo_IDs_base_to_tip',[6,5,4,7,3,2,1]).value    
        self.servo_neutral_cmds_base_to_tip = self.declare_parameter('bus_servo_neutral_cmds_base_to_tip', [500,500,500,500,500,500,500]).value
        
        self.cmd_all = self.servo_neutral_cmds_base_to_tip
        self.servo_pos = self.servo_neutral_cmds_base_to_tip
        

        # Try to set up the "xarm" library to commend the digital servo motors at the joints. 
        # If it fails, assume there's no arm present and just publish the commands instead. 
        try: 
            # arm is the first xArm detected which is connected to USB
            # Note: under the hood it is using "hid" API ("sudo pip3 install hidapi") to get access to the controller. 
            # self.arm = xarm.Controller('USB',debug=True)
            self.arm = xarm.Controller('USB')
            IDs = self.bus_servo_IDs_base_to_tip
            self.servos=[xarm.Servo(x) for x in IDs]
            self.arm_is_present = True
            
        except: 
            self.get_logger().info('No xArm controller Detected. Publishing only!')
            self.arm_is_present = False

        # command frequency parameter: how often we expect it to be updated    
        self.command_frequency = self.declare_parameter('command_frequency',5).value
        self.movement_time_ms = round(1000/self.command_frequency)  # milliseconds for movement time. 

        # Matched lists of angles and microsecond commands
        self.map_ang_rad_01 = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_joint_01',[-90,0,90]).value))
        self.map_cmd_01 = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_joint_01',[120, 500, 880]).value)
        self.map_ang_rad_12 = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_joint_12',[-180,-90,0]).value))
        self.map_cmd_12 = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_joint_12',[870,500,120]).value)
        self.map_ang_rad_23 = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_joint_23',[0,90,180]).value))
        self.map_cmd_23 = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_joint_23',[140,500,880]).value)
        self.map_ang_rad_34 = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_joint_34',[-112,-90,0,90,112]).value))
        self.map_cmd_34 = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_joint_34', [1000,890,505,140,0]).value)
        self.map_ang_rad_45 = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_joint_45',[-112,-90,0,90,112]).value))
        self.map_cmd_45 = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_joint_45',[0,120,490,880,1000]).value)
        self.map_ang_rad_56 = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_joint_56',[-112,-90,0,90,112]).value))
        self.map_cmd_56 = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_joint_56',[0,120,500,880,1000]).value)
        self.map_ang_rad_gripper = np.radians(np.array(self.declare_parameter('rotational_angles_for_mapping_gripper',[0, 90]).value))
        self.map_cmd_gripper = np.array(self.declare_parameter('bus_servo_cmd_for_mapping_gripper',[90,610]).value)

        ## Interpolating functions from Joint Angles to Commands, using the arm parameters
        self.f_interp_rad_to_cmd_01_params = interpolate.interp1d(self.map_ang_rad_01, self.map_cmd_01)
        self.f_interp_rad_to_cmd_12_params = interpolate.interp1d(self.map_ang_rad_12, self.map_cmd_12)
        self.f_interp_rad_to_cmd_23_params = interpolate.interp1d(self.map_ang_rad_23, self.map_cmd_23)
        self.f_interp_rad_to_cmd_34_params = interpolate.interp1d(self.map_ang_rad_34, self.map_cmd_34)
        self.f_interp_rad_to_cmd_45_params = interpolate.interp1d(self.map_ang_rad_45, self.map_cmd_45)
        self.f_interp_rad_to_cmd_56_params = interpolate.interp1d(self.map_ang_rad_56, self.map_cmd_56)
        self.f_interp_rad_to_cmd_gripper_params = interpolate.interp1d(self.map_ang_rad_gripper, self.map_cmd_gripper)

        ## Interpolating functions from Commands to Joint Angles, using the arm parameters
        self.f_interp_cmd_to_rad_01_params = interpolate.interp1d(self.map_cmd_01, self.map_ang_rad_01)
        self.f_interp_cmd_to_rad_12_params = interpolate.interp1d(self.map_cmd_12, self.map_ang_rad_12)
        self.f_interp_cmd_to_rad_23_params = interpolate.interp1d(self.map_cmd_23, self.map_ang_rad_23)
        self.f_interp_cmd_to_rad_34_params = interpolate.interp1d(self.map_cmd_34, self.map_ang_rad_34)
        self.f_interp_cmd_to_rad_45_params = interpolate.interp1d(self.map_cmd_45, self.map_ang_rad_45)
        self.f_interp_cmd_to_rad_56_params = interpolate.interp1d(self.map_cmd_56, self.map_ang_rad_56)
        self.f_interp_cmd_to_rad_gripper_params = interpolate.interp1d(self.map_cmd_gripper, self.map_ang_rad_gripper)

        # Initialize arm position, slowly: 
        self.initializing = True
        if self.arm_is_present:
            self.command_bus_servos(self.bus_servo_indices_base_to_tip , self.servo_neutral_cmds_base_to_tip)
        
        self.timer = self.create_timer(self.movement_time_ms/1000, self.set_joint_state)


    # =============================================================================
    #   # Callback function: receives a desired joint angle, computes servo commands. 
    def compute_commands(self, msg_in): 
        # unpack joint angle settings
        jt_all = msg_in.position
        
        # Pack the Commands message and publish it. 
        try: 
            # convert the joint state to bus servo commands
            cmd_all = self.convert_joint_state_to_commands(jt_all)
            self.bus_servo_commands_msg.command = cmd_all
            self.bus_servo_commands_msg.name = ['cmd00','cmd01','cmd02','cmd03','cmd04','cmd05','cmd06']
            self.bus_servo_commands_msg.enable = True
            self.bus_servo_commands_msg.header.stamp = self.get_clock().now().to_msg()
            # Publish the bus servo commands, and let the handler for that message receive it. 
            self.pub_bus_servo_commands.publish(self.bus_servo_commands_msg)
            # # Alternatively, call the function to move the servos directly, rather than publishing the message. 
            # self.move_servos_and_set_joint_state(self.bus_servo_commands_msg)
        except ValueError: 
            # traceback.print_exc(limit=1)
            # self.get_logger().error('ERROR: Value out of range. Not Publishing Joint State.')
            exc = traceback.format_exc(limit=1)
            self.get_logger().error(exc.splitlines()[-1])
        
    # =============================================================================
    #   # Callback function: receives servo commands and publishes /joint_states, e.g. for an RVIZ simulation
    def set_servos(self, msg_in):
        # look for a bunch of NaNs to indicate the Shutdown command. 
        # if all(np.isnan(msg_in.command)):
        
        if not(msg_in.enable):
            self.shutdown_arm()
            return
        
        # unpack the commands coming in. 
        self.cmd_all = msg_in.command 
                    
        # # send the servo commands if (and only if) there's an arm attached. 
        # if self.arm_is_present:
            # self.command_bus_servos( self.bus_servo_indices_base_to_tip, self.cmd_all)

        # # Pack the Joint State and publish it. 
        # try:
        #     # convert the joint state to bus servo commands
        #     jt_all = self.convert_commands_to_joint_state(cmd_all)
        #     self.joint_state_msg.position = jt_all
        #     self.joint_state_msg.name = ['base_joint', 'shoulder_joint', 'elbow_joint', 'forearm_joint', 'wrist_joint', 'fingers_joint', 'gripper']
        #     self.joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        #     self.pub_joint_states.publish(self.joint_state_msg)
        # except ValueError: 
        #     # traceback.print_exc(limit=1)
        #     # self.get_logger().error('ERROR: Value out of range. Not Publishing Joint State.')
        #     exc = traceback.format_exc(limit=1)
        #     self.get_logger().error(exc.splitlines()[-1])
        
         

    # Utility function to Read the current positions of all the servos, conver to Joint State, and publish the Joint State. 
    # Added: actually move the servos.
    def set_joint_state(self): 
        if self.arm_is_present:
            # Take the opportunity before sending commands to read current state. 
            self.read_bus_servos()
        else:
            self.servo_pos = self.cmd_all 
        # print(pos_all)
        # if not pos_all == None:
        # Pack the Joint State and publish it. 
        try:
            # convert the bus servo commands to joint state 
            jt_all = self.convert_commands_to_joint_state(self.servo_pos)
            self.joint_state_msg.position = jt_all
            self.joint_state_msg.name = ['base_joint', 'shoulder_joint', 'elbow_joint', 'forearm_joint', 'wrist_joint', 'fingers_joint', 'gripper']
            self.joint_state_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_joint_states.publish(self.joint_state_msg)
        except : 
            # traceback.print_exc()
            # self.get_logger().error('ERROR: Value out of range. Not Publishing Joint State.')
            exc = traceback.format_exc(limit=1)
            self.get_logger().error(exc.splitlines()[-1])
            
            
        # send the servo commands if (and only if) there's an arm attached. 
        if self.arm_is_present:
            self.command_bus_servos( self.bus_servo_indices_base_to_tip, self.cmd_all)
 
            
    # Utility function to convert Joint State into Commands (for the bus servos).
    def convert_joint_state_to_commands(self,jt_all):
        # unpack joint angle settings
        alpha0 = jt_all[0]
        beta1 = jt_all[1]
        beta2 = jt_all[2]
        gamma3 = jt_all[3]
        beta4 = jt_all[4]
        gamma5 = jt_all[5]    
        theta_gripper = jt_all[6]
        # Interpolate angles to servo microseconds to find servo commands 
        cmd_00 = self.f_interp_rad_to_cmd_01_params(alpha0)
        cmd_01 = self.f_interp_rad_to_cmd_12_params(beta1)
        cmd_02 = self.f_interp_rad_to_cmd_23_params(beta2)
        cmd_03 = self.f_interp_rad_to_cmd_34_params(gamma3)
        cmd_04 = self.f_interp_rad_to_cmd_45_params(beta4)
        cmd_05 = self.f_interp_rad_to_cmd_56_params(gamma5)
        cmd_06 = self.f_interp_rad_to_cmd_gripper_params(theta_gripper)
        # Convert to integers 
        cmd_all = list(map(int,[cmd_00,cmd_01,cmd_02,cmd_03,cmd_04,cmd_05,cmd_06]))
        return cmd_all
            
    # Utility function to convert Commands into Joint States. 
    def convert_commands_to_joint_state(self,cmd_all):
        # Interpolate to find joint_state 
        jt00 = self.f_interp_cmd_to_rad_01_params(cmd_all[0])
        jt01 = self.f_interp_cmd_to_rad_12_params(cmd_all[1])
        jt02 = self.f_interp_cmd_to_rad_23_params(cmd_all[2])
        jt03 = self.f_interp_cmd_to_rad_34_params(cmd_all[3])
        jt04 = self.f_interp_cmd_to_rad_45_params(cmd_all[4])
        jt05 = self.f_interp_cmd_to_rad_56_params(cmd_all[5])
        jt06 = self.f_interp_cmd_to_rad_gripper_params(cmd_all[6])
        jt_all = [jt00, jt01, jt02, jt03, jt04, jt05, jt06]
        return jt_all
        
    # Function to command any Hiwonder Bus Servo with a given command
    def command_bus_servo(self, servo_index, cmd):
        if self.arm_is_present:
            servo_ID = self.bus_servo_IDs_base_to_tip[servo_index]
            self.arm.setPosition([ [servo_ID, cmd] ], duration=self.movement_time_ms )

    # Function to command all Hiwonder Bus Servos with individual commands
    def command_bus_servos(self, servo_indices, cmds):
        if self.arm_is_present:
            # print(cmds)
            servo_IDs = [self.bus_servo_IDs_base_to_tip[x] for x in servo_indices]
            if self.initializing: 
                self.arm.setPosition( list(zip(servo_IDs, cmds)), duration=1000 )
                self.initializing = False
            else: 
                self.arm.setPosition( list(zip(servo_IDs, cmds)), duration=self.movement_time_ms )
        
    # Function to shut down all the servos by sending them the shutdown command.
    def shutdown_arm(self):
        if self.arm_is_present:
            self.arm.servoOff()
            print('Attempting to Power Down XArm.')

    # Function to read the values of all Hiwonder Bus Servos with individual commands
    def read_bus_servos(self):
        return
        # try:
        #     self.servo_pos = [self.arm.getPosition(s) for s in self.servos]
        #     # return servo_pos
        # except:
        #     # traceback.print_exc()
        #     self.get_logger().error('Bad servo read.')




def main(args=None):
    rclpy.init(args=args)
    command_xarm_instance = CommandXArm()
    executor = MultiThreadedExecutor()
    
    try: 
        rclpy.spin(command_xarm_instance, executor=executor)
        
        
    except: 
        traceback.print_exc()
        command_xarm_instance.shutdown_arm()
        


if __name__ == '__main__':
    main()
