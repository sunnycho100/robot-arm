#!/usr/bin/env python3

# ROS node to command an Endpoint Path to a HiWonder xArm 1S 
# Peter Adamczyk, University of Wisconsin - Madison
# Updated 2024-11-12

import rclpy
from rclpy.node import Node 
# from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup 
# from rclpy.executors import MultiThreadedExecutor
# import threading
import numpy as np
import traceback
import time
# from sensor_msgs.msg import JointState
# from xarmrob_interfaces.srv import ME439XArmInverseKinematics #, ME439XArmForwardKinematics
from xarmrob_interfaces.msg import ME439PointXYZ

import xarmrob.smooth_interpolation as smoo 


## Define a temporary function using Python "lambda" functionality to print colored text
# see https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal/3332860#3332860
# search that page for "CircuitSacul" to find that answer
coloredtext = lambda r, g, b, text: f'\033[38;2;{r};{g};{b}m{text}\033[38;2;255;255;255m'



class EndpointAutomatic(Node): 
    def __init__(self): 
        super().__init__('endpoint_automatic')
                
        
        self.xyz_goal = [0.15, 0.0, 0.10] # roughly upright neutral with wrist at 45 degrees. Formally: [0.1646718829870224, 0.0, 0.1546700894832611]
        self.old_xyz_goal = [0.15, 0.0, 0.10]
        self.xyz_traj = [self.old_xyz_goal]
        self.disp_traj = self.xyz_traj 
        self.gripper = 0
        self.idx = 0

        # =============================================================================
        #   # Publisher for the Endpoint goal. 
        # =============================================================================
        self.pub_endpoint_desired = self.create_publisher(ME439PointXYZ,'/endpoint_desired',1)#,callback_group=ReentrantCallbackGroup())
        # Create the message, with a nominal pose
        self.endpoint_desired_msg = ME439PointXYZ()
        self.endpoint_desired_msg.xyz = self.xyz_goal 

        # command frequency parameter: how often we expect it to be updated    
        self.command_frequency = self.declare_parameter('command_frequency',5).value
        self.movement_time_ms = round(1000/self.command_frequency)  # milliseconds for movement time. 
        self.endpoint_speed = self.declare_parameter('endpoint_speed',0.05).value  # nominal speed for continuous movement among points. 
        
        # SVG File Name
        self.filename = self.declare_parameter('path_svg_file','/home/pi/ros2_ws/src/xarmrob/xarmrob/RobotArmPath.svg').value
        #Get Height offset for safety (do practice runs in the air)
        self.vertical_offset = self.declare_parameter('vertical_offset',0.02).value
        
        self.set_endpoint_trajectory()
        
        time.sleep(5)
        
        # Set up a timer to send the commands at the specified rate. 
        self.timer = self.create_timer(self.movement_time_ms/1000, self.send_endpoint_desired)#,callback_group=ReentrantCallbackGroup())

    # Callback to publish the endpoint at the specified rate. 
    def send_endpoint_desired(self):
        print(self.idx)
        if self.idx>=len(self.disp_traj):
            self.idx = len(self.disp_traj) - 1
        self.xyz_goal = self.disp_traj[self.idx]
        self.idx += 1
        self.endpoint_desired_msg.xyz = self.xyz_goal 
        self.pub_endpoint_desired.publish(self.endpoint_desired_msg)

        
        
    def set_endpoint_trajectory(self):

        # Use an SVG file to specify the path: 
        import xarmrob.parse_svg_for_robot_arm_v03 as psvg
        endpoints = psvg.convert_svg_to_endpoints(self.filename, xlength=0.10, ylength=0.10, rmin=0.10, rmax=0.20)
        # Optionally insert a y-axis rotation to rotate the plane of the points up from the ground (e.g. suggest: -np.pi/3)
        theta = 0 # -np.pi/3.
        R = np.array( [[np.cos(theta), 0., np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
        endpoints = endpoints@R.transpose()
        endpoints = np.vstack( (np.array(self.old_xyz_goal), endpoints, np.array(self.old_xyz_goal)))
        endpoints[:,2] = endpoints[:,2]+self.vertical_offset
        # print(endpoints)
        
        # Build a long array of many endpoint locations using smooth constant-speed control: 
        self.disp_traj = np.ndarray( (0,3) )
        for ii in range(1,len(endpoints)):
            # Do constant velocity or minimum jerk interpolation
            # either smoo.constant_velocity_interpolation() or smoo.minimum_jerk_interpolation()
            t,seg_traj = smoo.constant_velocity_interpolation(np.array(endpoints[ii-1]), np.array(endpoints[ii]), self.endpoint_speed, self.command_frequency)
            self.disp_traj = np.vstack( (self.disp_traj,seg_traj[1:]) )
        self.idx = 0
        print(self.disp_traj)

            
        # # Run a loop that gets input from the user
        # while True:
            
        #     try: 
        #         in_string = input('Enter Endpoint Location as a list (in meters): x, y, z: \n    Ctrl-C and Enter to exit.\n')   
        #         in_floats = list(map(float, in_string.strip('[]').split(',')))
        #         assert len(in_floats) == 3
        #     except: 
        #         retry = input('Bad Input. Try again? [y]/n: ')
        #         if (retry=='y') or (retry=='Y') or (retry==''):
        #             continue
        #         else: 
        #             return
                    
        #     prnttmpl = coloredtext(50,255,50,'\n\tEndpoint Goal Input was [' + '{:.3f}, '*2 + '{:.3f}]')
        #     self.get_logger().info(prnttmpl.format(*in_floats))
        #     self.new_xyz_goal = in_floats
            
        #     # Do constant velocity or minimum jerk interpolation
        #     # either smoo.constant_velocity_interpolation() or smoo.minimum_jerk_interpolation()
	    #     self.t,self.disp_traj = smoo.minimum_jerk_interpolation(np.array(self.old_xyz_goal), np.array(self.new_xyz_goal), self.endpoint_speed, self.command_frequency)
            
        # # Reset counter and wait until the trajectory has been played
        # self.idx = 0
        # while(self.idx<len(self.disp_traj)):
        #     # print(self.idx)
        #     rclpy.spin_once(self)
        #     # time.sleep(0.08)
                
        #     self.old_xyz_goal = self.new_xyz_goal
            



def main(args=None):
    try: 
        rclpy.init(args=args)
        endpoint_automatic_instance = EndpointAutomatic()  
        rclpy.spin(endpoint_automatic_instance)
        
    except: 
        traceback.print_exc(limit=1)
        
# # Old
# def main(args=None):
#     try: 
#         rclpy.init(args=args)
#         endpoint_automatic_instance = EndpointAutomatic()  
#         thrd = threading.Thread(target=endpoint_automatic_instance.set_endpoint_trajectory())
#         thrd.start()
#         # No need to "rclpy.spin(endpoint_keyboard_instance)" here because there's a while() loop blocking and keeping it alive. 
#         executor=MultiThreadedExecutor()
#         rclpy.spin(endpoint_automatic_instance,executor=executor)
#         # rclpy.spin(endpoint_keyboard_instance)
        
#     except: 
#         traceback.print_exc(limit=1)
        


if __name__ == '__main__':
    main()






############################################################################



        
# # global variable to hold the waypoint currently being tracked
# waypoint = ME439PointXYZ()
# waypoint.xyz = (0.2015, 0.0, 0.2056)     # Set to 0 initially so it does not think the job is finished. 

# # global variable to hold the previous position the robot was in
# # assumes it was just the previous commanded target point. 
# position_current = ME439WaypointXYZ()
# position_current.xyz = (0.2015, 0.0, 0.2057)    # Set to 0 initially so it does not think the job is finished. 

# isWaypointReady = False;

# # Global to track the state of completion of the  waypoint    
# waypoint_complete = Bool()
# waypoint_complete.data = False    

# # =============================================================================
# #     Set up a waypoint seeker
# #     and to "waypoint_xyz"  (ME439WaypointXYZ)
# #     Publish "target_xyz" (ME439WaypointXYZ) (the "smoothed" instantaneous goal)
# # =============================================================================

# # Get parameters from rosparam
# endpoint_speed = rospy.get_param('/endpoint_speed') # how fast should we move?  
# command_frequency = rospy.get_param('/command_frequency') # how many times per second should we send a new command to the Arm? 
# drmax = endpoint_speed / command_frequency

# # Create the publisher. Name the topic "target_xyz", with message type "ME439WaypointXYZ"
# pub_target_xyz = rospy.Publisher('/target_xyz', ME439WaypointXYZ, queue_size=1)

# # Create the publisher for the topic "waypoint_complete", with message type "Bool"
# pub_waypoint_complete = rospy.Publisher('/waypoint_complete', Bool, queue_size=1)


# # Publish desired targets at the appropriate time. 
# def talker(): 
#     # Actually launch a node called "waypoint_seeker"
#     rospy.init_node('waypoint_seeker', anonymous=False)
    
#     # Subscriber to the "waypoint_xyz" topic
#     sub_waypoint = rospy.Subscriber('/waypoint_xyz', ME439WaypointXYZ, set_waypoint)
      
#     # set up a rate basis to keep it on schedule.
#     r = rospy.Rate(command_frequency)
    
#     # Send the arm to an initial target prior to receiving its first waypoint: 
#     initial_target = ME439WaypointXYZ() 
#     initial_target.xyz = position_current.xyz
#     pub_target_xyz.publish(initial_target)
    
#     # start the periodic calls to the target publisher. 
#     while not rospy.is_shutdown():
#         set_next_target_toward_waypoint()
        
#         r.sleep()
    


# # =============================================================================
# # # Function to update the path to the waypoint based on the robot's current position
# # =============================================================================
# def set_next_target_toward_waypoint():
#     # First assign the incoming message
#     global position_current, waypoint, waypoint_complete, isWaypointReady
#     if(isWaypointReady):
#         # set the path to be directly from here to the waypoint
#         dr = np.array( waypoint.xyz ) - np.array( position_current.xyz  )   # XYZ vector from current location to Waypoint
        
#         drlength = np.sqrt(dr.dot(dr))  # Compute the Cartesian distance to the Waypoint. 
#         drhat = dr/drlength
      
#         # Compute the ME439WaypointXYZ message        
#         target = ME439WaypointXYZ()        
#         if (drlength < drmax) :     # If close enough to reach the waypoint, go there and set it as complete
#             target = waypoint
#             if not waypoint_complete.data: # Only publish the "waypoint_complete" topic when first detected. 
#                 waypoint_complete.data = True
#                 pub_waypoint_complete.publish(waypoint_complete)
#         # Otherwise, go one step toward the waypoint
#         else:   
#             drtarget = drmax*drhat
#             target.xyz = position_current.xyz + drtarget
            
#         # Update "previous location"
#         position_current = target
        
#         #  Publish it
#         pub_target_xyz.publish(target)
        
#     # Else waypoint is not ready (hasn't started up)
#     else:
#         pass  # Do Nothing



# # Function to receive a Waypoint and set the goal point to it.     
# def set_waypoint(waypoint_msg_in): 
#     global waypoint, waypoint_complete, isWaypointReady
#     waypoint = waypoint_msg_in
#     waypoint_complete.data = False
#     isWaypointReady = True


        
    

# if __name__ == '__main__':
#     try: 
#         talker()
#     except rospy.ROSInterruptException: 
#         pass
