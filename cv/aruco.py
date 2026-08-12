#! /usr/bin/python3
import numpy as np
import cv2
# import PIL
from cv2 import aruco
import matplotlib.pyplot as plt
# import matplotlib as mpl
import pickle
import rclpy
from rclpy.node import Node
import traceback
from example_interfaces.msg import String

#Setup camera capture and resolution
camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,1280);
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,960);

# What is the size of each marker - length of a side in meters (or any other unit you are working with). Used in call to "estimatePoseSingleMarkers". 
marker_side_length = 0.061 # meters 

#comment this out to remove live display (and some other stuff below)
plt.figure()
#Load calibration from pickle files (Python2 format..  you'll have to recalibrate if you use another camera)
cam_matrix = pickle.load(open("/home/pi/ros2_ws/src/mobrob/mobrob/cam_matrix.p","rb"),encoding='bytes')
dist_matrix = pickle.load(open("/home/pi/ros2_ws/src/mobrob/mobrob/dist_matrix.p","rb"),encoding='bytes')

#Tell opencv which aruco tags we're using (this should match the generation script)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()


class ARUCOTracker(Node):
    def __init__(self): 
        #setup ROS stuff
        super().__init__('aruco_tracker')
        self.pub_aruco = self.create_publisher(String, '/aruco', 1)
        self.msg_aruco = String()
               
    def stream_aruco(self):
        # Run an infinite loop to continually stream the data. 
        while True: 
            #Read a frame from the camera
            retval, frame = camera.read()
            #convert to grayscale
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            #try to find fiducials in the image
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)
            if (len(corners) > 0):
                toSend = ""
                #loop through all the detected markers
                for i in range(0,len(corners)):
                    #Estimate the 3D location of the marker
                    rot_vec, trans_vec, marker_points = aruco.estimatePoseSingleMarkers(corners[i],marker_side_length,cam_matrix,dist_matrix);
                    axis = np.float32([[4,0,0],[0,4,0],[0,0,-4]]).reshape(-1,3)
                    #Calculate coordinates in the image of the markers
                    imgpts, jac = cv2.projectPoints(axis,rot_vec,trans_vec,cam_matrix,dist_matrix);
                    #draw the axes for the markers (remove for no live display)
                    # frame = aruco.drawAxis(frame,cam_matrix,dist_matrix,rot_vec,trans_vec,.1)
                    cv2.drawFrameAxes(frame,cam_matrix,dist_matrix,rot_vec,trans_vec,.1) 
                    #rotMat = cv2.Rodrigues(rot_vec);
                    #print(rotMat)
                    #String to send through the topic
                    toSend = toSend +  "ID:%d,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f " % (ids[i],trans_vec[0][0][0], trans_vec[0][0][1], trans_vec[0][0][2],rot_vec[0][0][0],rot_vec[0][0][1],rot_vec[0][0][2])
                    #Only display if one tag is shown (opencv can't do \n...)
                    if(len(corners) == 1):
                        cv2.putText(frame,toSend,(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,200,75),4)
                #draw outlines and ids for all the markers detected. (remove for no live display)
                aruco.drawDetectedMarkers(frame,corners,ids)
                #Send ROS message
                self.msg_aruco.data = toSend 
                self.pub_aruco.publish(self.msg_aruco)
            #display the image with debug info (remove for no live display)
            cv2.imshow("live video - press 'q' to exit", frame)
            #Get keys pushed when the image is in focus (ctr-c works if there isn't live display)
            key = cv2.waitKey(1)
            #If any key is pushed, quit the program
            if (key != -1):
                cv2.destroyAllWindows()
                exit()



def main(args=None):
    rclpy.init(args=args)
    aruco_tracker_instance = ARUCOTracker()
    # No need to 
    # rclpy.spin(aruco_tracker_instance)
    aruco_tracker_instance.stream_aruco()



if __name__=="__main__":
    try: 
        main()
    except SystemExit: 
        pass
    except: 
        traceback.print_exc()
