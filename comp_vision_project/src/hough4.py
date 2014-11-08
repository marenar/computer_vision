#!/usr/bin/env python
import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import avoid

#listens to /camera/image_raw topic and converts it to an opencv image

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image)
    
    cv2.namedWindow("Image window", 1)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/image_raw",Image,self.callback)
    self.image_height = 480
    self.image_width = 640
    self.max_leg_width = 100
    self.gradient_width = 10
    
  #function calculates mean gradient of surrounding pixels on x axis 
  def calc_gradient(self, line_upper_quartiles, gray):

    sink_sources = list()    

    for line in line_upper_quartiles:
       left_sum = 0
       right_sum = 0

       for n in range(self.gradient_width):
          posL = line - n
          posR = line + n
          if (posL < 0):
             posL = 0
          elif (posR > self.image_width - 1):
             posR = self.image_width - 1

          left_sum = left_sum +  gray[self.image_height/4, posL]
          right_sum = right_sum + gray[self.image_height/4, posR]

       if left_sum > right_sum:
          sink_sources.append(1)
       else:
          sink_sources.append(-1)

    return sink_sources

  #this function groups matching sinks and sources and filters noise
  def group_sink_sources(self, line_centers, sink_sources):

    leg_width = list()
    leg_center = list()
    left_bound = self.image_width + 1
    right_bound = -1 
 
    for n in range(len(sink_sources)):

       if sink_sources[n] == 1:	#is a source, left side of leg
         #set a new left_bound, only once in the beginning
         if left_bound > self.image_width:
	     left_bound = line_centers[n]
         #hit a left bound while having a right bound, end of old leg
         elif right_bound > 0: 
             leg_center.append((left_bound + right_bound)/2)
             leg_width.append(right_bound-left_bound)
             #reset variables for new leg
             left_bound = line_centers[n]
	     right_bound = -1
       else:
         #set right_bound only if a left_bound exists, keep updating
         if left_bound < self.image_width: 
             right_bound = line_centers[n]

    #cleanup remaining leg not terminated by another left bound
    if right_bound > 0:
       leg_center.append((left_bound + right_bound) / 2)
       leg_width.append(right_bound-left_bound)
    return [leg_center, leg_width]

  #function returns angle's compliment
  def compliment (self,rad_angle):

    if rad_angle < np.pi/2:
       return np.pi/2 - rad_angle
    elif rad_angle < np.pi:
       return np.pi - rad_angle
    elif rad_angle < np.pi * 3/2:
       return np.pi *3/2 - rad-angle
    else: 
       return np.pi * 2 - rad_angle

  def callback(self,data):

    line_centers = list()
    keypoints = list()

    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError, e:
      print e

    gray = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/180,100)
    for rho,theta in lines[0]:
     
       #variables used for displaying lines
       a = np.cos(theta)
       b = np.sin(theta)
       x0 = a*rho
       y0 = b*rho
       x1 = int(x0 + 1000*(-b))
       y1 = int(y0 + 1000*(a))
       x2 = int(x0 - 1000*(-b))
       y2 = int(y0 - 1000*(a))

       #filter for vertical lines only, centered around degrees 0 and 180
       deg_thresh = 10
       theta_deg = theta * 180 / np.pi
       if ((theta_deg < deg_thresh) and (theta_deg > (360-deg_thresh))) \
		or ((theta_deg > (180-deg_thresh)) and (theta_deg < (180+deg_thresh))):

          #display line
          cv2.line(cv_image,(x1,y1),(x2,y2),(0,0,255),2)

          #calculate x coordinate of line in center of image
          comp = self.compliment(theta)
          x_center = int(np.tan(comp)*self.image_height/2 + x0)
          line_centers.append(x_center)
          
          #calculate x coordinate of line in upper quartile of image
          key = [int(np.tan(comp)*self.image_height/4 + x0),0]
          keypoints.append(key)
    

    #calculate and display sink_sources
    sort_line_centers = sorted(line_centers)
    print keypoints
    sort_keypoints = sorted(keypoints)
    print sort_keypoints

    sink_sources = self.calc_gradient(sort_keypoints, gray)
    #for n in range(len(sink_sources)):
    #    if sink_sources[n] == 1:
    #       cv2.circle(cv_image, (sort_line_centers[n],self.image_height/2), 10, (255,0,0), -1)
    #    else:
    #       cv2.circle(cv_image, (sort_line_centers[n],self.image_height/2), 10, (0,255,0), -1)

    #55 pixels is 30 inches away
    ratio = float(30.0 / 55)
    #print ratio

    #calculate and display leg centers
    [leg_centers, leg_width] = self.group_sink_sources(sort_line_centers, sink_sources)
    print leg_width
    test = avoid.Obstacle_Avoidance(leg_width)
    test.main()
    for n in range(len(leg_centers)):
      # print leg_width[n]
       print "inches away: ", (ratio * leg_width[n])
       cv2.circle(cv_image, (leg_centers[n], self.image_height/2), 10, (255,255,255), -1) 

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError, e:
      print e

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print "Shutting down"
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
