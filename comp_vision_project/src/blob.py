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

#listens to /camera/image_raw topic and converts it to an opencv image

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image)

    cv2.namedWindow("Image window", 1)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/image_raw",Image,self.callback)

  def callback(self,data):

    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError, e:
      print e

    #Smooth it
    cv_image = cv2.blur(cv_image,(3,3))

    #opencv HSV range = [180, 255,255]
    #target color = 30, 255, 255
    #Convert to hsv and find range of colors
    hsv = cv2.cvtColor(cv_image,cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv,np.array((25, 200, 200)), np.array((35, 255, 255)))
        
    #Find contours in the threshold image
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    #Finding contour with maximum area and store it as best_cnt
    index = 0
    max_area = 0
    for i in range(len(contours)):
       area = cv2.contourArea(contours[i])
       if area > max_area:
           max_area = area
           best_cnt = contours[i]
           index = i

    try:
       best_cnt
    except NameError: 
       print "no blobs"
    else:
       #Finding centroids of best_cnt and draw a circle there
       M = cv2.moments(best_cnt)
       cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
       cv2.circle(cv_image,(cx,cy),10,255,-1)

       #draw the most likely contour
       cv2.drawContours(cv_image,  contours, index, (0,255,0), 3)

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
