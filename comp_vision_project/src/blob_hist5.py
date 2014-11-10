#!/usr/bin/env python

import roslib
#roslib.load_manifest('my_package')
import rospy
import cv2
import datetime
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Vector3
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_processed",Image)
    self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
    self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.laser_callback)

    cv2.namedWindow("Image window", 1)
    self.bridge = CvBridge()
    self.image_width = 640
    self.image_height = 480

    self.avoid = False
    self.blob = False 
    self.distance_to_wall = -1.0
    self.angle_of_wall = -1.0
    self.straight_ahead = -1.0
    self.direct = 1
    self.update = datetime.datetime.now()
    self.counter = 0
    self.target = []

  def make_color_range(self, rgbColor):
    rangeval = 15

    low = [rgbColor[0] - rangeval, rgbColor[1] - rangeval, rgbColor[2] - rangeval] 
    high = [rgbColor[0] + rangeval, rgbColor[1] + rangeval, rgbColor[2] + rangeval]

    for i in low:
      if i < 0:
        i = 0

    for i in high:
      if i > 255:
        i = 255

    return [low, high]

  def image_callback(self, data):
    # only run the k means clustering every 5 times we get image data, because image data is frequent and k means is computationally expensive
    if self.counter == 6:
      self.counter = 0
    else:
      self.counter += 1

    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError, e:
      print e

    cv_image = cv2.blur(cv_image,(3,3))
    image1 = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    if self.counter == 0:
      # shrink image for faster processing
      smaller = cv2.resize(image1,None,fx=.1, fy=.1, interpolation = cv2.INTER_CUBIC)
      # reshape the image to be a list of pixels
      image = smaller.reshape((smaller.shape[0] * smaller.shape[1], 3))

      # cluster the pixel intensities using k means clustering
      clt = KMeans(n_clusters = 3)
      clt.fit(image)

      # build a histogram of clusters
      hist = self.centroid_histogram(clt)

      percents = list(hist)
      # If the color in the histogram takes up less than 10% of the image, it meets our criteria for a small obstacle
      if min(percents) < 0.1:
        self.target = clt.cluster_centers_[percents.index(min(percents))].astype("uint8")
      else:
        self.target = []
        self.blob = False

    if len(self.target):
      [low, high] = self.make_color_range(self.target)

      thresh = cv2.inRange(image1, np.array(low), np.array(high))
          
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
         self.blob = False
      else:
         #Finding centroids of best_cnt and draw a circle there
         M = cv2.moments(best_cnt)
         cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
         # setting self.blob triggers the obstacle avoidance part of the finite state controller
         self.blob = cx - 320
         # if the direction has not been set within the last 3 seconds (avoids noise), turn away from the camera obstacle
         # (turn right if the obstacle appears in left of frame, turn left if obstacle appears in right of frame).
         if datetime.datetime.now() - self.update > datetime.timedelta(seconds = 3):
          print "blob time", datetime.datetime.now() - self.update
          if self.blob < 0:
            self.direct = -1
          else:
            self.direct = 1 
          self.update = datetime.datetime.now()

         cv2.circle(cv_image,(cx,cy),10,255,-1)

         #draw the most likely contour
         cv2.drawContours(cv_image,  contours, index, (0,255,0), 3)
         self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

      cv2.imshow("Image window", cv_image)
      cv2.waitKey(3)

  def laser_callback(self, msg):
    forward_measurements = []
    valid_measurements = []
    ninety = []
    twoseventy = []
    self.avoid = False
    for i in range(360):
      try:
        if msg.ranges[i] != 0 and msg.ranges[i] < 7:
          valid_measurements.append(msg.ranges[i])
        if i < 5 or i > 355:
          forward_measurements.append(msg.ranges[i])
        elif 85 < i < 95:
          ninety.append(msg.ranges[i])
        elif 265 < i < 275:
          twoseventy.append(msg.ranges[i])
      except IndexError: pass
    if len(forward_measurements):
      self.straight_ahead = sum(forward_measurements) / len(forward_measurements)
      # if there is an obstacle within 1 meter straight ahead, set self.avoid to True, triggering the obstacle avoidance part of the finite state controller.
      if self.straight_ahead != 0 and self.straight_ahead < 1:
        self.avoid = True
        try:
          # if the direction has not been set within the last 3 seconds (avoids noise), turn towards the direction (90 or 270 degrees) with more space
          if datetime.datetime.now() - self.update > datetime.timedelta(seconds = 3): 
            print "wall time", datetime.datetime.now() - self.update
            vals = [sum(ninety) / len(ninety), sum(twoseventy) / len(twoseventy)]
            if vals[0] > vals[1] or vals[0] == 0:
              self.direct = 1
            else:
              self.direct = -1
            self.update = datetime.datetime.now()
        except IndexError: pass
    if len(valid_measurements) and self.avoid == False:
      # wall following variables set, triggering wall following part of the finite state controller.
      self.distance_to_wall = min(valid_measurements)
      self.angle_of_wall = msg.ranges.index(self.distance_to_wall)
    else:
      self.distance_to_wall = -1.0
      self.angle_of_wall = -1.0
      self.straight_ahead = -1.0
      self.direct = 1

  def centroid_histogram(self, clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    
    return hist

  def main(self):
    rospy.init_node('obstacle_avoidance', anonymous=True)
    r = rospy.Rate(2)
    while not rospy.is_shutdown():
      if self.avoid != False or self.blob != False or datetime.datetime.now() - self.update < datetime.timedelta(seconds = 3):
        # An obstacle was found by the lidar or the camera in the last 3 seconds, so turn in the direction the lidar or camera specified
        msg = Twist(linear=Vector3(x=0.02),angular=Vector3(z=self.direct * 0.3))
      elif self.avoid == False and 0 < self.angle_of_wall < 180:
        # Wall following behavior
        msg = Twist(linear=Vector3(x=0.1), angular=Vector3(z=(self.angle_of_wall - 90) * 0.01))
      elif self.avoid == False and 180 < self.angle_of_wall < 360: 
        # Wall following behavior
        msg = Twist(linear=Vector3(x=0.1), angular=Vector3(z=(self.angle_of_wall - 270) * 0.01))
      else:
        # Go straight
        msg = Twist(linear=Vector3(x=0.1))
      self.vel_pub.publish(msg)
      r.sleep()
        
if __name__ == '__main__':
  try:
    ic = image_converter()
    ic.main()
  except rospy.ROSInterruptException:
    print "Shutting down"
    cv2.destroyAllWindows()
