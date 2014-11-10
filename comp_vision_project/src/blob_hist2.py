#!/usr/bin/env python
import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Vector3
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#listens to /camera/image_raw topic and converts it to an opencv image

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image)
    self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
    self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.laser_callback)

    cv2.namedWindow("Image window", 1)
    self.bridge = CvBridge()

    self.image_width = 640
    self.image_height = 480
    self.avoid = False
    self.blob = False
    self.counter = 0
    self.target = []
    self.blob_direct = 1
    self.avoid_direct = 1

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
    if self.counter == 11:
      self.counter = 0
    else:
      self.counter += 1

    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError, e:
      print e

    cv_image = cv2.blur(cv_image,(3,3))
    #print cv_image.shape
    #cv_image = cv_image[self.image_height/2: self.image_height, 0: self.image_width]
    #print cv_image.shape
    image1 = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    if self.counter == 0:
      smaller = cv2.resize(image1,None,fx=.1, fy=.1, interpolation = cv2.INTER_CUBIC)
      # reshape the image to be a list of pixels
      image = smaller.reshape((smaller.shape[0] * smaller.shape[1], 3))

      # cluster the pixel intensities
      clt = KMeans(n_clusters = 3)
      clt.fit(image)

      # build a histogram of clusters
      hist = self.centroid_histogram(clt)

      percents = list(hist)
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
         self.blob = cx - 320
         if self.blob < 0: 
          self.blob_direct = -1 
         else:
          self.blob_direct = 1
         cv2.circle(cv_image,(cx,cy),10,255,-1)

         #draw the most likely contour
         cv2.drawContours(cv_image,  contours, index, (0,255,0), 3)

      cv2.imshow("Image window", cv_image)
      cv2.waitKey(3)

  def laser_callback(self, msg):
    forward_measurements = []
    self.avoid = False
    for i in range(360):
      if i < 15 or i > 345:
        try:
          if msg.ranges[i] != 0 and msg.ranges[i] < 7:
            forward_measurements.append(msg.ranges[i])
        except IndexError: pass
    if len(forward_measurements):
      self.straight_ahead = sum(forward_measurements) / len(forward_measurements)
      if self.straight_ahead < 1:
        self.avoid = True
    try:
      if self.avoid == True:
        print msg.ranges[90], msg.ranges[270]
      if msg.ranges[90] + 0.5 > msg.ranges[270] or msg.ranges[90] == 0:
        self.avoid_direct = 1
      else:
        self.avoid_direct = -1
    except IndexError: pass

  def centroid_histogram(self, clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist

  def main(self):
    rospy.init_node('obstacle_avoidance', anonymous=True)
    r = rospy.Rate(2)
    while not rospy.is_shutdown():
      if self.avoid == True and self.blob != False:
        print "both", self.avoid_direct, self.blob_direct
        if self.avoid_direct != self.blob_direct:
          msg = Twist(linear=Vector3(x=0.05), angular=Vector3(z=0.4))
        else:
          msg = Twist(linear=Vector3(x=0.05), angular=Vector3(z=self.avoid_direct * 0.4))
      if self.avoid == True:
        print "avoid", self.avoid, self.blob, self.avoid_direct
        msg = Twist(linear=Vector3(x=0.05),angular=Vector3(z=self.avoid_direct * 0.4))
      elif self.blob != False:
        print "blob", self.avoid, self.blob, self.blob_direct
        msg = Twist(linear=Vector3(x=0.05),angular=Vector3(z=self.blob_direct * 0.2))
      else:
        print "line", self.avoid, self.blob
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
