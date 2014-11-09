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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#listens to /camera/image_raw topic and converts it to an opencv image

def callback(image_data, laser_scan):
  bridge = CvBridge()
  try:
    cv_image = bridge.imgmsg_to_cv2(image_data, "bgr8")
  except CvBridgeError, e:
    print e

  cv_image = cv2.blur(cv_image,(3,3))
  image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
  image2 = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

  # reshape the image to be a list of pixels
  image = image.reshape((image.shape[0] * image.shape[1], 3))

  # cluster the pixel intensities
  clt = KMeans(n_clusters = 3)
  clt.fit(image)

  # build a histogram of clusters
  hist = centroid_histogram(clt)

  percents = list(hist)
  target = clt.cluster_centers_[percents.index(min(percents))].astype("uint8")

  [low, high] = make_color_range(target)

  thresh = cv2.inRange(image2, np.array(low), np.array(high))
      
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

def make_color_range(rgbColor):
  rangeval = 5

  low = [rgbColor[0] - rangeval, rgbColor[1] - rangeval, rgbColor[2] - rangeval] 
  high = [rgbColor[0] + rangeval, rgbColor[1] + rangeval, rgbColor[2] + rangeval]

  for i in low:
    if i < 0:
      i = 0

  for i in high:
    if i > 255:
      i = 255

  return [low, high]

def centroid_histogram(clt):
  # grab the number of different clusters and create a histogram
  # based on the number of pixels assigned to each cluster
  numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
  (hist, _) = np.histogram(clt.labels_, bins = numLabels)

  # normalize the histogram, such that it sums to one
  hist = hist.astype("float")
  hist /= hist.sum()

  # return the histogram
  return hist


def main(args):
  ic = image_converter()
  image_pub = rospy.Publisher("image_topic_2",Image)
  vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
  image_sub = rospy.Subscriber("/camera/image_raw", Image)
  laser_sub = rospy.Subscriber("/scan", LaserScan)
  ts = message_filters.TimeSynchronizer([image_sub, laser_sub], 10)
  ts.registerCallback(callback)
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print "Shutting down"
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
