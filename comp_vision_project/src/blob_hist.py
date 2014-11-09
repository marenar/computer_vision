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

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image)

    cv2.namedWindow("Image window", 1)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/image_raw",Image,self.callback)

  def make_hsv_range(self, hsvColor):

    low = [hsvColor[0] - 10, hsvColor[1] - 20, hsvColor[2] - 20] 
    high = [hsvColor[0] + 10, hsvColor[1] + 20, hsvColor[2] + 20]

    for i in low:
      if i < 0:
        i = 0

    for i in high:
      if i > 255:
        i = 255

    return [low, high]

  def callback(self,data):

    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError, e:
      print e

    cv_image = cv2.blur(cv_image,(3,3))
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    #image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster the pixel intensities
    clt = KMeans(n_clusters = 3)
    clt.fit(image)

    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = self.centroid_histogram(clt)
    #print hist
    #print clt.cluster_centers_
    #bar = self.plot_colors(hist, clt.cluster_centers_)

    # # show our color bart
    # plt.figure()
    # plt.axis("off")
    # plt.imshow(bar)
    # plt.show()
    low = []
    high = []

    for (percent, color) in zip(hist, clt.cluster_centers_):
      if percent < 10:
        #print "color list"
        #print color.astype("uint8").tolist()
        #print type(color)

        integerColor = color.astype("uint8")
        print integerColor
        print type(integerColor[0])

        hsvColor = cv2.cvtColor(integerColor, cv2.COLOR_RGB2HSV)
        [low, high] = self.make_hsv_range(hsvColor)

    # #opencv HSV range = [180, 255,255]
    # #target color = 30, 255, 255
    # #Convert to hsv and find range of colors
    #hsv = cv2.cvtColor(cv_image,cv2.COLOR_BGR2HSV)
    #thresh = cv2.inRange(hsv,low, high)
        
    # #Find contours in the threshold image
    # contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    # #Finding contour with maximum area and store it as best_cnt
    # index = 0
    # max_area = 0
    # for i in range(len(contours)):
    #    area = cv2.contourArea(contours[i])
    #    if area > max_area:
    #        max_area = area
    #        best_cnt = contours[i]
    #        index = i

    # try:
    #    best_cnt
    # except NameError: 
    #    print "no blobs"
    # else:
    #    #Finding centroids of best_cnt and draw a circle there
    #    M = cv2.moments(best_cnt)
    #    cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    #    cv2.circle(cv_image,(cx,cy),10,255,-1)

    #    #draw the most likely contour
    #    cv2.drawContours(cv_image,  contours, index, (0,255,0), 3)

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)


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

  def plot_colors(self, hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
      print percent, color
      # plot the relative percentage of each cluster
      endX = startX + (percent * 300)
      cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
        color.astype("uint8").tolist(), -1)
      startX = endX
    # return the bar chart
    return bar


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
