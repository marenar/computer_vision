#!/usr/bin/env python
# Code found at http://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/

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

		cv_image = cv2.blur(cv_image,(3,3))
		image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

		# show our image
		plt.figure()
		plt.axis("off")
		plt.imshow(image)

		# reshape the image to be a list of pixels
		image = image.reshape((image.shape[0] * image.shape[1], 3))

		# cluster the pixel intensities
		clt = KMeans(n_clusters = 3)
		clt.fit(image)

		# build a histogram of clusters and then create a figure
		# representing the number of pixels labeled to each color
		hist = centroid_histogram(clt)
		print hist
		print clt.cluster_centers_

		bar = plot_colors(hist, clt.cluster_centers_)

		# show our color bart
		plt.figure()
		plt.axis("off")
		plt.imshow(bar)
		plt.show()

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

def plot_colors(hist, centroids):
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
