#!/usr/bin/env python
# Marena Richardson, 10/8/14

import rospy 
import math
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import LaserScan

class Obstacle_Avoidance:
	def __init__(self, leg_width):
		if len(leg_width):
			self.max_width = max(leg_width)

	def main(self):
		if self.max_width:
			print self.max_width 