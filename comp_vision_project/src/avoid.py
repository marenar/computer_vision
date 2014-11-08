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
		else:
			self.max_width = -1.0
		self.distance_to_wall = -1.0
		self.angle_of_wall = -1.0
		self.avoid = False
		self.straight_ahead = -1.0
		self.direct = 1

	def scan_received(self, msg):
		""" Callback function for msg of type sensor_msgs/LaserScan """
		valid_measurements = []
		forward_measurements = []
		for i in range(360):
			# reinitialize self.avoid as False each time. This is the default value and will create wall following behavior.
			self.avoid = False
			if msg.ranges[i] != 0 and msg.ranges[i] < 7:
				valid_measurements.append(msg.ranges[i])
				if i < 5 or i > 355:
					forward_measurements.append(msg.ranges[i])
				"""if len(forward_measurements):
					self.straight_ahead = sum(forward_measurements) / len(forward_measurements)
					# This is the finite state controller that triggers the obstacle avoidance behavior. If the average distance straight ahead is less than 1.3,
					# the self.avoid attribute is set to true. self.direct is set such that the robot always turns toward the direction with more free space. 
					if self.straight_ahead < 1.3:
						self.avoid = True
						if msg.ranges[90] > msg.ranges[270] or msg.ranges[90] == 0:
							self.direct = 1
						else:
							self.direct = -1 """
			if self.max_width <= 150:
				self.avoid = True
				if msg.ranges[90] > msg.ranges[270] or msg.ranges[90] == 0:
					self.direct = 1 
				else:
					self.direct = -1
			if len(valid_measurements) and self.avoid == False:
				# This part of the finite state controller handles the wall following by finding the closest object and its angle relative to the robot. 
				self.distance_to_wall = min(valid_measurements)
				self.angle_of_wall = msg.ranges.index(self.distance_to_wall)
			else:
				self.distance_to_wall = -1.0
				self.angle_of_wall = -1.0
				self.straight_ahead = -1.0
				self.direct = 1

	def main(self): 
		""" Run loop for the wall follow node """
		pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
		sub = rospy.Subscriber('/scan', LaserScan, self.scan_received)
		rospy.init_node('wall_follow', anonymous=True)
		r = rospy.Rate(12) #12 Hz
		while not rospy.is_shutdown():
			if self.avoid == True: 
				# Object avoidance. Turn away from the object.
				msg = Twist(linear=Vector3(x=0.05),angular=Vector3(z=self.direct * 0.4))
			elif self.avoid == False and 0 < self.angle_of_wall < 180:
				# Wall following. Go forward and course correct so the wall stays 90 degrees to the robot.
				msg = Twist(linear=Vector3(x=0.2), angular=Vector3(z=(self.angle_of_wall - 90) * 0.01))
			elif self.avoid == False and 180 < self.angle_of_wall < 360: 
				# Wall following. Go forward and course correct so the wall stays 270 degrees to the robot. 
				msg = Twist(linear=Vector3(x=0.2), angular=Vector3(z=(self.angle_of_wall - 270) * 0.01))
			else:
				# There are no obstacles and no walls. Go straight until something appears. 
				msg = Twist(linear=Vector3(x=0.2))
			pub.publish(msg)
			r.sleep()
