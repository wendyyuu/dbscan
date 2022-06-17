#!/usr/bin/env python3

import rosbag
import ros_numpy
import sensor_msgs
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

class dataParser:
	# constructor
	def __init__(self, bag) :
		self.bag = bag
		self.boschData = self.sensorData()
		pass

	def convert_pc2_to_np(self, pc_msg):   
		# is a data descriptor. that is required by the ros_numpy function used in the line below it
		pc_msg.__class__ = sensor_msgs.msg._PointCloud2.PointCloud2 # ASK
		# ros_numpy.numpify needs the pc_msgs to have the exact class sensor_msgs.msg._PointCloud2.PointCloud2 to be able to convert the point cloud into a numpy array

		# ros_numpy.numpify needs the exact class definition of pc_msg to be PointCloud2
		numped = ros_numpy.numpify(pc_msg)
		pc_int = []
		for t in numped :
			pc_int.append(t[3])    
		offset_sorted = {f.offset: f for f in pc_msg.fields}
		pc_msg.fields = [f for (_, f) in sorted(offset_sorted.items())] 
		pc_np = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc_msg, remove_nans=True)
		return pc_np, np.asarray(pc_int)

	def rosbagParser(self) :
		# topic: the topic of the message
		# msg: the message
		# t: time of message. The time is represented as a rospy Time object (t.secs, t.nsecs)
		for topic, msg, t in rosbag.Bag(self.bag).read_messages():
			if topic == "/s1/radar_pointCloud":
				bosch_np, intensity = self.convert_pc2_to_np(msg)
				bosch_np = bosch_np.transpose()
				bosch_np = bosch_np.tolist()
				self.boschData.x.append(bosch_np[0])
				self.boschData.y.append(bosch_np[1])
				self.boschData.z.append(bosch_np[2])
				self.boschData.intensity.append(intensity)
				self.boschData.time.append(np.ones(len(intensity))*int(str(t)))
				break

		return self.boschData

	class sensorData:
		def __init__(self):
			self.x = []
			self.y = []
			self.z = []
			self.intensity = []
			self.time = []	

p = dataParser('/home/wendyyu/Downloads/RadarEvaluation_0405_publicroad_0.bag') # argument
boschdata = p.rosbagParser()
arrx = np.array(boschdata.x)
arry = np.array(boschdata.y)
arrz = np.array(boschdata.z)
arr = np.append(arrx.T, arry.T, axis = 1)
arr = np.append(arr, arrz.T, axis = 1)
print(np.shape(arr))
# print(boschdata.x)
# print(arrx)
# print(arrx[1])
# print(boschdata.x[0])
# print(arrx[0][0])
# print(arrx[0][0])
# print(arry[0][1])
# print(arrz[0][2])
# print(np.shape(boschdata.x))
# print(type(boschdata.x))
# print(type(arrx))
# print(np.shape(arry))
# print(np.shape(arrz))


