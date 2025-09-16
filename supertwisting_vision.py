#!/usr/bin/env python
# license removed for brevity
import sys
import time
import datetime
import rospy
import math as m
import numpy as np
import tf
import tf2_ros as tf2
from numpy import interp
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String, Bool, UInt16, Float64
from geometry_msgs.msg import TwistStamped, Twist, PoseStamped, Pose
from sensor_msgs.msg import Imu

class rhonn_controller():
	def __init__(self):
		rospy.init_node("rhonn_control_py")

		self.T = 0.05
		self.initial_time = time.time()
		self.x_filtrada_prev = -2.2
		self.y_filtrada_prev = -0.5
		self.x = 0
		self.y = 0
		self.z = 0
		self.yaw = 0
		self.vx = 0
		self.vy = 0
		self.vz = 0
		self.vyaw = 0
		self.vel_yaw = 0

		self.x_prev = 0
		self.y_prev = 0
		self.z_prev = 0

		self.inputs = np.zeros((4,1))		

		self.zx_0 = 0
		self.zy_0 = 0
		self.zz_0 = 0
		self.zyaw_0 = 0
		self.zx_1 = 0
		self.zy_1 = 0
		self.zz_1 = 0
		self.zyaw_1 = 0
		self.zx_2 = 0
		self.zy_2 = 0
		self.zz_2 = 0
		self.zyaw_2 = 0
		self.zx_3 = 0
		self.zy_3 = 0
		self.zz_3 = 0
		self.zyaw_3 = 0

		self.beta_x=25
		self.k1_x=40
		self.k2_x=1
		self.beta_y=25
		self.k1_y=40
		self.k2_y=3
		self.beta_z=35
		self.k1_z=20
		self.k2_z=3
		self.beta_yaw=35
		self.k1_yaw=10
		self.k2_yaw=1

		self.sig_x_prev = 0 	
		self.sig_y_prev = 0 	
		self.sig_z_prev = 0 	
		self.sig_yaw_prev = 0 	
		self.sig_x_int = 0 
		self.sig_y_int = 0 
		self.sig_z_int = 0 
		self.sig_yaw_int = 0 	
		
		self.pos_prev = np.zeros((4,1))
		self.state = np.zeros((8,1))

		self.u = np.zeros((8,1))

		self.local_inputs = np.zeros((4,1))

		#print("Type the position to track")
		self.x_desired = -1
		self.y_desired = 0
		self.z_desired = 0
		self.yaw_desired = 0.20

		rospy.Subscriber("/BlueRov2/imu/data", Imu, self.imu_callback)

		rospy.Subscriber("/vision/states/x", Float64, self.x_callback)
		rospy.Subscriber("/vision/states/y", Float64, self.y_callback)
		rospy.Subscriber("/vision/states/z", Float64, self.z_callback)
		#rospy.Subscriber("/vision/states/vx", Float64, self.vx_callback)
		#rospy.Subscriber("/vision/states/vy", Float64, self.vy_callback)
		#rospy.Subscriber("/vision/states/vz", Float64, self.vz_callback)

		self.imu = Imu()
		#self.x = 0
		#self.y = 0
		#self.z = 0
		self.vel = np.zeros((4,1))
		self.timestamp = rospy.Time()

		self.pub_vertical=rospy.Publisher("BlueRov2/rc_channel3/set_pwm", UInt16, queue_size=1)
		self.pub_lateral=rospy.Publisher("BlueRov2/rc_channel6/set_pwm", UInt16, queue_size=1)
		self.pub_forward=rospy.Publisher("BlueRov2/rc_channel5/set_pwm", UInt16, queue_size=1)
		self.pub_yaw=rospy.Publisher("BlueRov2/rc_channel4/set_pwm", UInt16, queue_size=1)

		self.pub_states_x=rospy.Publisher("FRHONN/states/x", Float64, queue_size=1)
		self.pub_states_y=rospy.Publisher("FRHONN/states/y", Float64, queue_size=1)
		self.pub_states_z=rospy.Publisher("FRHONN/states/z", Float64, queue_size=1)
		self.pub_states_yaw=rospy.Publisher("FRHONN/states/yaw", Float64, queue_size=1)
		self.pub_states_vx=rospy.Publisher("FRHONN/states/vx", Float64, queue_size=1)
		self.pub_states_vy=rospy.Publisher("FRHONN/states/vy", Float64, queue_size=1)
		self.pub_states_vz=rospy.Publisher("FRHONN/states/vz", Float64, queue_size=1)
		self.pub_states_vyaw=rospy.Publisher("FRHONN/states/vyaw", Float64, queue_size=1)

		self.pub_tracking_error_x=rospy.Publisher("FRHONN/error/x", Float64, queue_size=1)
		self.pub_tracking_error_y=rospy.Publisher("FRHONN/error/y", Float64, queue_size=1)
		self.pub_tracking_error_z=rospy.Publisher("FRHONN/error/z", Float64, queue_size=1)
		self.pub_tracking_error_yaw=rospy.Publisher("FRHONN/error/yaw", Float64, queue_size=1)
		self.pub_tracking_error_vx=rospy.Publisher("FRHONN/error/vx", Float64, queue_size=1)
		self.pub_tracking_error_vy=rospy.Publisher("FRHONN/error/vy", Float64, queue_size=1)
		self.pub_tracking_error_vz=rospy.Publisher("FRHONN/error/vz", Float64, queue_size=1)
		self.pub_tracking_error_vyaw=rospy.Publisher("FRHONN/error/vyaw", Float64, queue_size=1)

		self.pub_inputs_fwd=rospy.Publisher("FRHONN/inputs/forward", Float64, queue_size=1)
		self.pub_inputs_lat=rospy.Publisher("FRHONN/inputs/lateral", Float64, queue_size=1)
		self.pub_inputs_ver=rospy.Publisher("FRHONN/inputs/vertical", Float64, queue_size=1)
		self.pub_inputs_yaw=rospy.Publisher("FRHONN/inputs/yaw", Float64, queue_size=1)

		self.pub_inputs_fwd_2=rospy.Publisher("FRHONN/inputs/forward_2", Float64, queue_size=1)
		self.pub_inputs_lat_2=rospy.Publisher("FRHONN/inputs/lateral_2", Float64, queue_size=1)
		self.pub_inputs_ver_2=rospy.Publisher("FRHONN/inputs/vertical_2", Float64, queue_size=1)
		self.pub_inputs_yaw_2=rospy.Publisher("FRHONN/inputs/yaw_2", Float64, queue_size=1)

		self.pub_mode = rospy.Publisher('/BlueRov2/mode/set', String, queue_size=1)

		self.pub_mode.publish("alt_hold")
        	#self.arm_service = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)

		current_time = str(datetime.datetime.now())		
		current_time = current_time.replace(':', '-')
		header = "t\tx\ty\tv_x\tv_y\te_x\te_y\te_vx\te_vy\tu_x\tu_y"
		self.file = open(current_time + ".txt",'a')
		self.file.write(header + "\n")

	def imu_callback(self, data):
		#self.timestamp = data.header.stamp		
		self.yaw = data.orientation.z
		lapse = time.time() - self.initial_time
		while lapse < 1:
			self.yaw_desired = self.yaw
		self.vel_yaw = data.angular_velocity.z
		self.vel_yaw = 0

	def x_callback(self, data):
		self.x = data.data

	def y_callback(self, data):
		self.y = data.data

	def z_callback(self, data):
		self.z = data.data

	def vx_callback(self, data):
		self.vx = data.data
		self.vx = 0

	def vy_callback(self, data):
		self.vy = data.data
		self.vy = 0

	def vz_callback(self, data):
		self.vz = data.data
		self.vz = 0


	def get_state(self):
		#print(self.x, self.y, self.z, self.yaw)
		self.state[0:4, 0] = [self.x, self.y, self.z, self.yaw]

		#self.state[4:8, 0] = [self.vx, self.vy, self.vz, self.vyaw]

		self.state[4:7, 0] = self.levant(self.state[0:3, 0])
		#print(self.state[4:7, 0])

		#self.state[4:7, 0] = self.derivative(self.state[0:3, 0])
		#print(self.state[4:7, 0])
		
		self.state[4:8, 0] = [0, 0, 0, 0]

		self.state[7, 0] = self.vel_yaw
#		self.pos_prev = self.state[0:4,0]

	
	def control_inputs(self):
		theta = self.state[3]	
		rot = np.array([[m.cos(theta), - m.sin(theta)], [m.sin(theta), m.cos(theta)]])
		
		self.state_desired = np.array([[self.x_desired],[ self.y_desired],[ self.z_desired], [self.yaw_desired], [0], [0], [0], [0]])
		self.state_desired_d = 0 * np.ones((8,1))

		#self.state_desired[0:2, 0] = self.trayectory_1()
		self.state_desired[0:2, 0] = self.trayectory_2()
		print(self.state_desired)
		self.tracking_error = self.state_desired - self.state

		s_x = self.tracking_error[4] + self.beta_x * self.tracking_error[0]
		self.sig_x_int = self.sig_x_int + (np.sign(s_x) + self.sig_x_prev) * self.T / 2
		self.inputs[0] = 1500 - (- self.beta_x * self.tracking_error[4] - self.k1_x * m.sqrt(abs(s_x)) * np.sign(s_x) - self.k2_x * self.sig_x_int)

		s_y = self.tracking_error[5] + self.beta_y * self.tracking_error[1]
		self.sig_y_int = self.sig_y_int + (np.sign(s_y) + self.sig_y_prev) * self.T / 2
		self.inputs[1] = 1500 - (- self.beta_y * self.tracking_error[4] - self.k1_y * m.sqrt(abs(s_y)) * np.sign(s_y) - self.k2_y * self.sig_y_int)

		s_z = self.tracking_error[6] + self.beta_z * self.tracking_error[2]
		self.sig_z_int = self.sig_z_int + (np.sign(s_z) + self.sig_z_prev) * self.T / 2
		self.inputs[2] = 1500 + (- self.beta_z * self.tracking_error[6] - self.k1_z * m.sqrt(abs(s_z)) * np.sign(s_z) - self.k2_z * self.sig_z_int)

		s_yaw = self.tracking_error[7] + self.beta_yaw * self.tracking_error[3]
		self.sig_yaw_int = self.sig_yaw_int + (np.sign(s_yaw) + self.sig_yaw_prev) * self.T / 2
		self.inputs[3] = 1500 + (- self.beta_yaw * self.tracking_error[7] - self.k1_yaw * m.sqrt(abs(s_yaw)) * np.sign(s_yaw) - self.k2_yaw * self.sig_yaw_int)


		self.sig_x_prev = np.sign(s_x)	
		self.sig_y_prev = np.sign(s_y)	
		self.sig_z_prev = np.sign(s_z)	
		self.sig_yaw_prev = np.sign(s_yaw)		
		
		self.inputs[0] = interp(self.inputs[0],[1100,1900],[-100,100])
		self.inputs[1] = interp(self.inputs[1],[1100,1900],[-100,100])

		self.inputs[0:2] = (np.matmul(rot, self.inputs[0:2]))

		self.inputs[0] = interp(self.inputs[0],[-100,100],[1300,1700])
		self.inputs[1] = interp(self.inputs[1],[-100,100],[1300,1700])
		self.inputs[2] = self.pwm_limit(self.inputs[2])
		self.inputs[3] = self.pwm_limit(self.inputs[3])

		self.pub_states_x.publish(self.state[0])
		self.pub_states_y.publish(self.state[1])
		self.pub_states_z.publish(self.state[2])
		self.pub_states_yaw.publish(self.state[3])
		self.pub_states_vx.publish(self.state[4])
		self.pub_states_vy.publish(self.state[5])
		self.pub_states_vz.publish(self.state[6])
		self.pub_states_vyaw.publish(self.state[7])
		self.pub_tracking_error_x.publish(self.tracking_error[0])
		self.pub_tracking_error_y.publish(self.tracking_error[1])
		self.pub_tracking_error_z.publish(self.tracking_error[2])
		self.pub_tracking_error_yaw.publish(self.tracking_error[3])
		self.pub_tracking_error_vx.publish(self.state_desired[0])
		self.pub_tracking_error_vy.publish(self.state_desired[1])
		self.pub_tracking_error_vz.publish(self.tracking_error[6])
		self.pub_tracking_error_vyaw.publish(self.tracking_error[7])

		if self.inputs[0] == self.inputs[0]:
			self.pub_inputs_fwd.publish(self.inputs[0])
			self.pub_inputs_lat.publish(self.inputs[1])
			self.pub_inputs_ver.publish(self.inputs[2])
			self.pub_inputs_yaw.publish(self.inputs[3])
		#self.pub_inputs_fwd_2.publish(inputs[0])
		#self.pub_inputs_lat_2.publish(inputs[1])
		#self.pub_inputs_ver_2.publish(inputs[2])
		#self.pub_inputs_yaw_2.publish(inputs[3])
		#print(inputs)
			#print('ok')
			#print(self.tracking_error)

			self.pub_forward.publish(self.inputs[0])
			self.pub_lateral.publish(self.inputs[1])
			#self.pub_vertical.publish(self.inputs[2])
			self.pub_yaw.publish(self.inputs[3])

	def pwm_limit(self, n):
		n_min = 1300
		n_max = 1700
		return max(min(n_max, n),n_min)

	def trayectory_2(self):
		current_time = time.time()
		lapse = current_time - self.initial_time
		tau = 10
		pos_x = (-2.2, -1.7)
		pos_y = (-0.5, 0, 0.5)
		#x_filtrada = -2.5
		#y_filtrada = -0.5
		print(lapse)
		if lapse < 20:
			x = pos_x[0]
			#x = 0
			y = pos_y[0]
		if lapse > 20 and lapse <40:
			x = pos_x[1]
			y = pos_y[0]
		if lapse > 40 and lapse <60:
			x = pos_x[1]
			y = pos_y[1]
		if lapse > 60 and lapse <80:
			x = pos_x[0]
			y = pos_y[1]
		if lapse > 80 and lapse <100:
			x = pos_x[0]
			y = pos_y[2]
		if lapse > 100 and lapse <120:
			x = pos_x[1]
			y = pos_y[2]
		x_filtrada = self.x_filtrada_prev + (self.T / tau) * (x - self.x_filtrada_prev)
		self.x_filtrada_prev = x_filtrada
		y_filtrada = self.y_filtrada_prev + (self.T / tau) * (y - self.y_filtrada_prev)
		self.y_filtrada_prev = y_filtrada
		#print((self.T))
		return [x_filtrada, y_filtrada]

	def trayectory_1(self):
		current_time = time.time()
		lapse = current_time - self.initial_time
		pos_x = (-2)
		pos_y = (-0.5, 0, 0.5)
		if lapse < 60:
			x = pos_x
			y = pos_y[0]
		if lapse > 60 and lapse <120:
			x = pos_x
			y = pos_y[1]
		if lapse > 120 and lapse <180:
			x = pos_x
			y = pos_y[2]


		return [x, y]

	def derivative(self, state):

		vx = ( state[0] - self.x_prev ) / self.T
		vy = ( state[1] - self.y_prev ) / self.T
		vz = ( state[2] - self.z_prev ) / self.T

		self.vel[0, 0] = vx
		self.vel[1, 0] = vy
		self.vel[2, 0] = vz

		self.x_prev = state[0]
		self.y_prev = state[1]
		self.z_prev = state[2]

		return self.vel[0:3, 0]


	def levant(self, state):

		vx_0 = self.zx_1 - 21.213203435596427 * m.pow(abs(self.zx_0 - state[0]), 0.75) * np.sign(self.zx_0 - state[0])
		vx_1 = self.zx_2 - 27.144176165949062 * m.pow(abs(self.zx_1 - vx_0), 0.6666) * np.sign(self.zx_1 - vx_0)
		vx_2 = self.zx_3 - 75 * m.pow(abs(self.zx_2 - vx_1), 0.5) * np.sign(self.zx_2 - vx_1)
		self.zx_0 = self.zx_0 + self.T * vx_0
		self.zx_1 = self.zx_1 + self.T * vx_1
		self.zx_2 = self.zx_2 + self.T * vx_2
		self.zx_3 = self.zx_3 + self.T * (-2750 * np.sign(self.zx_3 - vx_2))
		self.vel[0, 0]

		vy_0 = self.zy_1 - 21.213203435596427 * m.pow(abs(self.zy_0 - state[1]), 0.75) * np.sign(self.zy_0 - state[1])
		vy_1 = self.zy_2 - 27.144176165949062 * m.pow(abs(self.zy_1 - vy_0), 0.6666) * np.sign(self.zy_1 - vy_0)
		vy_2 = self.zy_3 - 75 * m.pow(abs(self.zy_2 - vy_1), 0.5) * np.sign(self.zy_2 - vy_1)
		self.zy_0 = self.zy_0 + self.T * vy_0
		self.zy_1 = self.zy_1 + self.T * vy_1
		self.zy_2 = self.zy_2 + self.T * vy_2
		self.zy_3 = self.zy_3 + self.T * (-2750 * np.sign(self.zy_3 - vy_2))
		self.vel[1, 0] = vy_0

		vz_0 = self.zz_1 - 10.498906534741749 * m.pow(abs(self.zz_0 - state[2]), 0.75) * np.sign(self.zz_0 - state[2])
		vz_1 = self.zz_2 - 10.626585691826110 * m.pow(abs(self.zz_1 - vz_0), 0.6666) * np.sign(self.zz_1 - vz_0)
		vz_2 = self.zz_3 - 18.371173070873837 * m.pow(abs(self.zz_2 - vz_1), 0.5) * np.sign(self.zz_2 - vz_1)
		self.zz_0 = self.zz_0 + self.T * vz_0
		self.zz_1 = self.zz_1 + self.T * vz_1
		self.zz_2 = self.zz_2 + self.T * vz_2
		self.zz_3 = self.zz_3 + self.T * (-165 * np.sign(self.zz_3 - vz_2))
		self.vel[2, 0] = vz_0

		return self.vel[0:3, 0]

	def save_csv(self):

		#header = "t\tx\ty\tv_x\tv_y\te_x\te_y\te_vx\te_vy\tu_x\tu_y"	
		out = np.array([time.time(), self.state[0], self.state[1], self.state[4], self.state[5], self.tracking_error[0], self.tracking_error[1], self.tracking_error[4], self.tracking_error[5], self.inputs[0], self.inputs[1]])
		out = np.array(out).reshape((1, 11))

		np.savetxt(self.file, out, delimiter='\t')

def main():	
	rhoncon = rhonn_controller()
	rate = rospy.Rate(20)
	while not rospy.is_shutdown():
		rhoncon.get_state()
		rhoncon.control_inputs()
		rhoncon.save_csv()
		#rate.sleep()
	rospy.spin()


if __name__ == '__main__':	
	try:
		main()
	except rospy.ROSInterruptException:
		pass
