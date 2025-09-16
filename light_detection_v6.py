#!/usr/bin/env python
"""
BlueRov video capture class
"""

import cv2
import gi
import numpy as np
import time
import math as m
import rospy
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import String, Bool, UInt16, Float64
from video import Video


gi.require_version('Gst', '1.0')
from gi.repository import Gst

class Estimator():

	def __init__(self):

		self.vel = np.zeros((3,1))

		self.zx_0 = 0
		self.zy_0 = 0
		self.zz_0 = 0
		self.zx_1 = 0
		self.zy_1 = 0
		self.zz_1 = 0
		self.zx_2 = 0
		self.zy_2 = 0
		self.zz_2 = 0
		self.zx_3 = 0
		self.zy_3 = 0
		self.zz_3 = 0

		self.x_prev = 0
		self.y_prev = 0
		self.z_prev = 0
		
		self.previous_time = 0

if __name__ == '__main__':
    rospy.init_node("light_detection_v4")
    pub_vision_estimated_x=rospy.Publisher("vision/states/x", Float64, queue_size=1)
    pub_vision_estimated_y=rospy.Publisher("vision/states/y", Float64, queue_size=1)
    pub_vision_estimated_z=rospy.Publisher("vision/states/z", Float64, queue_size=1)
    pub_vision_time=rospy.Publisher("vision/timer", Float64, queue_size=1)


    video = Video()
    estimation = Estimator()
	
    width = 800
    height = 600

    focal_length = 6.3
    sensor_length = 8.75
    BD_length = 374

    cpv = height * 650 / 1080
    cph = width * 935 / 1920

    cpv = 300
    cph = 400

    cameraMatrix = np.array([[979.77, 0, 930.41], [0, 979.02, 468.14], [0, 0, 1]])
    dist = np.array([[0.097, -0.09, 0.001, 0.0349, 0.0749]])

    square_br1 = 0
    square_br2 = 0

    writer_og = cv2.VideoWriter('deteccion_og.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
    writer_frame = cv2.VideoWriter('deteccion_frame.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
    writer_lmask = cv2.VideoWriter('deteccion_lmask.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
    writer_bmask = cv2.VideoWriter('deteccion_bmask.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
    writer_gmask = cv2.VideoWriter('deteccion_gmask.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
    writer_rmask = cv2.VideoWriter('deteccion_rmask.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
    writer_hsv = cv2.VideoWriter('deteccion_hsv.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

    prev_frame_time = 0
    new_frame_time = 0
    lights_prev = np.empty((1, 3))
  
    while True:
      if video.frame_available():
        frame = video.frame()
	#print(frame.shape)
	og_frame = video.frame()

	h, w = frame.shape[:2]
	newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

	dst = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)
	x, y, w, h = roi
	frame = dst[y:y + h, x:x + w]

	#print(frame)
	#writer.write(og_frame)

    	check_pixels = np.zeros(3)
    	rgb = ("RED", "GREEN", "BLUE")
	color = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
	light_system = ("A", "B", "C", "D", "E")
    	blurred_frame = cv2.GaussianBlur(frame, (9, 9), 0)
    	hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)



	lower = np.array([0, 3, 3])
 	upper = np.array([179, 180, 255])
 	mask_light = cv2.inRange(hsv, lower, upper)
 	mask_green = cv2.inRange(hsv, (36, 3, 3), (86, 255, 255))
 	mask_red = cv2.inRange(hsv, (167, 5, 5), (179, 255, 255))
	mask_red_2 = cv2.inRange(hsv, (1, 5, 5), (16, 255, 255))
	mask_red = cv2.bitwise_or(mask_red, mask_red_2)
  	mask_blue = cv2.inRange(hsv, (91, 5, 5), (130 , 255, 255))
  	#mask_light = cv2.bitwise_or(mask_light, mask_blue)
  	#mask_light = cv2.bitwise_or(mask_light, mask_green)
  	mask_light = cv2.bitwise_or(mask_light, mask_red)


	#print(frame)
	contours_light, hierarchy_light = cv2.findContours(mask_light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

	if len(contours_light)!=0:

		hierarchy_light = hierarchy_light[0]
	
		lights = np.empty((len(contours_light), 3))
		lights.fill(np.nan)
		#print(len(contours_light))	
		index = -1

	
		for component in zip(contours_light, hierarchy_light):
			index += 1
			#print(index)        	
			contour = component[0]
        		hierarchy = component[1]
			if len(contour) < 3: 
				continue
        		area = cv2.contourArea(contour)
			area_contour = area
			#print(area_contour)
			if area == 0:
				continue
        		if area > 1:
            			convexhull = cv2.convexHull(contour)
            			area_contour = cv2.contourArea(convexhull)
            			if area / area_contour > 0.01:
                			poly = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                			if 1 < len(poly):
                    			#convexity_defects = cv2.convexityDefects(contour, convexhull)
                    				moments = cv2.moments(contour)
                    				center_position_x = int(moments["m10"] / moments["m00"])
                    				center_position_y = int(moments["m01"] / moments["m00"])
                    				if center_position_y < 200:
                        				continue

                    				height, width, depth = frame.shape
                    				check = np.zeros((height, width), dtype=np.uint8)
                   				check = cv2.circle(check, (center_position_x, center_position_y), 30, 255, -1)
                    				check_red = cv2.bitwise_and(mask_red, check)
                    				check_green = cv2.bitwise_and(mask_green, check)
                    				check_blue = cv2.bitwise_and(mask_blue, check)
                    				check_pixels[0] = np.sum(check_red == 255)
                    				check_pixels[1] = np.sum(check_green == 255)
                    				check_pixels[2] = np.sum(check_blue == 255)
                    				max_value = max(check_pixels)
						if max_value == 0:
							continue
                    				max_index = np.argmax(check_pixels)
						lights[index] = [max_index, center_position_x, center_position_y]

                    				cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
                    				cv2.circle(frame, (center_position_x, center_position_y), 5, (0, 0, 255), -1)

    		lights_leader = np.empty_like(lights)
    		lights_leader.fill(np.nan)
    		lights_prev = lights
		#print(lights)
    		if len(lights_prev) == 1:
        		lights_prev.fill(np.nan)
    		i = 0
    		for j in range(len(lights)):
			#if i == 5:
				#continue
        		if lights[j, [0]] != lights[j, [0]]:
            			continue
        		for k in range(len(lights_prev)):
            			if lights_prev[k, 0] != lights[j, 0]:
                			continue
            			#x_error = abs(lights_prev[k, [1]] - lights[j, [1]]) / abs(lights_prev[k, [1]])
            			#y_error = abs(lights_prev[k, [2]] - lights[j, [2]]) / abs(lights_prev[k, [2]])

            			#if x_error or y_error > 0.01:
                			#continue
            			lights_leader[j, :] = lights[j, :]
            			#print(lights)
				#print(lights_leader)
				i += 1
		#if np.isnan(lights).all:
			#cv2.putText(frame, 'No lights detected', (int(0.05*width), int(0.9*height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)	

    		lights_leader_ord = np.empty((5, 3))
    		lights_leader_ord.fill(np.nan)
		distances = np.empty((6, 2))
		distances.fill(np.nan)
    		index_b = np.asarray(np.where(lights_leader[:, 0] == 0))
    		index_d = np.asarray(np.where(lights_leader[:, 0] == 2))
		#print(index_b)
    		#print(lights_leader)
    		if len(index_b[0]) != 0:
        		index_b = np.squeeze(index_b, axis=0)
        		possible_b = lights_leader[index_b, :]
        		index_b = np.argmax(possible_b[:, 2])
        		#index_b2 = np.asarray(np.where(possible_b == np.amax(possible_b[:, 2])))
        		lights_leader_ord[1, :] = possible_b[[index_b], :]
        		#lights_leader_ord[1, :] = lights_leader[[index_b[0,0]], :]
    		if len(index_d[0]) != 0:
        		index_d = np.squeeze(index_d, axis=0)
        		possible_d = lights_leader[index_d, :]
        		index_d = np.argmax(possible_d[:, 2])
        		lights_leader_ord[3, :] = possible_d[[index_d], :]
        		#lights_leader_ord[3, :] = lights_leader[[index_d[0,0]], :]
    		index_a = np.where((lights_leader[:, 0] == 1) & (lights_leader[:, 1] < lights_leader_ord[1, 1]))
    		if len(index_a[0]) != 0:
        		index_a = np.squeeze(np.asarray(index_a), axis=0)
        		possible_a = lights_leader[index_a, :]
        		index_a = np.argmax(possible_a[:, 2])
        		lights_leader_ord[0, :] = possible_a[[index_a], :]
        		#lights_leader_ord[0, :] = lights_leader[[index_a], :]

    		index_c = np.where((lights_leader[:, 0] == 1) & (lights_leader[:, 2] < lights_leader_ord[1, 2]))
    		if len(index_c[0]) != 0:
        		index_c = np.squeeze(np.asarray(index_c), axis=0)
        		possible_c = lights_leader[index_c, :]
        		index_c = np.argmax(possible_c[:, 2])
        		lights_leader_ord[2, :] = possible_c[[index_c], :]
        		#lights_leader_ord[2, :] = lights_leader[[index_c], :]
    		index_e = np.where((lights_leader[:, 0] == 1) & (lights_leader[:, 1] > lights_leader_ord[3, 1]))
    		if len(index_e[0]) != 0:
        		index_e = np.squeeze(np.asarray(index_e), axis=0)
        		possible_e = lights_leader[index_e, :]
        		index_e = np.argmax(possible_e[:, 2])
        		lights_leader_ord[4, :] = possible_e[[index_e], :]  
        		#lights_leader_ord[4, :] = lights_leader[[index_e], :]
    		#print(lights_leader_ord)
    		for i in range(len(lights_leader_ord)):
        		#print(i)
        		if lights_leader_ord[i, 0] != lights_leader_ord[i, 0]:
            			continue
        		frame = cv2.putText(frame, light_system[i], (int(lights_leader_ord[i, 1]), int(lights_leader_ord[i, 2])), cv2.FONT_HERSHEY_SIMPLEX, 1, color[int(lights_leader_ord[i, 0])], 2, cv2.LINE_AA)
    		if lights_leader_ord[1, 0] == 0 and lights_leader_ord[3, 0] == 2:
        		distance_x = lights_leader_ord[3, 1] - lights_leader_ord[1, 1]
        		distance_y = lights_leader_ord[3, 2] - lights_leader_ord[1, 2]
        		distances[0, 0] = (m.sqrt(distance_x * distance_x + distance_y * distance_y))
        		distances[0, 1] = (m.atan(distance_y / distance_x) * 180 / m.pi)
        		coordinate_1 = tuple((lights_leader_ord[1, -2:].astype(np.int32)).tolist())
        		coordinate_2 = tuple((lights_leader_ord[3, -2:].astype(np.int32)).tolist())
        		position_x = int(lights_leader_ord[1, 1] + distance_x / 2)
        		position_y = int(lights_leader_ord[1, 2] + distance_y / 2)
        		square_br1 = (int(round(position_x - 0.3 * distances[0, 0])), int(round(position_y - 0.3 * distances[0, 0])))
        		square_br2 = (int(round(position_x + 0.3 * distances[0, 0])), int(round(position_y + 0.3 * distances[0, 0])))
        		cv2.rectangle(frame, square_br1, square_br2, (0, 255, 255), 2, cv2.LINE_4)
        		dh = - cph + position_x
        		dv = - cpv + position_y
        		angle_h = (dh * (64 / 2) / cph) * (m.pi / 180)
        		angle_v = (dv * (80 / 2)/ cpv) * (m.pi / 180)

        		estimated_T = -(focal_length * BD_length * width) / (sensor_length * distances[0, 0]) / 1000
			estimated_x = m.cos(angle_h) * estimated_T        		
			estimated_y = m.tan(angle_h) * estimated_x
        		estimated_z = m.tan(angle_v) * estimated_x
			state = np.array([[estimated_x],[estimated_y],[estimated_z]])


        		string = "x={} y={} z={}".format(round(estimated_x, 2), round(estimated_y, 2), round(estimated_z, 2))
        		frame = cv2.putText(frame, string, (int(0.4*width), int(0.9*height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 155, 155), 2, cv2.LINE_AA)
        
        		#string = "{}px. {}".format(round(distances[0, 0], 2), round(distances[0, 1]), 2)
        		#frame = cv2.putText(frame, string, (position_x, position_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        		frame = cv2.line(frame, coordinate_1, coordinate_2, (0, 255, 255), 3)

			x = Float64()
			x.data = estimated_x
			y = Float64()
			y.data = estimated_y
			z = Float64()
			z.data = estimated_z
			pub_vision_estimated_x.publish(x)
			pub_vision_estimated_y.publish(y)
			pub_vision_estimated_z.publish(z)

    		if lights_leader_ord[0, 0] == 1 and lights_leader_ord[4, 0] == 1:
        		distance_x = lights_leader_ord[4, 1] - lights_leader_ord[0, 1]
        		distance_y = lights_leader_ord[4, 2] - lights_leader_ord[0, 2]
        		distances[1, 0] = (m.sqrt(distance_x * distance_x + distance_y * distance_y))
        		distances[1, 1] = (m.atan(distance_y / distance_x) * 180 / m.pi)
        		coordinate_1 = tuple((lights_leader_ord[0, -2:].astype(np.int32)).tolist())
        		coordinate_2 = tuple((lights_leader_ord[4, -2:].astype(np.int32)).tolist())
        		position_x = int(lights_leader_ord[0, 1] + distance_x / 2)
        		position_y = int(lights_leader_ord[0, 2] + distance_y / 2)
        		#string = "{}px. {}".format(round(distances[1, 0], 2), round(distances[1, 1]), 2)
        		#frame = cv2.putText(frame, string, (position_x, position_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        		frame = cv2.line(frame, coordinate_1, coordinate_2, (0, 255, 255), 3)

    		if lights_leader_ord[0, 0] == 1 and lights_leader_ord[1, 0] == 0:
        		distance_x = lights_leader_ord[1, 1] - lights_leader_ord[0, 1]
        		distance_y = lights_leader_ord[1, 2] - lights_leader_ord[0, 2]
        		distances[4, 0] = (m.sqrt(distance_x * distance_x + distance_y * distance_y))
        		distances[4, 1] = (m.atan(distance_y / distance_x) * 180 / m.pi)
        		coordinate_1 = tuple((lights_leader_ord[0, -2:].astype(np.int32)).tolist())
        		coordinate_2 = tuple((lights_leader_ord[1, -2:].astype(np.int32)).tolist())
        		position_x = int(lights_leader_ord[0, 1] + distance_x / 2)
        		position_y = int(lights_leader_ord[0, 2] + distance_y / 2)
        		#string = "{}px. {}".format(round(distances[4, 0], 2), round(distances[4, 1]), 2)	
        		#frame = cv2.putText(frame, string, (position_x, position_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        		frame = cv2.line(frame, coordinate_1, coordinate_2, (0, 255, 255), 3)

    		if lights_leader_ord[3, 0] == 2 and lights_leader_ord[4, 0] == 1:
        		distance_x = lights_leader_ord[4, 1] - lights_leader_ord[3, 1]
        		distance_y = lights_leader_ord[4, 2] - lights_leader_ord[3, 2]
        		distances[5, 0] = (m.sqrt(distance_x * distance_x + distance_y * distance_y))
        		distances[5, 1] = (m.atan(distance_y / distance_x) * 180 / m.pi)
        		coordinate_1 = tuple((lights_leader_ord[3, -2:].astype(np.int32)).tolist())
        		coordinate_2 = tuple((lights_leader_ord[4, -2:].astype(np.int32)).tolist())
        		position_x = int(lights_leader_ord[3, 1] + distance_x / 2)
        		position_y = int(lights_leader_ord[3, 2] + distance_y / 2)
        		#string = "{}px. {}".format(round(distances[5, 0], 2), round(distances[5, 1]), 2)	
        		#frame = cv2.putText(frame, string, (position_x, position_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        		frame = cv2.line(frame, coordinate_1, coordinate_2, (0, 255, 255), 3)

   		if lights_leader_ord[1, 0] == 0 and lights_leader_ord[2, 0] == 1:
        		distance_x = lights_leader_ord[2, 1] - lights_leader_ord[1, 1]
        		distance_y = lights_leader_ord[2, 2] - lights_leader_ord[1, 2]
        		distances[2, 0] = (m.sqrt(distance_x * distance_x + distance_y * distance_y))
        		distances[2, 1] = (m.atan(distance_y / distance_x) * 180 / m.pi)
        		coordinate_1 = tuple((lights_leader_ord[1, -2:].astype(np.int32)).tolist())
        		coordinate_2 = tuple((lights_leader_ord[2, -2:].astype(np.int32)).tolist())
        		position_x = int(lights_leader_ord[1, 1] + distance_x / 2)
        		position_y = int(lights_leader_ord[1, 2] + distance_y / 2)
        		#string = "{}px. {}".format(round(distances[2, 0], 2), round(distances[2, 1]), 2)	
        		#frame = cv2.putText(frame, string, (position_x, position_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        		frame = cv2.line(frame, coordinate_1, coordinate_2, (0, 255, 255), 3)

    		if lights_leader_ord[2, 0] == 1 and lights_leader_ord[3, 0] == 2:
        		distance_x = lights_leader_ord[3, 1] - lights_leader_ord[2, 1]
        		distance_y = lights_leader_ord[3, 2] - lights_leader_ord[2, 2]
        		distances[3, 0] = (m.sqrt(distance_x * distance_x + distance_y * distance_y))
        		distances[3, 1] = (m.atan(distance_y / distance_x) * 180 / m.pi)
        		coordinate_1 = tuple((lights_leader_ord[2, -2:].astype(np.int32)).tolist())
        		coordinate_2 = tuple((lights_leader_ord[3, -2:].astype(np.int32)).tolist())
        		position_x = int(lights_leader_ord[2, 1] + distance_x / 2)
        		position_y = int(lights_leader_ord[2, 2] + distance_y / 2)
        		#string = "{}px. {}".format(round(distances[3, 0], 2), round(distances[3, 1]), 2)
        		#frame = cv2.putText(frame, string, (position_x, position_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        		frame = cv2.line(frame, coordinate_1, coordinate_2, (0, 255, 255), 3)
				
	#print(lights)	
	new_frame_time = time.time()
	pub_vision_time.publish(new_frame_time)
	font = cv2.FONT_HERSHEY_SIMPLEX
  	cv2.putText(frame, 'Time: {}'.format(new_frame_time), (400, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
	fps = 1 / (new_frame_time - prev_frame_time)
	prev_frame_time = new_frame_time
	fps = int(fps)
	fps = str(fps)
  	font = cv2.FONT_HERSHEY_SIMPLEX
  	cv2.putText(frame, 'FPS: {}'.format(fps), (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)

   	square_1 = (int(round(cph - 0.2 * cph)), int(round(cpv - 0.2 * cpv)))
    	square_2 = (int(round(cph + 0.2 * cph)), int(round(cpv + 0.2 * cpv)))

    	if square_1 > square_br1 and square_2 < square_br2:
        	color_deteccion = (0, 255, 0)
    	else:
        	color_deteccion = (0, 0, 255)

    	cv2.rectangle(frame, square_1, square_2, color_deteccion, 2, cv2.LINE_4)
    	cv2.line(frame, (int(round(cph - 0.08 * cph)), cpv), (int(round(cph + 0.08 * cph)), cpv), color_deteccion, 2)
    	cv2.line(frame, (cph, int(round(cpv - 0.08 * cpv))), (cph, int(round(cpv + 0.08 * cpv))), color_deteccion, 2)

	#print('ok')
   	#cv2.imshow('light mask', mask_light)
    	#cv2.imshow('light', light)
    	#cv2.imshow('green mask', mask_green)

    	#cv2.imshow('red mask', mask_red)

    	#cv2.imshow('blue mask', mask_blue)
	#print(frame.shape)
    	#cv2.imshow('og frame', og_frame)
    	#cv2.imshow('hsv', hsv)

	out_frame = cv2.resize(frame,(800, 600))
	out_og_frame = cv2.resize(og_frame,(800, 600))

	out_mask_light = cv2.resize(mask_light,(800, 600))
	#out_mask_light = cv2.cvtColor(out_mask_light, cv2.COLOR_GRAY2RGB)

	#out_mask_green = cv2.resize(mask_green,(800, 600))
	#out_mask_green = cv2.cvtColor(out_mask_green, cv2.COLOR_GRAY2RGB)

	#out_mask_red = cv2.resize(mask_red,(800, 600))
	#out_mask_red = cv2.cvtColor(out_mask_red, cv2.COLOR_GRAY2RGB)

	#out_mask_blue = cv2.resize(mask_blue,(800, 600))
	#out_mask_blue = cv2.cvtColor(out_mask_blue, cv2.COLOR_GRAY2RGB)

	out_hsv = cv2.resize(hsv,(800, 600))

    	cv2.imshow('frame', frame)
    	#cv2.imshow('frame_dist', frame_dist)
	writer_frame.write(out_frame)
	writer_og.write(out_og_frame)
	#writer_lmask.write(out_mask_light)
	#writer_gmask.write(out_mask_green)
	#writer_rmask.write(out_mask_red)
	#writer_bmask.write(out_mask_blue)
	#writer_hsv.write(out_hsv)

    	cv2.imshow('lights', mask_light)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

      #else:
	#break

    #frame.release()
    #out_frame.release()
    #og_frame.release()
    cv2.destroyAllWindows()
  
