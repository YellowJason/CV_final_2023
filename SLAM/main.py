from slam import process
from display import Display
from pointmap import PointMap

import cv2
import csv
import os
import numpy as np
import open3d as o3d
import yaml
pmap = PointMap()
display = Display()

def main():
	pcd = o3d.geometry.PointCloud()
	visualizer = o3d.visualization.Visualizer()
	visualizer.create_window(window_name="3D plot", width=1440, height=928)
	with open('../ITRI_dataset/seq1/all_timestamp.txt', 'r') as file:
		lines = file.readlines()

	# Remove the newline character ('\n') from each line
	lines = [line.strip() for line in lines]

	# print(lines[3])  # Print the 4th entry (index 3)
	camera_pos = ["f", "fl", "fr", "b"]
	for pos in camera_pos:
		print("Current Processed camera: ", pos)
		camera_mask = cv2.imread("../ITRI_dataset/camera_info/lucid_cameras_x00/gige_100_%s_hdr_mask.png" % pos, 0)
		xyz_list = np.empty((0, 3))
		# print(camera_mask)
		with open("../ITRI_dataset/camera_info/lucid_cameras_x00/gige_100_%s_hdr_camera_info.yaml" % pos, "r") as cam_file:
			cam_info = yaml.safe_load(cam_file)
			cam_matrix = cam_info["camera_matrix"]["data"]
			# print(cam_matrix)
		for i in range(len(lines)):
			# if (i==7):
			# 	exit()
			input = cv2.imread("../ITRI_dataset/seq1/dataset/%s/raw_image.jpg" %(lines[i]))
			bb = np.genfromtxt("../ITRI_dataset/seq1/dataset/%s/detect_road_marker.csv" %(lines[i]), delimiter=",")
			with open("../ITRI_dataset/seq1/dataset/%s/camera.csv" %(lines[i]), 'r') as file:
				camera = file.readline()
			if (camera == ("/lucid_cameras_x00/gige_100_%s_hdr" % pos)):
				#print(camera)
				#print(input.shape)
				#input = cv2.resize(input, (960, 540))
				#print(input.shape)
				img, tripoints, kpts, matches = process(input, intrinsic = cam_matrix, bb = bb, camera_mask=camera_mask, timestamp=lines[i])
				xyz = pmap.collect_points(tripoints)
				
				if kpts is not None or matches is not None:
					display.display_points2d(img, kpts, matches)
				else:
					pass
				# cv2.imwrite("Current frame.jpg", display.display_points2d(img, kpts, matches))
				display.display_vid(input)
				if xyz is not None and xyz.shape[0] > 3:
					# print(xyz.shape)
					xyz_list = np.concatenate((xyz_list, xyz), axis=0)
					# print("XYZ shape", xyz_list.shape)
					if (i>300):
						np.savetxt("./concat.csv", xyz_list, delimiter=',', fmt='%s')
						exit()
					display.display_points3d(xyz, pcd, visualizer)
				else:
					pass
				if cv2.waitKey(1) == 27:
					break
				# Create the new folder
				print("Current Timestamps: ", lines[i])
				folder_path = "./output/%s" % (lines[i])
				os.makedirs(folder_path, exist_ok=True)

				# Generate the file path for the CSV file
				file_path = os.path.join(folder_path, 'output.csv')
				if (xyz is not None):
					np.savetxt(file_path, xyz, delimiter=',', fmt='%s')
	cv2.destroyAllWindows()		

if __name__ == '__main__':
	main()
