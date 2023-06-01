from slam import process
from pointmap import PointMap

import cv2
import csv
import os
import numpy as np
pmap = PointMap()

def main():
	with open('../ITRI_dataset/seq1/all_timestamp.txt', 'r') as file:
		lines = file.readlines()

	# Remove the newline character ('\n') from each line
	lines = [line.strip() for line in lines]

	print(lines[3])  # Print the 4th entry (index 3)
	camera_pos = ["f", "fl", "fr", "b"]
	for pos in camera_pos:
		print(pos)
		for i in range(len(lines)):
			input = cv2.imread("../ITRI_dataset/seq1/dataset/%s/raw_image.jpg" %(lines[i]))
			with open("../ITRI_dataset/seq1/dataset/%s/camera.csv" %(lines[i]), 'r') as file:
				camera = file.readline()
			if (camera == ("/lucid_cameras_x00/gige_100_%s_hdr" % pos)):
				#print(camera)
				#print(input.shape)
				#input = cv2.resize(input, (960, 540))
				#print(input.shape)
				img, tripoints, kpts, matches = process(input)
				xyz = pmap.collect_points(tripoints)
				# Create the new folder
				folder_path = "./output/%s" % (lines[i])
				os.makedirs(folder_path, exist_ok=True)

				# Generate the file path for the CSV file
				file_path = os.path.join(folder_path, 'output.csv')
				if (xyz is not None):
					np.savetxt(file_path, xyz, delimiter=',', fmt='%s')
				

if __name__ == '__main__':
	main()
