from display import Display
from extractor import Extractor
from convertor import cart2hom
from normalize import compute_essential_normalized, compute_P_from_essential, reconstruct_one_point, triangulation


import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

display = Display()
extractor = Extractor()


def process(img, bb, intrinsic, camera_mask, timestamp):
	pts1, pts2, kpts, matches = extractor.extract_keypoints(img=img, bb=bb, camera_mask=camera_mask, timestamp=timestamp)
	# print(pts1)
	# converto to 3 dimensional
	points1 = cart2hom(pts1)
	points2 = cart2hom(pts2)
	# print(points1.shape)
	img_h, img_w, img_ch = img.shape
	intrinsic = np.reshape(intrinsic, (3, 3))
	# print(intrinsic)
	# intrinsic = np.array([[3000,0,img_w/2],
	# 			[0,3000,img_h/2],
	# 			[0,0,1]])
	tripoints3d = []
	
	if points1.ndim != 1 or points2.ndim != 1:
		points1_norm = np.dot(np.linalg.inv(intrinsic), points1)
		points2_norm = np.dot(np.linalg.inv(intrinsic), points2)
		E = compute_essential_normalized(points1_norm, points2_norm)
		P1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
		P2s = compute_P_from_essential(E)
		ind = -1
		# print("4 solutions", P2s)
		for i, P2 in enumerate(P2s):
			d1 = reconstruct_one_point(points1_norm[:, 0], points2_norm[:, 0], P1, P2)
			# print("Current d1:", d1)
			P2_homogenous = np.linalg.inv(np.vstack([P2, [0,0,0,1]]))

			d2 = np.dot(P2_homogenous[:3, :4], d1)
			
			if d1[2] > 0 and d2[2] > 0:
				# print("Current d:", d1[2], d2[2])
				ind = i
		ind = 2

		P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
		tripoints3d = triangulation(points1_norm, points2_norm, P1, P2)
		
		# mask = np.any(abs(tripoints3d) > 5, axis=0)
		# tripoints3d = tripoints3d[:, ~mask]
		if (timestamp == "1681710720_131437318"):
			for x, y in zip(points1[0], points1[1]):
				cv2.circle(img, (int(x), int(y)), radius=2, color=(0, 255, 0), thickness=-1)
			cv2.imshow("img1", img)
			fig1 = plt.figure(1)
			plt.gca().invert_yaxis()
			plt.plot(points1_norm[0], points1_norm[1], 'o')
			plt.plot(points2_norm[0], points2_norm[1], 'o')
			# Adding labels and title
			plt.xlabel('X-axis')
			plt.ylabel('Y-axis')
			plt.title('2D Points')
			# Create a figure and a 3D axis
			fig2 = plt.figure(2)
			# ax = fig2.add_subplot(11, projection='2d')
			print(tripoints3d[:2, :])
			# Extract the x, y, and z coordinates from 'tripoints3d_all'
			x = tripoints3d[0]
			y = tripoints3d[1]
			# z = tripoints3d[2]
			z = np.zeros((tripoints3d.shape[1],))
			# # Create the 3D scatter plot
			# ax.scatter(x, y, c='b', marker='o')

			# # Set labels for the axes
			# ax.set_xlabel('X')
			# ax.set_ylabel('Y')
			# ax.set_zlabel('Z')
			
			# Plotting the (x, y) points
			plt.scatter(x, y)
			# Show the plot
			plt.show()
			exit()
	else:
		print("Wrong dimension of array")
		pass

	return img, tripoints3d, kpts, matches
