# from display import Display
from extractor import Extractor
from convertor import cart2hom
from normalize import compute_essential_normalized, compute_P_from_essential, reconstruct_one_point, triangulation


import cv2
import numpy as np
import open3d as o3d


# display = Display()
extractor = Extractor()


def process(img_path, camera):
	pts1, pts2, kpts, matches, pts1_c, pts2_c = extractor.extract_keypoints(img_path)
	img = cv2.imread(f"{img_path}/raw_image.jpg")
	# print(pts1)
	# converto to 3 dimensional
	points1 = cart2hom(pts1)
	points2 = cart2hom(pts2)
	# points from find_corner
	points1_c = cart2hom(pts1_c)
	points2_c = cart2hom(pts2_c)
	# print(points1.shape)
	# print(points2.shape)
	img_h, img_w, img_ch = img.shape

	intrinsic_dic = {'f' : np.array([661.949026684, 0.0, 720.264314891, 0.0, 662.758817961, 464.188882538, 0.0, 0.0, 1.0]).reshape((3,3)),
		  			 'fl': np.array([658.929184246, 0.0, 721.005287695, 0.0, 658.798994733, 460.495402628, 0.0, 0.0, 1.0]).reshape((3,3)),
					 'fr': np.array([660.195664468, 0.0, 724.021995966, 0.0, 660.202323944, 467.498636505, 0.0, 0.0, 1.0]).reshape((3,3)),
					 'b' : np.array([658.897676983, 0.0, 719.335668486, 0.0, 659.869992391, 468.32106087, 0.0, 0.0, 1.0]).reshape((3,3))}
	intrinsic = intrinsic_dic[camera]
	
	tripoints3d = []
	if points1.ndim != 1 or points2.ndim != 1:
		points1_norm = np.dot(np.linalg.inv(intrinsic), points1)
		points2_norm = np.dot(np.linalg.inv(intrinsic), points2)
		if len(points1_c) == 3:
			# print('yyy', points1_c.shape)
			points1_norm_c = np.dot(np.linalg.inv(intrinsic), points1_c)
			points2_norm_c = np.dot(np.linalg.inv(intrinsic), points2_c)
		else:
			return img, tripoints3d, kpts, matches
		# print(points1_norm)
		# print(points2_norm)

		E = compute_essential_normalized(points1_norm, points2_norm)

		P1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
		P2s = compute_P_from_essential(E)

		ind = -1
		for i, P2 in enumerate(P2s):
			d1 = reconstruct_one_point(points1_norm[:, 0], points2_norm[:, 0], P1, P2)

			P2_homogenous = np.linalg.inv(np.vstack([P2, [0,0,0,1]]))

			d2 = np.dot(P2_homogenous[:3, :4], d1)

			if d1[2] > 0 and d2[2] > 0:
				ind = i

		P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
		# tripoints3d = triangulation(points1_norm, points2_norm, P1, P2)
		tripoints3d = triangulation(points1_norm_c, points2_norm_c, P1, P2)
		# print('tri',tripoints3d.shape)

	else:
		print("Wrong dimension of array")
		pass

	return img, tripoints3d, kpts, matches
