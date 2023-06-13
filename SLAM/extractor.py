import cv2
import numpy as np
# from gms_matcher import GmsMatcher
import time
from display import Display
display = Display()
class Extractor(object):
	def __init__(self):
		self.orb = cv2.orb = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)
		self.orb.setFastThreshold(0)
		self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
		self.last = None
		self.last_img = None
		# self.gms = GmsMatcher(self.orb, self.bf)

	def extract_keypoints(self, img, bb, camera_mask, timestamp):
		# detection
		# print("image:",img.shape)
		pts_list = np.empty((0, 2), dtype=int)
		for i in range(bb.shape[0]):
			current_bb = bb[i].astype(int)
			crop_img = img[current_bb[1]:current_bb[3], current_bb[0]:current_bb[2]]
			# print(current_bb[1],current_bb[3], current_bb[0],current_bb[2])
			# print(crop_img.shape)
			# cv2.imwrite('output.jpg', img)
			# cv2.imwrite("crop.jpg", crop_img)
			# time.sleep(30)
			if len(crop_img.shape) > 2:
				pts = cv2.goodFeaturesToTrack(image=np.mean(crop_img, axis=2).astype(np.uint8), maxCorners=450,
					qualityLevel=0.06, minDistance=5)

			else:
				pts = cv2.goodFeaturesToTrack(image=np.array(crop_img).astype(np.uint8), maxCorners=450,
					qualityLevel=0.06, minDistance=5)
			
			if pts is not None:
				pts = np.squeeze(np.array(pts), axis=1)
				pts[:, 0] += current_bb[0]
				pts[:, 1] += current_bb[1]
				# Iterate over each point and check if it lies within the white area
				points_within_mask = []
				# print(camera_mask)
				for point in pts:
					x, y = point
					if camera_mask[int(y), int(x)] == 0:  # White pixel (255) indicates the area of interest
						points_within_mask.append(point)

				# Convert the filtered points list back to a NumPy array
				filtered_points = np.array(points_within_mask).astype(int)
			else:
				filtered_points = np.empty((0, 2), dtype=int)
			# display.display_points2d(img, pts, None)
			# Convert the coordinates to integers
			# center_x = filtered_points[:, 0].astype(int)
			# center_y = filtered_points[:, 1].astype(int)

			# # Iterate over each point and draw a circle
			# for x, y in zip(center_x, center_y):
			# 	cv2.circle(img, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
			# time.sleep(30)
			# cv2.imwrite("corner.jpg", img)
			# exit()
			pts_list = np.concatenate((pts_list, filtered_points), axis=0)
		# extraction
		# print(len(pts_list))
		center_x = pts_list[:, 0]
		center_y = pts_list[:, 1]
		# for x, y in zip(center_x, center_y):
		# 	cv2.circle(img, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
		# cv2.imwrite("corner.jpg", img)
		# kpts = [cv2.KeyPoint(p[0][0],p[0][1], _size=30) for p in pts_list]
		
		pts_new = np.load("../ITRI_dataset/seq1/dataset/%s/filtered_corners_(y,x).npy" % timestamp)

		if (pts_new.shape[0] > 10):
			print(pts_list.shape)
			print(pts_new.shape)
			print(pts_list[:5, :])
			print(pts_new[:5, :])
			pts_new[:, [0, 1]] = pts_new[:, [1, 0]]
			pts_list = pts_new
		kpts = [cv2.KeyPoint(float(p[0]), float(p[1]), size=30) for p in pts_list]
		kpts, des = self.orb.compute(img, kpts)
		# print(len(kpts))

		# matching
		ret = []
		if self.last is not None:
			matches = self.bf.knnMatch(des, self.last["des"], k=2)
			# matches, kpt_last, kpt_now = self.gms.compute_matches(self.last_img, img)
			# print(matches)
			match_new = []
			for m, n in matches:
				if m.distance < 0.55* n.distance:
					# print("Maximum Distance: ", m.distance)
					if m.distance < 30:
						kpt1_match = kpts[m.queryIdx]
						kpt2_match = self.last["kpts"][m.trainIdx]
						ret.append((kpt1_match, kpt2_match))
						match_new.append((m, n))
			coords1_match_pts = np.asarray([kpts[m.queryIdx].pt for m,n in match_new])
			coords2_match_pts = np.asarray([self.last["kpts"][m.trainIdx].pt for m,n in match_new])
			# coords1_match_pts = np.asarray([m for m,n in ret])
			# coords2_match_pts = np.asarray([n for m,n in ret])
			# for m in matches:
			# 	try:
			# 		kpt1_match = kpt_now[m.trainIdx]
			# 		kpt2_match = kpt_last[m.queryIdx]
			# 	except IndexError:
			# 		print(m.queryIdx)
			# 		print(m.trainIdx)
			# 		print(len(kpt_now))
			# 		print(len(kpt_last))
			# 		exit()
			# 	ret.append((kpt1_match, kpt2_match))

			# coords1_match_pts = np.asarray([kpt_now[m.trainIdx].pt for m in matches])
			# coords2_match_pts = np.asarray([kpt_last[m.queryIdx].pt for m in matches])
			# find transformation between two matched points
			retval, mask = cv2.findHomography(coords1_match_pts, coords2_match_pts, cv2.RANSAC, 10.0)
			mask = mask.ravel()
			# print(np.sort(mask))
			pts1 = coords1_match_pts[mask==1]
			pts2 = coords2_match_pts[mask==1]
			# if (timestamp == "1681710720_131437318"):
				
			# 	print(filtered_key)
			self.last = {"kpts":kpts, "des":des}
			self.last_img = img

			# pts1 : current
			# pts2 : last
			# kpts : current key
			#
			return pts1.T, pts2.T, kpts, ret
		
		else:
			self.last = {"kpts":kpts, "des":des}
			self.last_img = img
			return np.array([0]),np.array([0]), 0, 0
		
