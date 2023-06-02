import cv2
import numpy as np


class Extractor(object):
	def __init__(self):
		self.orb = cv2.orb = cv2.ORB_create(nfeatures=1, scoreType=cv2.ORB_FAST_SCORE)
		self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
		self.last = None
		self.last_corner = None

	def extract_keypoints(self, img_path):
		img = cv2.imread(f"{img_path}/raw_image.jpg")
		# detection
		# print("image:",img)
		if len(img.shape) > 2:
			pts = cv2.goodFeaturesToTrack(image=np.mean(img, axis=2).astype(np.uint8), maxCorners=4500,
				qualityLevel=0.02, minDistance=3)

		else:
			pts = cv2.goodFeaturesToTrack(image=np.array(img).astype(np.uint8), maxCorners=4500,
				qualityLevel=0.02, minDistance=3)
		
		# points from find_corners
		pts_corner = np.load(f"{img_path}/corners_(y,x).npy").astype(float) # (y,x)
		# print(pts_corner.shape)

		# extraction
		kpts = [cv2.KeyPoint(p[0][0],p[0][1], _size=30) for p in pts]
		kpts_corner = [cv2.KeyPoint(p[1],p[0], _size=30) for p in pts_corner]

		kpts, des = self.orb.compute(img, kpts)
		kpts_c, des_c = self.orb.compute(img, kpts_corner)
		# print('llll', len(kpts_corner))
		# print(kpts)
		# matching
		ret = []
		ret_c = []
		coords1_match_pts = []
		coords2_match_pts = []
		coords1_match_pts_c = []
		coords2_match_pts_c = []
		No_point = False
		if self.last is not None:
			matches = self.bf.knnMatch(des, self.last["des"], k=2)
			try:
				matches_c = self.bf.knnMatch(des_c, self.last_corner["des"], k=2)
			except:
				matches_c = []
				No_point = True
			
			# match feature points
			for m, n in matches:
				if m.distance < 0.55* n.distance:
					if m.distance < 64:
						kpt1_match = kpts[m.queryIdx]
						kpt2_match = self.last["kpts"][m.trainIdx]
						ret.append((kpt1_match, kpt2_match))
						coords1_match_pts.append(kpt1_match.pt)
						coords2_match_pts.append(kpt2_match.pt)
			coords1_match_pts = np.array(coords1_match_pts)
			coords2_match_pts = np.array(coords2_match_pts)
			# print('aaa', coords1_match_pts.shape, np.array(ret).shape)
			# match corner points
			# print(np.array(matches_c).shape)
			try:
				for m, n in matches_c:
					if m.distance < 0.55* n.distance:
						if m.distance < 64:
							kpt1_match = kpts_corner[m.queryIdx]
							kpt2_match = self.last_corner["kpts"][m.trainIdx]
							ret_c.append((kpt1_match, kpt2_match))
							coords1_match_pts_c.append(kpt1_match.pt)
							coords2_match_pts_c.append(kpt2_match.pt)
			except:
				if not No_point:
					min_d = float('inf')
					min_match = matches_c[0][0]
					for m in matches_c[:][0]:
						if m.distance < min_d:
							min_d = m.distance
							min_match = m
					kpt1_match = kpts_corner[min_match.queryIdx]
					kpt2_match = self.last_corner["kpts"][min_match.trainIdx]
					ret_c.append((kpt1_match, kpt2_match))
					coords1_match_pts_c.append(kpt1_match.pt)
					coords2_match_pts_c.append(kpt2_match.pt)
			coords1_match_pts_c = np.array(coords1_match_pts_c)
			coords2_match_pts_c = np.array(coords2_match_pts_c)
			
			# find transformation between two matched points
			# retval, mask = cv2.findHomography(coords1_match_pts, coords2_match_pts, cv2.RANSAC, 100.0)
			# mask = mask.ravel()

			# pts1 = coords1_match_pts[mask==1]
			# pts2 = coords2_match_pts[mask==1]

			pts1 = coords1_match_pts
			pts2 = coords2_match_pts

			self.last = {"kpts":kpts, "des":des}
			self.last_corner = {"kpts":kpts_c, "des":des_c}

			print('size', pts1.shape, pts2.shape, coords1_match_pts_c.shape, coords2_match_pts_c.shape)

			return pts1.T, pts2.T, kpts, ret, coords1_match_pts_c.T, coords2_match_pts_c.T
		
		else:
			self.last = {"kpts":kpts, "des":des}
			self.last_corner = {"kpts":kpts_c, "des":des_c}
			return np.array([0]),np.array([0]), 0, 0, np.array([0]),np.array([0])
		
