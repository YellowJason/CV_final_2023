import numpy as np
import math
from scipy.spatial.transform import Rotation

qs = {'b_2_fl': np.array([0.074732, -0.794, -0.10595, 0.59393]),
      'fl_2_r': np.array([-0.117199, -0.575476, -0.0686302, 0.806462]),
      'fr_2_r': np.array([-0.0806252, 0.607127, 0.0356452, 0.789699]),
      'f_2_base': np.array([-0.5070558775462676, 0.47615311808704197, -0.4812773544166568, 0.5334272708696808])}

def quaternion_to_rotation_matrix(q):  # x, y ,z ,w
    rot_matrix = np.array(
        [[1.0 - 2 * (q[1] * q[1] + q[2] * q[2]), 2 * (q[0] * q[1] - q[3] * q[2]), 2 * (q[3] * q[1] + q[0] * q[2])],
         [2 * (q[0] * q[1] + q[3] * q[2]), 1.0 - 2 * (q[0] * q[0] + q[2] * q[2]), 2 * (q[1] * q[2] - q[3] * q[0])],
         [2 * (q[0] * q[2] - q[3] * q[1]), 2 * (q[1] * q[2] + q[3] * q[0]), 1.0 - 2 * (q[0] * q[0] + q[1] * q[1])]],
        dtype=q.dtype)
    return rot_matrix

def quart_to_rpy(q):
    [x, y, z, w] = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw

# 以x,y,z為軸的轉動
(r, p, y) = quart_to_rpy(qs['fr_2_r'])
print('fr', r,p,y)
(r, p, y) = quart_to_rpy(qs['fl_2_r'])
print('fl', r,p,y)
(r, p, y) = quart_to_rpy(qs['b_2_fl'])
print('b ', r,p,y)
(r, p, y) = quart_to_rpy(qs['f_2_base'])
print('base ', r,p,y)

p1 = np.array([5, 0, 0]).T
p2 = np.array([0, 0, 5]).T
r_m = quaternion_to_rotation_matrix(qs['fr_2_r'])
print(np.dot(r_m, p1), np.dot(r_m, p2))

rotation = Rotation.from_quat(qs['f_2_base'])
print(rotation.apply(p1), rotation.apply(p2))