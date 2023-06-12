import numpy as np
import cv2
import argparse
import os
import yaml
import csv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

qs = {'b_2_fl': np.array([0.074732, -0.794, -0.10595, 0.59393]),
      'fl_2_f': np.array([-0.117199, -0.575476, -0.0686302, 0.806462]),
      'fr_2_f': np.array([-0.0806252, 0.607127, 0.0356452, 0.789699]),
      'f_2_base': np.array([-0.5070558775462676, 0.47615311808704197, -0.4812773544166568, 0.5334272708696808])}

rotation_b_fl = Rotation.from_quat(qs['b_2_fl'])
rotation_fl_f = Rotation.from_quat(qs['fl_2_f'])
rotation_fr_f = Rotation.from_quat(qs['fr_2_f'])
rotation_f_base = Rotation.from_quat(qs['f_2_base'])

def pinhole(args):
    if args.seq in ['seq1', 'seq2', 'seq3']:
        seq_path = os.path.join('./ITRI_dataset', args.seq)
    elif args.seq in ['test1', 'test2']:
        seq_path = os.path.join('./ITRI_DLC', args.seq)

    # file of all time stamp
    time_stamp_path = os.path.join(seq_path, 'all_timestamp.txt')
    time_stamp_file = open(time_stamp_path, 'r')
    lines = time_stamp_file.readlines()
    # files of camera info
    with open('ITRI_dataset\camera_info\lucid_cameras_x00\gige_100_b_hdr_camera_info.yaml', 'r') as stream:
        camera_b = yaml.safe_load(stream)
    with open('ITRI_dataset\camera_info\lucid_cameras_x00\gige_100_f_hdr_camera_info.yaml', 'r') as stream:
        camera_f = yaml.safe_load(stream)
    with open('ITRI_dataset\camera_info\lucid_cameras_x00\gige_100_fl_hdr_camera_info.yaml', 'r') as stream:
        camera_fl = yaml.safe_load(stream)
    with open('ITRI_dataset\camera_info\lucid_cameras_x00\gige_100_fr_hdr_camera_info.yaml', 'r') as stream:
        camera_fr = yaml.safe_load(stream)
    
    cameras = {'b': camera_b, 'f': camera_f, 'fl': camera_fl, 'fr': camera_fr}

    for i in range(len(lines)):
        line = lines[i] 
        print('\r', end='')
        print(f'Running pinhole {i+1}/{len(lines)}', end=' ')
        floder_path = os.path.join(seq_path, 'dataset', line[:-1]) # remove '\n'
        # read image
        img = cv2.imread(os.path.join(floder_path, 'raw_image.jpg')).astype('uint8')
        h, w, c = img.shape
        # read camera
        camera_f = open(os.path.join(floder_path, 'camera.csv'), 'r')
        camera = csv.reader(camera_f)
        for ca in camera:
            camera = str(ca[0])
            camera_name = camera.split('_')[4] # fl, f, b, fr
            camera = cameras[camera_name]
        p_matrix = np.array(camera['projection_matrix']['data']).reshape((3,4))
        # print(p_matrix)
        # read corners
        corners = np.load(os.path.join(floder_path, 'corners_(y,x).npy'))
        # corners = np.load(os.path.join(floder_path, 'filtered_corners_(y,x).npy'))
        # print(corners.shape)
        
        x_row = corners[:, 1]
        y_row = corners[:, 0]

        # Normalized 3D point (x, y, 1) at camera coordination
        dstx = (x_row-p_matrix[0,2]) / p_matrix[0,0]
        dsty = (y_row-p_matrix[1,2]) / p_matrix[1,1]
        dstz = np.ones((len(dstx), 1))
        
        dstx = dstx.reshape(-1,1)
        dsty = dsty.reshape(-1,1)
        dstz = dstz.reshape(-1,1)
        xyz_in_camera = np.concatenate((dstx, dsty, dstz), axis=1)
        # print(xyz_in_camera)

        # Convert to base_link (only rotate), for base_link: x=front, y=left, z=top
        # origin of base_link = origin of front camera
        origin = np.array([0,0,0])
        if camera_name == 'f':
            xyz_in_base = rotation_f_base.apply(xyz_in_camera)
        elif camera_name == 'fr':
            xyz_in_base = rotation_fr_f.apply(xyz_in_camera)
            origin = origin + np.array([0.559084, 0.0287952, -0.0950537])
            xyz_in_base = rotation_f_base.apply(xyz_in_base)
            origin = rotation_f_base.apply(origin)
        elif camera_name == 'fl':
            xyz_in_base = rotation_fl_f.apply(xyz_in_camera)
            origin = origin + np.array([-0.564697, 0.0402756, -0.028059])
            xyz_in_base = rotation_f_base.apply(xyz_in_base)
            origin = rotation_f_base.apply(origin)
        elif camera_name == 'b':
            xyz_in_base = rotation_b_fl.apply(xyz_in_camera)
            origin = origin + np.array([-1.2446, 0.21365, -0.91917])
            xyz_in_base = rotation_fl_f.apply(xyz_in_base)
            origin = rotation_fl_f.apply(origin) + np.array([-0.564697, 0.0402756, -0.028059])
            xyz_in_base = rotation_f_base.apply(xyz_in_base)
            origin = rotation_f_base.apply(origin)
        # print(camera_name, origin)

        # Find the cross point with ground
        xyz_in_base = xyz_in_base * -1.63 / xyz_in_base[:, 2].reshape((-1,1))
        # Shift origin
        xyz_in_base = xyz_in_base + origin
        # print(xyz_in_base)
        near_point = []
        min_d = 15
        for i in range(len(xyz_in_base)):
            dist = xyz_in_base[i,0]**2 + xyz_in_base[i,1]**2
            if dist < min_d**2:
                near_point.append(xyz_in_base[i])
        near_point = np.array(near_point)
        print('Point number:', near_point.shape, end = '')
        # if len(near_point) != 0:
        #     plt.scatter(near_point[:,0], near_point[:,1], s=2)
        #     plt.xlim([-min_d, min_d])
        #     plt.ylim([-min_d, min_d])
        #     plt.savefig(os.path.join(floder_path, 'plot.png'))
        #     plt.close()
        
        file_path = os.path.join(floder_path, 'output.csv')
        np.savetxt(file_path, near_point, delimiter=',', fmt='%s')
    print('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Read dataset & marker')
    parser.add_argument('--seq', default = 'seq1', type=str, help = 'Which sequence do you want to read')
    args = parser.parse_args()
    pinhole(args)