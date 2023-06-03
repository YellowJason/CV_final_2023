import numpy as np
import cv2
import argparse
import os
import yaml
import csv
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description = 'Read dataset & marker')
    parser.add_argument('--seq', default = 'seq1', type=str, help = 'Which sequence do you want to read')
    args = parser.parse_args()

    seq_path = os.path.join('./ITRI_dataset', args.seq)

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
        print(f'Processing {i+1}/{len(lines)}', end='')
        floder_path = os.path.join(seq_path, 'dataset', line[:-1]) # remove '\n'
        # read image
        img = cv2.imread(os.path.join(floder_path, 'raw_image.jpg')).astype('uint8')
        h, w, c = img.shape
        # read camera
        camera_f = open(os.path.join(floder_path, 'camera.csv'), 'r')
        camera = csv.reader(camera_f)
        for ca in camera:
            camera = str(ca[0])
            camera = camera.split('_')[4] # fl, f, b, fr
            camera = cameras[camera]
        p_matrix = np.array(camera['projection_matrix']['data']).reshape((3,4))
        # print(p_matrix)
        # read corners
        corners = np.load(os.path.join(floder_path, 'corners_(y,x).npy'))
        # print(corners.shape)
        
        x_row = corners[:, 1]
        y_row = corners[:, 0]

        dsty = (1.63*p_matrix[1,1]) / (y_row-p_matrix[1,2])
        dstx = (x_row-p_matrix[0,2]) * (dsty/p_matrix[0,0])

        # dstx = np.round(dstx).astype(int)
        # dsty = np.round(dsty).astype(int)
        
        h_mask = (0<=dsty)*(dsty<=25)
        w_mask = (-25<=dstx)*(dstx<=25)
        mask   = h_mask*w_mask

        plt.scatter(dstx, dsty, s=3)
        plt.savefig(os.path.join(floder_path, 'plot.png'))
        plt.close()

if __name__ == '__main__':
    main()