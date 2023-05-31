import numpy as np
import cv2
import argparse
import os
import yaml
import csv

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
        Intrinsics = np.array(camera['camera_matrix']['data']).reshape((3,3))
        trans = np.dot(Intrinsics, p_matrix)
        print(trans)
        trans_inv = np.linalg.inv(trans[:,[0,1,2]])
        # print(trans_inv)
        
        out_img = np.zeros((h,w,c))
        xc, yc = np.meshgrid(np.arange(0, w, 1), np.arange(0, h, 1), sparse = False)
        xrow = xc.reshape((1, w*h))
        yrow = yc.reshape((1, w*h))
        onerow = np.ones((1, w*h))
        M = np.concatenate((xrow, yrow, onerow), axis = 0)
        
        Mbar = np.dot(trans_inv, M)
        Mbar = np.divide(Mbar, Mbar[-1,:])
        dsty = np.round( Mbar[1,:].reshape((h, w)) ).astype(int)
        dstx = np.round( Mbar[0,:].reshape((h, w)) ).astype(int) 

        dstx = dstx - np.min(dstx)
        dstx = dsty - np.min(dsty)
        print(dstx)
        h_mask = (0<=dsty)*(dsty<h)
        w_mask = (0<=dstx)*(dstx<w)
        mask   = h_mask*w_mask
        out_img[dsty[mask], dstx[mask]] = img[yc[mask], xc[mask]]

        cv2.imwrite(os.path.join(floder_path, 'image_top_view.jpg'), out_img)

if __name__ == '__main__':
    main()