import numpy as np
import cv2
import argparse
import os
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

    # file of diff cam timestamp
    f_timestamp_file = open(os.path.join(seq_path, 'f_timestamp.txt'), 'w')
    b_timestamp_file = open(os.path.join(seq_path, 'b_timestamp.txt'), 'w')
    fr_timestamp_file = open(os.path.join(seq_path, 'fr_timestamp.txt'), 'w')
    fl_timestamp_file = open(os.path.join(seq_path, 'fl_timestamp.txt'), 'w')


    for i in range(len(lines)):
        line = lines[i] 
        print('\r', end='')
        print(f'Processing {i+1}/{len(lines)}', end='')
        folder_path = os.path.join(seq_path, 'dataset', line[:-1]) # remove '\n'

        # read camera
        camera_f = open(os.path.join(folder_path, 'camera.csv'), 'r')
        camera = csv.reader(camera_f)
        for c in camera:
            camera = str(c[0])
            camera = camera.split('_')[4] # fl, f, b, fr
            # print(camera)

        if camera == 'f':
            f_timestamp_file.write(f'{line[:-1]}\n')
        elif camera == 'b':
            b_timestamp_file.write(f'{line[:-1]}\n')
        elif camera == 'fr':
            fr_timestamp_file.write(f'{line[:-1]}\n')
        elif camera == 'fl':
            fl_timestamp_file.write(f'{line[:-1]}\n')


if __name__ == '__main__':
    main()