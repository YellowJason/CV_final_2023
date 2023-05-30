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

    # file of diff cam timestamp
    f_timestamp_file = open(os.path.join(seq_path, 'f_timestamp.txt'), 'r')
    lines_f = f_timestamp_file.readlines()
    b_timestamp_file = open(os.path.join(seq_path, 'b_timestamp.txt'), 'r')
    lines_b = b_timestamp_file.readlines()
    fr_timestamp_file = open(os.path.join(seq_path, 'fr_timestamp.txt'), 'r')
    lines_fr = fr_timestamp_file.readlines()
    fl_timestamp_file = open(os.path.join(seq_path, 'fl_timestamp.txt'), 'r')
    lines_fl = fl_timestamp_file.readlines()

    # Make video
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    videowriter_f_filtered = cv2.VideoWriter(f"{seq_path}/marker_f_filtered.avi", fourcc, 24, (1440, 928))
    videowriter_b_filtered = cv2.VideoWriter(f"{seq_path}/marker_b_filtered.avi", fourcc, 24, (1440, 928))
    videowriter_fr_filtered = cv2.VideoWriter(f"{seq_path}/marker_fr_filtered.avi", fourcc, 24, (1440, 928))
    videowriter_fl_filtered = cv2.VideoWriter(f"{seq_path}/marker_fl_filtered.avi", fourcc, 24, (1440, 928))
 
    
    for i in range(1, len(lines_f) - 1, 1):
        front_frame_now = os.path.join(seq_path, 'dataset', lines_f[i][:-1])
        front_frame_former1 = os.path.join(seq_path, 'dataset', lines_f[i - 1][:-1])
        front_frame_later1 = os.path.join(seq_path, 'dataset', lines_f[i + 1][:-1])
        print('\r', end='')
        print(f'f_Processing {i+1}/{len(lines_f)}', end='')

        corners_n = np.load(os.path.join(front_frame_now, 'corners.npy'))
        corners_f1 = np.load(os.path.join(front_frame_former1, 'corners.npy'))
        corners_l1 = np.load(os.path.join(front_frame_later1, 'corners.npy'))

        out = modify_matrix(corners_f1, corners_n)
        out = modify_matrix(corners_l1, out)

        img = cv2.imread(os.path.join(front_frame_now, 'raw_image.jpg')).astype('uint8')
        true_coordinates = np.argwhere(out)
        true_coordinates_tuples = [tuple(coordinate) for coordinate in true_coordinates]
        for coordinate in true_coordinates_tuples:
            swap = (coordinate[1], coordinate[0])
            cv2.circle(img, swap, 2, (0, 0, 255), -1)
        
        videowriter_f_filtered.write(img)

    for i in range(1, len(lines_b) - 1, 1):
        front_frame_now = os.path.join(seq_path, 'dataset', lines_b[i][:-1])
        front_frame_former1 = os.path.join(seq_path, 'dataset', lines_b[i - 1][:-1])
        front_frame_later1 = os.path.join(seq_path, 'dataset', lines_b[i + 1][:-1])
        print('\r', end='')
        print(f'b_Processing {i+1}/{len(lines_b)}', end='')

        corners_n = np.load(os.path.join(front_frame_now, 'corners.npy'))
        corners_f1 = np.load(os.path.join(front_frame_former1, 'corners.npy'))
        corners_l1 = np.load(os.path.join(front_frame_later1, 'corners.npy'))

        out = modify_matrix(corners_f1, corners_n)
        out = modify_matrix(corners_l1, out)

        img = cv2.imread(os.path.join(front_frame_now, 'raw_image.jpg')).astype('uint8')
        true_coordinates = np.argwhere(out)
        true_coordinates_tuples = [tuple(coordinate) for coordinate in true_coordinates]
        for coordinate in true_coordinates_tuples:
            swap = (coordinate[1], coordinate[0])
            cv2.circle(img, swap, 2, (0, 0, 255), -1)
        
        videowriter_b_filtered.write(img)

    for i in range(1, len(lines_fr) - 1, 1):
        front_frame_now = os.path.join(seq_path, 'dataset', lines_fr[i][:-1])
        front_frame_former1 = os.path.join(seq_path, 'dataset', lines_fr[i - 1][:-1])
        front_frame_later1 = os.path.join(seq_path, 'dataset', lines_fr[i + 1][:-1])
        print('\r', end='')
        print(f'frProcessing {i+1}/{len(lines_fr)}', end='')

        corners_n = np.load(os.path.join(front_frame_now, 'corners.npy'))
        corners_f1 = np.load(os.path.join(front_frame_former1, 'corners.npy'))
        corners_l1 = np.load(os.path.join(front_frame_later1, 'corners.npy'))

        out = modify_matrix(corners_f1, corners_n)
        out = modify_matrix(corners_l1, out)

        img = cv2.imread(os.path.join(front_frame_now, 'raw_image.jpg')).astype('uint8')
        true_coordinates = np.argwhere(out)
        true_coordinates_tuples = [tuple(coordinate) for coordinate in true_coordinates]
        for coordinate in true_coordinates_tuples:
            swap = (coordinate[1], coordinate[0])
            cv2.circle(img, swap, 2, (0, 0, 255), -1)
        
        videowriter_fr_filtered.write(img)

    for i in range(1, len(lines_fl) - 1, 1):
        front_frame_now = os.path.join(seq_path, 'dataset', lines_fl[i][:-1])
        front_frame_former1 = os.path.join(seq_path, 'dataset', lines_fl[i - 1][:-1])
        front_frame_later1 = os.path.join(seq_path, 'dataset', lines_fl[i + 1][:-1])
        print('\r', end='')
        print(f'fl_Processing {i+1}/{len(lines_fl)}', end='')

        corners_n = np.load(os.path.join(front_frame_now, 'corners.npy'))
        corners_f1 = np.load(os.path.join(front_frame_former1, 'corners.npy'))
        corners_l1 = np.load(os.path.join(front_frame_later1, 'corners.npy'))

        out = modify_matrix(corners_f1, corners_n)
        out = modify_matrix(corners_l1, out)

        img = cv2.imread(os.path.join(front_frame_now, 'raw_image.jpg')).astype('uint8')
        true_coordinates = np.argwhere(out)
        true_coordinates_tuples = [tuple(coordinate) for coordinate in true_coordinates]
        for coordinate in true_coordinates_tuples:
            swap = (coordinate[1], coordinate[0])
            cv2.circle(img, swap, 2, (0, 0, 255), -1)
        
        videowriter_fl_filtered.write(img)
        

    videowriter_f_filtered.release()
    videowriter_b_filtered.release()
    videowriter_fr_filtered.release()
    videowriter_fl_filtered.release()

def modify_matrix(A, B):
    rows, cols = A.shape
    result = np.copy(B)
    indices = np.argwhere(B)  # 找到B矩陣中所有為True的位置
    radius = 20
    for idx in indices:
        i, j = idx[0], idx[1]
        top_i = max(i-radius, 0)
        left_j = max(j-radius, 0)
        bot_i = min(rows, i+radius+1)
        right_j = min(cols, j+radius+1)
        neighborhood = A[top_i:bot_i, left_j:right_j]  # 取得中心41*41方格

        if np.any(neighborhood):
            result[i, j] = True
        else:
            result[i, j] = False
    
    return result

if __name__ == '__main__':
    main()