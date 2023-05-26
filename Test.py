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
    # process img at each time stamp
    for i in range(len(lines)):
        line = lines[i]
        print('\r', end='')
        print(f'Reading {i+1}/{len(lines)}', end='')
        # print(line[:-1])
        floder_path = os.path.join(seq_path, 'dataset', line[:-1]) # remove '\n'
        # read image
        img = cv2.imread(os.path.join(floder_path, 'raw_image.jpg')).astype(np.float64)
        # print(img.shape)
        # read marker
        marker_f = open(os.path.join(floder_path, 'detect_road_marker.csv'), 'r')
        marker = csv.reader(marker_f)
        for row in marker:
            x1, y1 = int(float(row[0])), int(float(row[1]))
            x2, y2 = int(float(row[2])), int(float(row[3]))
            class_id, prob = int(row[4]), float(row[5])
            # print(x1, y1, x2, y2, class_id, prob)
            class_color =  [(255,0,0), (255,255,0), (0,255,0), (0,255,255), (0,0,255)]
            if prob > 0.2:
                cv2.rectangle(img, (x1, y1), (x2, y2), class_color[class_id], 2)
        
        cv2.imwrite(os.path.join(floder_path, 'image_with_marker.jpg'), img)

if __name__ == '__main__':
    main()