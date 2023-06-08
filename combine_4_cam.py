import numpy as np
import cv2
import argparse
import os
import yaml
import csv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def main():
    parser = argparse.ArgumentParser(description = 'Read dataset & marker')
    parser.add_argument('--seq', default = 'seq1', type=str, help = 'Which sequence do you want to read')
    args = parser.parse_args()

    if args.seq in ['seq1', 'seq2', 'seq3']:
        seq_path = os.path.join('./ITRI_dataset', args.seq)
    elif args.seq in ['test1', 'test2']:
        seq_path = os.path.join('./ITRI_DLC', args.seq)

    # file of all time stamp
    time_stamp_path = os.path.join(seq_path, 'all_timestamp.txt')
    time_stamp_file = open(time_stamp_path, 'r')
    lines = time_stamp_file.readlines()

    # Combine point cloud of 4 cameras at near time stamp
    for i in range(0, len(lines), 4):
        try:
            line_0 = lines[i] 
            line_1 = lines[i+1] 
            line_2 = lines[i+2] 
            line_3 = lines[i+3]
        except:
            stop = i
            break
        print('\r', end='')
        print(f'Processing {i+1}/{len(lines)}', end=' ')
        floder_path_0 = os.path.join(seq_path, 'dataset', line_0[:-1])
        floder_path_1 = os.path.join(seq_path, 'dataset', line_1[:-1])
        floder_path_2 = os.path.join(seq_path, 'dataset', line_2[:-1])
        floder_path_3 = os.path.join(seq_path, 'dataset', line_3[:-1])
        # read point clouds
        f = open(os.path.join(floder_path_0, 'output.csv'), 'r')
        xyz_0 = np.array(list(csv.reader(f))).reshape(-1,3)
        f = open(os.path.join(floder_path_1, 'output.csv'), 'r')
        xyz_1 = np.array(list(csv.reader(f))).reshape(-1,3)
        f = open(os.path.join(floder_path_2, 'output.csv'), 'r')
        xyz_2 = np.array(list(csv.reader(f))).reshape(-1,3)
        f = open(os.path.join(floder_path_3, 'output.csv'), 'r')
        xyz_3 = np.array(list(csv.reader(f))).reshape(-1,3)
        xyz = np.concatenate((xyz_0,xyz_1,xyz_2,xyz_3), axis=0)
        # print(xyz_0.shape, xyz_1.shape, xyz_2.shape, xyz_3.shape, xyz.shape)
        
        min_d = 15
        # plt.scatter(xyz[:,0], xyz[:,1], s=2)
        # plt.xlim([-min_d, min_d])
        # plt.ylim([-min_d, min_d])
        # plt.savefig(os.path.join(floder_path_0, 'plot_merge.png'))
        # plt.savefig(os.path.join(floder_path_1, 'plot_merge.png'))
        # plt.savefig(os.path.join(floder_path_2, 'plot_merge.png'))
        # plt.savefig(os.path.join(floder_path_3, 'plot_merge.png'))
        # plt.close()
        
        np.savetxt(os.path.join(floder_path_0, 'output_merge.csv'), xyz, delimiter=',', fmt='%s')
        np.savetxt(os.path.join(floder_path_1, 'output_merge.csv'), xyz, delimiter=',', fmt='%s')
        np.savetxt(os.path.join(floder_path_2, 'output_merge.csv'), xyz, delimiter=',', fmt='%s')
        np.savetxt(os.path.join(floder_path_3, 'output_merge.csv'), xyz, delimiter=',', fmt='%s')

    for i in range(stop, len(lines)):
        line_0 = lines[i]
        line_1 = lines[i-1] 
        line_2 = lines[i-2] 
        line_3 = lines[i-3]
        print('\r', end='')
        print(f'Processing {i+1}/{len(lines)}', end=' ')
        floder_path_0 = os.path.join(seq_path, 'dataset', line_0[:-1])
        floder_path_1 = os.path.join(seq_path, 'dataset', line_1[:-1])
        floder_path_2 = os.path.join(seq_path, 'dataset', line_2[:-1])
        floder_path_3 = os.path.join(seq_path, 'dataset', line_3[:-1])
        # read point clouds
        f = open(os.path.join(floder_path_0, 'output.csv'), 'r')
        xyz_0 = np.array(list(csv.reader(f))).reshape(-1,3)
        f = open(os.path.join(floder_path_1, 'output.csv'), 'r')
        xyz_1 = np.array(list(csv.reader(f))).reshape(-1,3)
        f = open(os.path.join(floder_path_2, 'output.csv'), 'r')
        xyz_2 = np.array(list(csv.reader(f))).reshape(-1,3)
        f = open(os.path.join(floder_path_3, 'output.csv'), 'r')
        xyz_3 = np.array(list(csv.reader(f))).reshape(-1,3)
        xyz = np.concatenate((xyz_0,xyz_1,xyz_2,xyz_3), axis=0)
        # print(xyz_0.shape, xyz_1.shape, xyz_2.shape, xyz_3.shape, xyz.shape)
        
        min_d = 15
        # plt.scatter(xyz[:,0], xyz[:,1], s=2)
        # plt.xlim([-min_d, min_d])
        # plt.ylim([-min_d, min_d])
        # plt.savefig(os.path.join(floder_path_0, 'plot_merge.png'))
        # plt.savefig(os.path.join(floder_path_1, 'plot_merge.png'))
        # plt.savefig(os.path.join(floder_path_2, 'plot_merge.png'))
        # plt.savefig(os.path.join(floder_path_3, 'plot_merge.png'))
        # plt.close()
        
        np.savetxt(os.path.join(floder_path_0, 'output_merge.csv'), xyz, delimiter=',', fmt='%s')
if __name__ == '__main__':
    main()