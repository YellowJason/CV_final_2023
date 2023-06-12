import numpy as np
import cv2
import argparse
import os
import yaml
import csv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def combine_2(args):

    if args.seq in ['seq1', 'seq2', 'seq3']:
        seq_path = os.path.join('./ITRI_dataset', args.seq)
    elif args.seq in ['test1', 'test2']:
        seq_path = os.path.join('./ITRI_DLC', args.seq)

    # file of all time stamp
    time_stamp_path = os.path.join(seq_path, 'all_timestamp.txt')
    time_stamp_file = open(time_stamp_path, 'r')
    lines = time_stamp_file.readlines()

    #read speed data
    import glob
    speed = {}
    speed_time = []
    speed_stamp_path = os.path.join(seq_path, 'other_data')
    for speed_stamp_path in glob.glob(os.path.join(speed_stamp_path, "*_raw_speed.csv")):
        speed_stamp_file = open(speed_stamp_path, 'r')
        filename = speed_stamp_path.split('/')[-1]
        filename = filename.split("\\")[-1]
        # print(filename)
        time = int(filename.split('_')[0]) + int(filename.split('_')[1]) * 10**(-9)
        speed[time] = float(list(csv.reader(speed_stamp_file))[0][0])
        speed_time.append(time)
    speed_time.sort()

    #read imu data
    ori = {}
    imu_time = []
    acc = {}
    imu_stamp_path = os.path.join(seq_path, 'other_data')
    for imu_stamp_path in glob.glob(os.path.join(imu_stamp_path, "*_raw_imu.csv")):
        imu_stamp_file = open(imu_stamp_path, 'r')
        filename = imu_stamp_path.split('/')[-1]
        filename = filename.split("\\")[-1]
        time = int(filename.split('_')[0]) + int(filename.split('_')[1]) * 10**(-9)
        l_1 = np.array(list(csv.reader(imu_stamp_file)))[1]
        acc[time] = float(l_1[0]), float(l_1[1]), float(l_1[2])
        imu_stamp_file_2 = open(imu_stamp_path, 'r')
        l_2 = np.array(list(csv.reader(imu_stamp_file_2)))[2]
        ori[time] = float(l_2[0]), float(l_2[1]), float(l_2[2])
        imu_time.append(time)
    imu_time.sort()

    stop = 0
    speed_index = 0
    imu_index = 0
    speed_t_0 = speed_time[0]
    speed_t = speed_time[-1]
    imu_t_0 = imu_time[0]
    imu_t = imu_time[-1]

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
        time_1 = int(line_0.split('_')[0]) + int(line_0.split('_')[1]) * 10**(-9)
        time_2 = int(line_1.split('_')[0]) + int(line_1.split('_')[1]) * 10**(-9)
        time_3 = int(line_2.split('_')[0]) + int(line_2.split('_')[1]) * 10**(-9)
        time_4 = int(line_3.split('_')[0]) + int(line_3.split('_')[1]) * 10**(-9)
        v_1 = [0, 0, 0]
        a_1 = [0, 0, 0]
        o_1 = [0, 0, 0]
        v_2 = [0, 0, 0]
        a_2 = [0, 0, 0]
        o_2 = [0, 0, 0]
        v_3 = [0, 0, 0]
        a_3 = [0, 0, 0]
        o_3 = [0, 0, 0]
        shift_1 = [0, 0, 0]
        shift_2 = [0, 0, 0]
        shift_3 = [0, 0, 0]
        if time_1 < speed_t_0 or time_1 < imu_t_0:
            pass
        elif time_4 > speed_t or time_4 > imu_t:
            pass
        else:
            while time_2 > speed_time[speed_index]:
                speed_index += 1
            while time_2 > imu_time[imu_index]:
                imu_index += 1
            s_1 = (((speed[speed_time[speed_index]]-speed[speed_time[speed_index-1]])/(speed_time[speed_index]-speed_time[speed_index-1]))*(time_2-speed_time[speed_index-1])+speed[speed_time[speed_index-1]])
            o_1[0] = (((ori[imu_time[imu_index]][0]-ori[imu_time[imu_index-1]][0])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_2-imu_time[imu_index-1])+ori[imu_time[imu_index-1]][0])
            o_1[1] = (((ori[imu_time[imu_index]][1]-ori[imu_time[imu_index-1]][1])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_2-imu_time[imu_index-1])+ori[imu_time[imu_index-1]][1])
            o_1[2] = (((ori[imu_time[imu_index]][2]-ori[imu_time[imu_index-1]][2])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_2-imu_time[imu_index-1])+ori[imu_time[imu_index-1]][2])
            o_1_unit = (o_1[0]**2+o_1[1]**2+o_1[2]**2)**(0.5)
            v_1 = [j * (s_1/o_1_unit) for j in o_1]
            a_1[0] = (((acc[imu_time[imu_index]][0]-acc[imu_time[imu_index-1]][0])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_2-imu_time[imu_index-1])+acc[imu_time[imu_index-1]][0])
            a_1[1] = (((acc[imu_time[imu_index]][1]-acc[imu_time[imu_index-1]][1])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_2-imu_time[imu_index-1])+acc[imu_time[imu_index-1]][1])
            a_1[2] = (((acc[imu_time[imu_index]][2]-acc[imu_time[imu_index-1]][2])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_2-imu_time[imu_index-1])+acc[imu_time[imu_index-1]][2])
            while time_3 > speed_time[speed_index]:
                speed_index += 1
            while time_3 > imu_time[imu_index]:
                imu_index += 1
            s_2 = (((speed[speed_time[speed_index]]-speed[speed_time[speed_index-1]])/(speed_time[speed_index]-speed_time[speed_index-1]))*(time_3-speed_time[speed_index-1])+speed[speed_time[speed_index-1]])
            o_2[0] = (((ori[imu_time[imu_index]][0]-ori[imu_time[imu_index-1]][0])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_3-imu_time[imu_index-1])+ori[imu_time[imu_index-1]][0])
            o_2[1] = (((ori[imu_time[imu_index]][1]-ori[imu_time[imu_index-1]][1])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_3-imu_time[imu_index-1])+ori[imu_time[imu_index-1]][1])
            o_2[2] = (((ori[imu_time[imu_index]][2]-ori[imu_time[imu_index-1]][2])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_3-imu_time[imu_index-1])+ori[imu_time[imu_index-1]][2])
            o_2_unit = (o_2[0]**2+o_2[1]**2+o_2[2]**2)**(0.5)
            v_2 = [j * (s_2/o_2_unit) for j in o_2]
            a_2[0] = (((acc[imu_time[imu_index]][0]-acc[imu_time[imu_index-1]][0])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_3-imu_time[imu_index-1])+acc[imu_time[imu_index-1]][0])
            a_2[1] = (((acc[imu_time[imu_index]][1]-acc[imu_time[imu_index-1]][1])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_3-imu_time[imu_index-1])+acc[imu_time[imu_index-1]][1])
            a_2[2] = (((acc[imu_time[imu_index]][2]-acc[imu_time[imu_index-1]][2])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_3-imu_time[imu_index-1])+acc[imu_time[imu_index-1]][2])
            while time_4 > speed_time[speed_index]:
                speed_index += 1
            while time_4 > imu_time[imu_index]:
                imu_index += 1
            s_3 = (((speed[speed_time[speed_index]]-speed[speed_time[speed_index-1]])/(speed_time[speed_index]-speed_time[speed_index-1]))*(time_4-speed_time[speed_index-1])+speed[speed_time[speed_index-1]])
            o_3[0] = (((ori[imu_time[imu_index]][0]-ori[imu_time[imu_index-1]][0])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_4-imu_time[imu_index-1])+ori[imu_time[imu_index-1]][0])
            o_3[1] = (((ori[imu_time[imu_index]][1]-ori[imu_time[imu_index-1]][1])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_4-imu_time[imu_index-1])+ori[imu_time[imu_index-1]][1])
            o_3[2] = (((ori[imu_time[imu_index]][2]-ori[imu_time[imu_index-1]][2])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_4-imu_time[imu_index-1])+ori[imu_time[imu_index-1]][2])
            o_3_unit = (o_3[0]**2+o_3[1]**2+o_3[2]**2)**(0.5)
            v_3 = [j * (s_3/o_3_unit) for j in o_3]
            a_3[0] = (((acc[imu_time[imu_index]][0]-acc[imu_time[imu_index-1]][0])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_4-imu_time[imu_index-1])+acc[imu_time[imu_index-1]][0])
            a_3[1] = (((acc[imu_time[imu_index]][1]-acc[imu_time[imu_index-1]][1])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_4-imu_time[imu_index-1])+acc[imu_time[imu_index-1]][1])
            a_3[2] = (((acc[imu_time[imu_index]][2]-acc[imu_time[imu_index-1]][2])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_4-imu_time[imu_index-1])+acc[imu_time[imu_index-1]][2])
        for k in range(3):
            shift_1[k] = v_1[k] * (time_2 - time_1) + 0.5 * a_1[k] * (time_2 - time_1)**2
            shift_2[k] = v_2[k] * (time_3 - time_1) + 0.5 * a_2[k] * (time_3 - time_1)**2
            shift_3[k] = v_3[k] * (time_4 - time_1) + 0.5 * a_3[k] * (time_4 - time_1)**2
        floder_path_0 = os.path.join(seq_path, 'dataset', line_0[:-1])
        floder_path_1 = os.path.join(seq_path, 'dataset', line_1[:-1])
        floder_path_2 = os.path.join(seq_path, 'dataset', line_2[:-1])
        floder_path_3 = os.path.join(seq_path, 'dataset', line_3[:-1])
        # read point clouds
        f = open(os.path.join(floder_path_0, 'output.csv'), 'r')
        xyz_0 = np.array(list(csv.reader(f))).reshape(-1,3)
        f = open(os.path.join(floder_path_1, 'output.csv'), 'r')
        xyz_1 = np.array(list(csv.reader(f)), dtype=np.float32).reshape(-1,3)
        for i in range(np.shape(xyz_1)[0]):
            xyz_1[i] -= shift_1
        f = open(os.path.join(floder_path_2, 'output.csv'), 'r')
        xyz_2 = np.array(list(csv.reader(f)), dtype=np.float32).reshape(-1,3)
        for i in range(np.shape(xyz_2)[0]):
            xyz_2[i] -= shift_2
        f = open(os.path.join(floder_path_3, 'output.csv'), 'r')
        xyz_3 = np.array(list(csv.reader(f)), dtype=np.float32).reshape(-1,3)
        for i in range(np.shape(xyz_3)[0]):
            xyz_3[i] -= shift_3
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
        line_3 = lines[i]
        line_2 = lines[i-1] 
        line_1 = lines[i-2] 
        line_0 = lines[i-3]
        print('\r', end='')
        print(f'Combine four cameras {i+1}/{len(lines)}', end=' ')
        time_1 = int(line_0.split('_')[0]) + int(line_0.split('_')[1]) * 10**(-9)
        time_2 = int(line_1.split('_')[0]) + int(line_1.split('_')[1]) * 10**(-9)
        time_3 = int(line_2.split('_')[0]) + int(line_2.split('_')[1]) * 10**(-9)
        time_4 = int(line_3.split('_')[0]) + int(line_3.split('_')[1]) * 10**(-9)
        v_1 = [0, 0, 0]
        a_1 = [0, 0, 0]
        o_1 = [0, 0, 0]
        v_2 = [0, 0, 0]
        a_2 = [0, 0, 0]
        o_2 = [0, 0, 0]
        v_3 = [0, 0, 0]
        a_3 = [0, 0, 0]
        o_3 = [0, 0, 0]
        shift_1 = [0, 0, 0]
        shift_2 = [0, 0, 0]
        shift_3 = [0, 0, 0]
        if time_1 < speed_t_0 or time_1 < imu_t_0:
            pass
        elif time_4 > speed_t or time_4 > imu_t:
            pass
        else:
            while time_2 > speed_time[speed_index]:
                speed_index += 1
            while time_2 > imu_time[imu_index]:
                imu_index += 1
            s_1 = (((speed[speed_time[speed_index]]-speed[speed_time[speed_index-1]])/(speed_time[speed_index]-speed_time[speed_index-1]))*(time_2-speed_time[speed_index-1])+speed[speed_time[speed_index-1]])
            o_1[0] = (((ori[imu_time[imu_index]][0]-ori[imu_time[imu_index-1]][0])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_2-imu_time[imu_index-1])+ori[imu_time[imu_index-1]][0])
            o_1[1] = (((ori[imu_time[imu_index]][1]-ori[imu_time[imu_index-1]][1])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_2-imu_time[imu_index-1])+ori[imu_time[imu_index-1]][1])
            o_1[2] = (((ori[imu_time[imu_index]][2]-ori[imu_time[imu_index-1]][2])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_2-imu_time[imu_index-1])+ori[imu_time[imu_index-1]][2])
            o_1_unit = (o_1[0]**2+o_1[1]**2+o_1[2]**2)**(0.5)
            v_1 = [j * (s_1/o_1_unit) for j in o_1]
            a_1[0] = (((acc[imu_time[imu_index]][0]-acc[imu_time[imu_index-1]][0])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_2-imu_time[imu_index-1])+acc[imu_time[imu_index-1]][0])
            a_1[1] = (((acc[imu_time[imu_index]][1]-acc[imu_time[imu_index-1]][1])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_2-imu_time[imu_index-1])+acc[imu_time[imu_index-1]][1])
            a_1[2] = (((acc[imu_time[imu_index]][2]-acc[imu_time[imu_index-1]][2])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_2-imu_time[imu_index-1])+acc[imu_time[imu_index-1]][2])
            while time_3 > speed_time[speed_index]:
                speed_index += 1
            while time_3 > imu_time[imu_index]:
                imu_index += 1
            s_2 = (((speed[speed_time[speed_index]]-speed[speed_time[speed_index-1]])/(speed_time[speed_index]-speed_time[speed_index-1]))*(time_3-speed_time[speed_index-1])+speed[speed_time[speed_index-1]])
            o_2[0] = (((ori[imu_time[imu_index]][0]-ori[imu_time[imu_index-1]][0])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_3-imu_time[imu_index-1])+ori[imu_time[imu_index-1]][0])
            o_2[1] = (((ori[imu_time[imu_index]][1]-ori[imu_time[imu_index-1]][1])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_3-imu_time[imu_index-1])+ori[imu_time[imu_index-1]][1])
            o_2[2] = (((ori[imu_time[imu_index]][2]-ori[imu_time[imu_index-1]][2])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_3-imu_time[imu_index-1])+ori[imu_time[imu_index-1]][2])
            o_2_unit = (o_2[0]**2+o_2[1]**2+o_2[2]**2)**(0.5)
            v_2 = [j * (s_2/o_2_unit) for j in o_2]
            a_2[0] = (((acc[imu_time[imu_index]][0]-acc[imu_time[imu_index-1]][0])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_3-imu_time[imu_index-1])+acc[imu_time[imu_index-1]][0])
            a_2[1] = (((acc[imu_time[imu_index]][1]-acc[imu_time[imu_index-1]][1])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_3-imu_time[imu_index-1])+acc[imu_time[imu_index-1]][1])
            a_2[2] = (((acc[imu_time[imu_index]][2]-acc[imu_time[imu_index-1]][2])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_3-imu_time[imu_index-1])+acc[imu_time[imu_index-1]][2])
            while time_4 > speed_time[speed_index]:
                speed_index += 1
            while time_4 > imu_time[imu_index]:
                imu_index += 1
            s_3 = (((speed[speed_time[speed_index]]-speed[speed_time[speed_index-1]])/(speed_time[speed_index]-speed_time[speed_index-1]))*(time_4-speed_time[speed_index-1])+speed[speed_time[speed_index-1]])
            o_3[0] = (((ori[imu_time[imu_index]][0]-ori[imu_time[imu_index-1]][0])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_4-imu_time[imu_index-1])+ori[imu_time[imu_index-1]][0])
            o_3[1] = (((ori[imu_time[imu_index]][1]-ori[imu_time[imu_index-1]][1])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_4-imu_time[imu_index-1])+ori[imu_time[imu_index-1]][1])
            o_3[2] = (((ori[imu_time[imu_index]][2]-ori[imu_time[imu_index-1]][2])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_4-imu_time[imu_index-1])+ori[imu_time[imu_index-1]][2])
            o_3_unit = (o_3[0]**2+o_3[1]**2+o_3[2]**2)**(0.5)
            v_3 = [j * (s_3/o_3_unit) for j in o_3]
            a_3[0] = (((acc[imu_time[imu_index]][0]-acc[imu_time[imu_index-1]][0])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_4-imu_time[imu_index-1])+acc[imu_time[imu_index-1]][0])
            a_3[1] = (((acc[imu_time[imu_index]][1]-acc[imu_time[imu_index-1]][1])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_4-imu_time[imu_index-1])+acc[imu_time[imu_index-1]][1])
            a_3[2] = (((acc[imu_time[imu_index]][2]-acc[imu_time[imu_index-1]][2])/(imu_time[imu_index]-imu_time[imu_index-1]))*(time_4-imu_time[imu_index-1])+acc[imu_time[imu_index-1]][2])
        for k in range(3):
            shift_1[k] = v_1[k] * (time_2 - time_1) + 0.5 * a_1[k] * (time_2 - time_1)**2
            shift_2[k] = v_2[k] * (time_3 - time_1) + 0.5 * a_2[k] * (time_3 - time_1)**2
            shift_3[k] = v_3[k] * (time_4 - time_1) + 0.5 * a_3[k] * (time_4 - time_1)**2
        floder_path_0 = os.path.join(seq_path, 'dataset', line_0[:-1])
        floder_path_1 = os.path.join(seq_path, 'dataset', line_1[:-1])
        floder_path_2 = os.path.join(seq_path, 'dataset', line_2[:-1])
        floder_path_3 = os.path.join(seq_path, 'dataset', line_3[:-1])
        # read point clouds
        f = open(os.path.join(floder_path_0, 'output.csv'), 'r')
        xyz_0 = np.array(list(csv.reader(f))).reshape(-1,3)
        f = open(os.path.join(floder_path_1, 'output.csv'), 'r')
        xyz_1 = np.array(list(csv.reader(f)), dtype=np.float32).reshape(-1,3)
        for i in range(np.shape(xyz_1)[0]):
            xyz_1[i] -= shift_1
        f = open(os.path.join(floder_path_2, 'output.csv'), 'r')
        xyz_2 = np.array(list(csv.reader(f)), dtype=np.float32).reshape(-1,3)
        for i in range(np.shape(xyz_2)[0]):
            xyz_2[i] -= shift_2
        f = open(os.path.join(floder_path_3, 'output.csv'), 'r')
        xyz_3 = np.array(list(csv.reader(f)), dtype=np.float32).reshape(-1,3)
        for i in range(np.shape(xyz_3)[0]):
            xyz_3[i] -= shift_3
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
    print('')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Read dataset & marker')
    parser.add_argument('--seq', default = 'seq1', type=str, help = 'Which sequence do you want to read')
    args = parser.parse_args()
    combine_2(args)
