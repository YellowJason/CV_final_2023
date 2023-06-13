from slam import process
from display import Display
from pointmap import PointMap

import cv2
import csv
import os
import numpy as np
import open3d as o3d
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
pmap = PointMap()
display1 = Display()
display2 = Display()
pcd1 = o3d.geometry.PointCloud()
pcd2 = o3d.geometry.PointCloud()
visualizer1 = o3d.visualization.Visualizer()
# visualizer1.create_window(window_name="3D plot2", width=1440, height=928)
# visualizer2 = o3d.visualization.Visualizer()
# visualizer2.create_window(window_name="3D plot3", width=1440, height=928)
with open('../ITRI_dataset/seq1/all_timestamp.txt', 'r') as file:
	lines = file.readlines()

# Remove the newline character ('\n') from each line
lines = [line.strip() for line in lines]

print(lines[3])  # Print the 4th entry (index 3)
camera_pos = ["f", "fl", "fr", "b"]
for i in range(4, 5):
    folder_path = "./output/%s" % (lines[i])
    gt_folder = "../ITRI_dataset/seq1/dataset/%s" % (lines[i])
    # file_path = os.path.join(folder_path, 'output.csv')
    file_path = os.path.join("./", 'concat.csv')
    gt_path = os.path.join(gt_folder, 'sub_map.csv')
    # data1 = np.genfromtxt(file_path, delimiter=",")
    data1 = np.load("./array.npy")
    data2 = np.genfromtxt(gt_path, delimiter=",")
    print(data1.shape)
    print(data2.shape)
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract the x, y, and z coordinates from 'tripoints3d_all'
    x = data1[:, 0]
    y = data1[:, 1]
    z = data1[:, 2]

    # Create the 3D scatter plot
    ax.scatter(x, y, z, c='b', marker='o')

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()
    # display1.display_points3d(data1, pcd1, visualizer1)
    # display2.display_points3d(data2, pcd2, visualizer2)
    # time.sleep(30)