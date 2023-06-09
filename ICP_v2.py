import os, sys, argparse, csv, copy
import numpy as np
import open3d as o3d
from glob import glob


def ICP(source, target, threshold, init_pose, iteration=30):
    # implement iterative closet point and return transformation matrix
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_pose,
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iteration)
    )
    print(reg_p2p)
    assert len(reg_p2p.correspondence_set) != 0, 'The size of correspondence_set between your point cloud and sub_map should not be zero.'
    #print(reg_p2p.transformation)
    return reg_p2p.transformation

def csv_reader(filename):
    # read csv file into numpy array
    data = np.loadtxt(filename, delimiter=',')
    return data

def numpy2pcd(arr):
    # turn numpy array into open3d point cloud format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    return pcd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Read dataset & marker')
    parser.add_argument('--seq', default = 'seq1', type=str, help = 'Which sequence do you want to read')
    args = parser.parse_args()

    with open(f'./ITRI_DLC2/{args.seq}/localization_timestamp.txt', 'r') as file:
        lines = file.readlines()
    
    # Remove the newline character ('\n') from each line
    lines = [line.strip() for line in lines]
    
    if args.seq in ['seq1', 'seq2', 'seq3']:
        out_path = f'./ITRI_dataset/{args.seq}/pred_pose.txt'
    elif args.seq in ['test1', 'test2']:
        out_path = f'./solution/{args.seq}/pred_pose.txt'
    
    with open(out_path, 'w+') as pred:
        for i in range(len(lines)):
            if args.seq in ['seq1', 'seq2', 'seq3']:
                path_name = f'./ITRI_dataset/{args.seq}/dataset/{lines[i]}'
            elif args.seq in ['test1', 'test2']:
                path_name = f'./ITRI_DLC/{args.seq}/dataset/{lines[i]}'
            path_name2 = f'./ITRI_DLC2/{args.seq}/new_init_pose/{lines[i]}'
            # Target point cloud
            target = csv_reader(f'{path_name}/sub_map.csv')
            target_pcd = numpy2pcd(target)

            # Source point cloud
            #TODO: Read your point cloud here#
            source = csv_reader(f'{path_name}/output_merge.csv')
            source_pcd = numpy2pcd(source)

            # Initial pose
            init_pose = csv_reader(f'{path_name2}/initial_pose.csv')

            # Implement ICP
            transformation = ICP(source_pcd, target_pcd, threshold=0.5, init_pose=init_pose)
            pred_x = transformation[0,3]
            pred_y = transformation[1,3]
            # print(pred_x, pred_y)
            pred.write(str(pred_x))
            pred.write(" ")
            pred.write(str(pred_y))
            pred.write("\n")