import os, sys, argparse
import numpy as np


def calculate_dist(label, pred):
    assert label.shape[0] == pred.shape[0], 'The number of predicted results should be the same as the number of ground truth.'
    for i in range(label.shape[0]):

        print("Ground  truth: ", label[i])
        print("Predict truth:", pred[i])

    dist = np.sqrt(np.sum((label-pred)**2, axis=1))
    dist = np.mean(dist)
    return dist



def benchmark(dataset_path, sequences):
    for seq in sequences:
        if (seq == 'seq1'):
            label = np.loadtxt(os.path.join(dataset_path, 'gt_pose.txt'), delimiter=",")
            pred = np.loadtxt(os.path.join(dataset_path, 'pred_pose.txt'), delimiter=",")  #TODO: Enter your filename here#
            score = calculate_dist(label, pred)
            print(f'Mean Error of {seq}: {score:.5f}')



if __name__ == '__main__':
    dataset_path = './'
    sequences = ['seq1', 'seq2', 'seq3']
    benchmark(dataset_path, sequences)