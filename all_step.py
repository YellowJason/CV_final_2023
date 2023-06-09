import argparse
from find_corners import find_corners
from pinhole import pinhole
from combine_4_cam import combine

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Read dataset & marker')
    parser.add_argument('--seq', default = 'seq1', type=str, help = 'Which sequence do you want to read')
    args = parser.parse_args()
    find_corners(args)
    pinhole(args)
    combine(args)