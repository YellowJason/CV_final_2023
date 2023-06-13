# CV Final

# SLAM

- Simultaneous localization and mapping
- [https://zhuanlan.zhihu.com/p/538017402](https://zhuanlan.zhihu.com/p/538017402) Read **Epipolar Geometry** for details
- Utilize current frame and last frame
    1. Expand into 3 dimensions
    2. **Feature Detection, Extraction and Matching**
    3. Compute transformation matrix and mapping
    4. Compute **Essential matrix**
    5. Perform SVD and compute **P**
    6. Map into real 3D space using **Triangulation**

# Package Download

- pip install -r requirements.txt

# File Description

### converter.py

Convert array from 2d to 3d

### eval.py

Currently only evaluate seq1 result

### extractor.py

Road marker key points detection

### gt_gen.py

Make gt_pose.txt for evaluation

### ICP.py

Run ICP to obtain x,y of all local timestamps

### main.py

- Currently, only seq1 is computed
- Every camera is independent

### normalize.py

Functions for step 4,5,6

### pointmap.py

Map result array into 3d coord (x,y,z)


# Reproduce

Only tested on sequence 1 with visualization
Eipolar Geometry does not work well on objects on one surface

1. python find_corners.py --seq seq1 => generate corners_(y,x).npy
2. python main.py

After visualizing some images, program would display matches of one pair, generated point cloud and exit