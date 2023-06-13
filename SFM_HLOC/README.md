# SFM

## HLOC - the hierarchical localization toolbox

Reference: https://github.com/cvg/Hierarchical-Localization 
Revise a little part of this code to support our own road marker feature detection
(only revisit demo.py and extract_features.py)

# Package Download

- python -m pip install -e .


# Step
1. Extract SuperPoint local features for all database and query images
2. Only preserve features based on filtered_corners(y,x).npy
  => If feature lies within the square of size 10, where the center was each corner in the numpy array
  => This feature is considered to be a useful feature for road markers, append to feature list
3. Build a reference 3D SfM model
   1. Find covisible database images, with retrieval or a prior SfM model
   2. Match these database pairs with SuperGlue
   3. Triangulate a new SfM model with COLMAP

# Reproduce

This model failed to reconstruct any SFM model based on the images
The feature selection technique may reduce too many keypoints and lead to wrong matching
Linear Solver in COLMAP failed to converge

1. Make sure filtered_corners(y,x).npy is in each folder
2. python demo.py