%load_ext autoreload
%autoreload 2
import tqdm, tqdm.notebook
tqdm.tqdm = tqdm.notebook.tqdm  # notebook-friendly progress bars
from pathlib import Path

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d

images = Path('../ITRI_dataset/seq1/dataset/')
outputs = Path('outputs/demo/')
!rm -rf $outputs
sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']


# Only demo forward camera
with open("../ITRI_dataset/seq1/all_timestamp.txt") as f:
    lines = f.readlines()
lines = [s.rstrip("\n") for s in lines]
references = [s + "/raw_image.jpg" for s in lines]
# references = [p.relative_to(images).as_posix() for p in (images).iterdir()]
# references = [s for s in references if s.endswith(".jpg")]
references = references[10:30]
print(len(references), "mapping images")
plot_images([read_image(images / r) for r in references[:4]], dpi=50)

extract_features.main(feature_conf, images, image_list=references, feature_path=features)
pairs_from_exhaustive.main(sfm_pairs, image_list=references)
match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)


# Given camera parameters
camera_matrix_data = [658.897676983, 0.0, 719.335668486, 0.0, 659.869992391, 468.32106087, 0.0, 0.0, 1.0]
distortion_coefficients_data = [-0.00790509510948, -0.0356504181626, 0.00803540169827, 0.0059685787996]

# Extract camera parameters
fx = camera_matrix_data[0]
fy = camera_matrix_data[4]
cx = camera_matrix_data[2]
cy = camera_matrix_data[5]

# Create camera model in COLMAP format
camera_model = 'PINHOLE'
camera_params = ','.join(map(str, (fx, fy, cx, cy)))
opts = dict(camera_model=camera_model, camera_params=camera_params)

model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references)
fig = viz_3d.init_figure()
viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping")
fig.show()