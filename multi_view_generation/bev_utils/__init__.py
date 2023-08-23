import os
from pathlib import Path
from multi_view_generation.bev_utils.util import Cameras, Dataset
from multi_view_generation.bev_utils.visualize import camera_bev_grid, batched_camera_bev_grid, viz_bev, argoverse_camera_bev_grid, raw_output_data_bev_grid, save_binary_as_image, return_binary_as_image

ARGOVERSE_DIR = Path(os.getenv('ARGOVERSE_DATA_DIR', 'datasets/av2')).expanduser().resolve()
NUSCENES_DIR = Path(os.getenv('NUSCENES_DATA_DIR', 'datasets/nuscenes')).expanduser().resolve()
SAVE_DATA_DIR = Path(os.getenv('SAVE_DATA_DIR', 'datasets')).expanduser().resolve()