import numpy as np
import torch
from torch.utils.data import Dataset
from random import randint

from multi_view_generation.bev_utils.util import Cameras, get_fake_stage_2_data

def random_bbox(bbox):
    v = [randint(0, v) for v in bbox]
    left = min(v[0], v[2])
    upper = min(v[1], v[3])
    right = max(v[0], v[2])
    lower = max(v[1], v[3])
    return [left, upper, right, lower]

class NuScenesDatasetFake(Dataset):
    def __init__(self, stage='stage_2', cam_h=256, cam_w=256, seg_channels=21, cam_names=None, **kwargs):
        self.stage = stage
        self.cam_h = cam_h
        self.cam_w = cam_w
        self.seg_channels = seg_channels
        self.cam_names = Cameras[cam_names]
        
    def __getitem__(self, index: int):
        if self.stage == 'stage_2':
            return get_fake_stage_2_data(self.cam_h, self.cam_w, self.seg_channels, self.cam_names)
        elif self.stage == 'stage_1':
            return {
                'image': torch.randn(([self.cam_h, self.cam_w, 3]), dtype=torch.float32),
                'segmentation': torch.randn(([256, 256, 3]), dtype=torch.float32), 
                'angle': torch.pi,
                'dataset': 'nuscenes',
            }

    def __len__(self):
        return 100