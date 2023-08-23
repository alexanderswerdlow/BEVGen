import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import Union
from image_utils import Im
from multi_view_generation.bev_utils.util import Cameras
import multi_view_generation.bev_utils.util as util
from multi_view_generation.bev_utils import Dataset
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from einops import repeat
import os

# Taken from:
# https://github.com/bradyz/cross_view_transformers/blob/master/cross_view_transformer/visualizations/common.py
# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/color_map.py
COLORS = {
    # static
    "lane": (110, 110, 110),
    "road_segment": (90, 90, 90),
    # dividers
    "road_divider": (255, 200, 0),
    "lane_divider": (130, 130, 130),
    # dynamic
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "bus": (255, 127, 80),
    "trailer": (255, 140, 0),
    "construction": (233, 150, 70),
    "pedestrian": (0, 0, 230),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "nothing": (200, 200, 200),
}

ARGOVERSE_COLORS = {
    "driveable_area": (110, 110, 110),
    "lane_divider": (130, 130, 130),
    "ped_xing": (255, 200, 0),
    "pedestrian": (0, 0, 230),
    "vehicle": (255, 158, 0),
    "large_vehicle": (255, 99, 71),
    "other": (255, 127, 80),
    "nothing": (200, 200, 200),
}

def save_binary_as_image(data, filename):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    os.makedirs(Path(filename).parents[0], exist_ok=True)
    plt.imsave(filename, data, cmap=cm.hot, vmin=0, vmax=1)

def return_binary_as_image(data):
    return repeat(((data.detach().cpu().numpy()) * 255).astype(np.uint8), '... -> ... c', c=3)

def viz_bev(bev: Union[np.ndarray, torch.FloatTensor], dataset: Dataset = Dataset.NUSCENES) -> Im:
    """(c h w) torch [0, 1] float -> (h w 3) np [0, 255] uint8"""
    
    if torch.is_tensor(bev):
        bev = bev.detach().cpu().numpy()

    if bev.dtype == np.uint8:
        if bev.max() > 1:
            bev = (bev / 255.0)
    
    bev = bev.astype(np.float32)
    assert len(bev.shape) == 3
    assert bev.dtype == np.float32
    if bev.shape[1] == bev.shape[2] and bev.shape[0] < bev.shape[1]:
        bev = bev.transpose(1, 2, 0)

    # img is (h w c) np [0, 1] float
    if dataset == Dataset.ARGOVERSE:
        color_dict = ARGOVERSE_COLORS
        classes = ["driveable_area", "lane_divider", "ped_xing", "other", "pedestrian", "vehicle", "large_vehicle"]
        bev[..., range(bev.shape[-1])] = bev[..., [4, 5, 6, 3, 1, 0, 2]]
    else:
        raise ValueError()

    assert 0 <= bev.min() <= bev.max() <= 1.0
    colors = np.array([color_dict[s] for s in classes], dtype=np.uint8)

    h, w, c = bev.shape
    assert c == len(classes)

    # Prioritize higher class labels
    eps = (1e-5 * np.arange(c))[None, None]
    idx = (bev + eps).argmax(axis=-1)
    val = np.take_along_axis(bev, idx[..., None], -1)

    # Spots with no labels are light grey
    empty = np.uint8(color_dict["nothing"])[None, None]

    result = (val * colors[idx]) + ((1 - val) * empty)
    result = np.uint8(result)

    return Im(result)

def raw_output_data_bev_grid(batch):
    images = Im(batch['image']).denormalize().torch
    ret_images = []

    for i in range(batch['image'].shape[0]):
        if batch['dataset'][i] == 'nuscenes':
            viz_func = camera_bev_grid
        else:
            viz_func = argoverse_camera_bev_grid
        image_dict = {batch['cam_name'][k][i]: images[i, k] for k in range(images.shape[1])}
        ret_images.append(viz_func(images=image_dict, bev=batch["segmentation"][i]))
    return ret_images


def batched_camera_bev_grid(cfg, images, bev=None, pred=None):
    if len(images.shape) == 4:
        images = images.unsqueeze(0)

    image_names = cfg.cam_names
    if cfg.dataset == Dataset.NUSCENES:
        viz_func = camera_bev_grid
        image_names = Cameras.NUSCENES_CAMERAS if images.shape[1] == 6 else Cameras.NUSCENES_ABLATION_CAMERAS
    else:
        viz_func = argoverse_camera_bev_grid
        # image_names = Cameras.ARGOVERSE_CAMERAS
        
    ret_images = []
    for i in range(images.shape[0]):
        image_dict = {image_names[k]: images[i, k] for k in range(images.shape[1])}
        ret_images.append(Im(viz_func(image_dict, bev[i] if bev is not None else None, pred[i] if pred is not None else None)).torch)

    return torch.stack(ret_images, dim=0)


def ground_truth_camera_bev_grid(images: dict, bev=None, pred=None, keep_bev_space=False, add_car=True):
    images = {k: (Im(v).pil if not isinstance(v, Image.Image) else v) for k, v in images.items()}

    landscape_width = images[next(iter(images))].size[0]
    landscape_height = images[next(iter(images))].size[1]

    horiz_padding = 0
    vert_padding = 5

    six_cameras = len(images) == 6
    height = 2 * landscape_height + 2 * vert_padding + landscape_width
    width = landscape_width + landscape_width + landscape_width + 3 * horiz_padding + (landscape_height * 2 + horiz_padding if pred is not None else 0)
    pred_width = 0

    # bev = None

    if bev is not None:
        bev = viz_bev(bev, dataset=Dataset.NUSCENES).pil.resize((landscape_width, landscape_width))
    elif keep_bev_space:
        bev = Image.new('RGB', (height, height))

    if pred is not None:
        pred = Im(pred).pil.resize((height, height))

    dst = Image.new('RGBA', (width, height))
    
    if bev:
        bev_width, bev_height = bev.size[0], bev.size[1]
        if add_car:
            bev = ImageDraw.Draw(bev)
            width_, height_ = 6/256 * bev_width, 12/256 * bev_height
            bev.rectangle((bev_width // 2 - width_, bev_height // 2 - height_, bev_width // 2 + width_, bev_height // 2 + height_), fill="#00FF11")
            bev = bev._image
        
        dst.paste(bev, (5 + landscape_width, landscape_height + vert_padding))
    else:
        bev_width = 0

    bev_width = 0

    if pred:
        dst.paste(pred, (horiz_padding + bev_width, 0))
        pred_width = pred.size[0] + horiz_padding

    add_num = (landscape_height * 2 - landscape_width) // 2
    dst.paste(images['CAM_FRONT_LEFT'], (bev_width + horiz_padding, landscape_height - add_num))
    dst.paste(images['CAM_FRONT'], (5 + bev_width + landscape_width + 2 * horiz_padding, 0))
    dst.paste(images['CAM_FRONT_RIGHT'], (10 + bev_width + 2 * landscape_width + 3 * horiz_padding, landscape_height - add_num))

    if six_cameras:
        dst.paste(images['CAM_BACK_LEFT'].transpose(Image.Transpose.FLIP_LEFT_RIGHT), (bev_width + horiz_padding, landscape_height + vert_padding + landscape_height - add_num))
        dst.paste(images['CAM_BACK'].transpose(Image.Transpose.FLIP_LEFT_RIGHT), (5 + bev_width + landscape_width + 2 * horiz_padding, 2 * vert_padding + landscape_height + landscape_height + landscape_height - (2 * add_num)))
        dst.paste(images['CAM_BACK_RIGHT'].transpose(Image.Transpose.FLIP_LEFT_RIGHT), (10 + bev_width + 2 * landscape_width + 3 * horiz_padding, landscape_height + vert_padding + landscape_height - add_num))

    return dst

def camera_bev_grid(images: dict, bev=None, pred=None, keep_bev_space=False, add_car=True):
    images = {k: (Im(v).pil if not isinstance(v, Image.Image) else v) for k, v in images.items()}

    landscape_width = images[next(iter(images))].size[0]
    landscape_height = images[next(iter(images))].size[1]

    horiz_padding = 5
    vert_padding = 5

    six_cameras = len(images) == 6
    height = landscape_height + vert_padding + (landscape_height if six_cameras else 0)
    width = landscape_width + landscape_width + landscape_width + (landscape_height * 2) + 3 * horiz_padding + (landscape_height * 2 + horiz_padding if pred is not None else 0)
    pred_width = 0

    if bev is not None:
        bev = viz_bev(bev, dataset=Dataset.NUSCENES).pil.resize((height, height))
    elif keep_bev_space:
        bev = Image.new('RGB', (height, height))

    if pred is not None:
        pred = Im(pred).pil.resize((height, height))

    dst = Image.new('RGBA', (width, height))

    if bev:
        bev_width, bev_height = bev.size[0], bev.size[1]
        if add_car:
            bev = ImageDraw.Draw(bev)
            bev.rectangle((bev_width // 2 - 6, bev_height // 2 - 12, bev_width // 2 + 6, bev_height // 2 + 12), fill="#00FF11")
            bev = bev._image
        dst.paste(bev, (0, 0))
    else:
        bev_width = 0

    if pred:
        dst.paste(pred, (horiz_padding + bev_width, 0))
        pred_width = pred.size[0] + horiz_padding

    dst.paste(images['CAM_FRONT_LEFT'], (pred_width + bev_width + horiz_padding, 0))
    dst.paste(images['CAM_FRONT'], (pred_width + bev_width + landscape_width + 2 * horiz_padding, 0))
    dst.paste(images['CAM_FRONT_RIGHT'], (pred_width + bev_width + 2 * landscape_width + 3 * horiz_padding, 0))

    if six_cameras:
        dst.paste(images['CAM_BACK_LEFT'].transpose(Image.Transpose.FLIP_LEFT_RIGHT), (pred_width + bev_width + horiz_padding, landscape_height + vert_padding))
        dst.paste(images['CAM_BACK'].transpose(Image.Transpose.FLIP_LEFT_RIGHT), (pred_width + bev_width + landscape_width + 2 * horiz_padding, landscape_height + vert_padding))
        dst.paste(images['CAM_BACK_RIGHT'].transpose(Image.Transpose.FLIP_LEFT_RIGHT), (pred_width + bev_width + 2 * landscape_width + 3 * horiz_padding, landscape_height + vert_padding))

    return dst


def argoverse_camera_bev_grid(images: dict, bev=None, keep_bev_space=False, add_car=True):
    images = {k: (Im(v).pil if not isinstance(v, Image.Image) else v) for k, v in images.items()}

    landscape_width = images[next(iter(images))].size[0]
    landscape_height = images[next(iter(images))].size[1]

    horiz_padding = 5

    height = landscape_height
    width = len(images) * landscape_width + (landscape_height) + 4 * horiz_padding

    if bev is not None:
        bev = viz_bev(bev, dataset=Dataset.ARGOVERSE).pil.resize((height, height))
    elif keep_bev_space:
        bev = Image.new('RGB', (height, height))

    dst = Image.new('RGBA', (width, height))

    if bev:
        bev_width, bev_height = bev.size[0], bev.size[1]
        if add_car:
            bev = ImageDraw.Draw(bev)
            bev.rectangle((bev_width // 2 - 4, bev_height // 2 - 8, bev_width // 2 + 4, bev_height // 2 + 8), fill="#00FF11")
            bev = bev._image
        dst.paste(bev, (0, 0))
    else:
        bev_width = 0

    if len(images) == 4:
        dst.paste(images['ring_side_left'], (bev_width + horiz_padding, 0))
        dst.paste(images['ring_front_left'], (bev_width + 1 * landscape_width + 2 * horiz_padding, 0))
        dst.paste(images['ring_front_right'], (bev_width + 2 * landscape_width + 3 * horiz_padding, 0))
        dst.paste(images['ring_side_right'], (bev_width + 3 * landscape_width + 4 * horiz_padding, 0))
    elif len(images) == 1:
        dst.paste(next(iter(images.values())), (bev_width + horiz_padding, 0))
    elif len(images) == 3:
        dst.paste(images['ring_front_left'], (bev_width + horiz_padding, 0))
        dst.paste(images['ring_front_center'], (bev_width + 1 * landscape_height + 2 * horiz_padding, 0))
        dst.paste(images['ring_front_right'], (bev_width + 2 * landscape_height + 3 * horiz_padding, 0))

    return dst


if __name__ == "__main__":
    images = {k: Im(torch.randn((224, 400, 3))).pil for k in Cameras.NUSCENES_ABLATION_CAMERAS}
    camera_bev_grid(images, torch.randn((256, 256, 21)))
    batched_camera_bev_grid(torch.randn((2, 6, 256, 256, 3)))
