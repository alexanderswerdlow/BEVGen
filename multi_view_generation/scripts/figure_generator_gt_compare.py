import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from multi_view_generation.bev_utils import Cameras
from multi_view_generation.bev_utils.visualize import argoverse_camera_bev_grid, viz_bev
from multi_view_generation.modules.transformer.mingpt_sparse import Dataset
from multi_view_generation.bev_utils import camera_bev_grid
from pathlib import Path
import typer
import numpy as np
import os
from os.path import exists
import tarfile
import os.path
from PIL import Image
from tqdm import tqdm
import random
from typing import Union
from image_utils import Im
from multi_view_generation.bev_utils import Cameras, util, CLASSES
from multi_view_generation.bev_utils import Dataset
from multi_view_generation.modules.transformer.mingpt_sparse import GPTConfig
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from einops import repeat
import os

def create_grid(rows, cols, images, index, padding, nuscenes_special=False):
    """
    Create a grid of images with padding in between.
    """
    # h, w
    if nuscenes_special:
        image_height, image_width = images[0].size[::-1]
        grid_width = sum([image.size[0] for i, image in enumerate(images) if i < cols])
    else:
        image_height, image_width = images[0].size[::-1]
        grid_width = cols * image_width + (cols - 1) * padding

    grid_height = rows * image_height + (rows - 1) * padding
    grid = Image.new("RGBA", (grid_width, grid_height))
    for image, index in zip(images, index):
        row = index // cols
        col = index % cols
        if nuscenes_special:
            if col > 3:
                x = 2 * (image_width + padding) + (images[3].size[1] + padding) + (col - 3) * (image_width + padding)
            else:
                x = col * (image_width + padding)
        else:
            x = col * (image_width + padding)
        y = row * (image_height + padding)
        grid.paste(image, (x, y))
    return grid

def argoverse_camera_bev_grid_gt(images: dict, images_gt: dict, bev=None, keep_bev_space=False, add_car=True):
    landscape_height = images[next(iter(images))].size[1]
    height = landscape_height

    if bev is not None:
        bev = viz_bev(bev, dataset=Dataset.ARGOVERSE).pil.resize((height, height))
    elif keep_bev_space:
        bev = Image.new('RGBA', (height, height))

    if add_car:
        bev_width, bev_height = bev.size[0], bev.size[1]
        bev = ImageDraw.Draw(bev)
        bev.rectangle((bev_width // 2 - 4, bev_height // 2 - 8, bev_width // 2 + 4, bev_height // 2 + 8), fill="#00FF11")
        bev = bev._image

    dst = create_grid(2, 4, [images['ring_front_left'], images['ring_front_center'], images['ring_front_right'], bev, images_gt['ring_front_left'], images_gt['ring_front_center'], images_gt['ring_front_right']], [1, 2, 3, 0, 5, 6, 7], 5)
    return dst

def nuscenes_camera_bev_grid_gt(images: dict, images_gt: dict, bev=None, keep_bev_space=False, add_car=True):
    landscape_height = images[next(iter(images))].size[1]
    height = landscape_height
    padding = 5

    if bev is not None:
        bev = viz_bev(bev, dataset=Dataset.NUSCENES).pil.resize((2 * height + padding, 2 * height+ padding))
    elif keep_bev_space:
        bev = Image.new('RGBA', (height + padding, height + padding))

    if add_car:
        bev_width, bev_height = bev.size[0], bev.size[1]
        bev = ImageDraw.Draw(bev)
        bev.rectangle((bev_width // 2 - 4, bev_height // 2 - 8, bev_width // 2 + 4, bev_height // 2 + 8), fill="#00FF11")
        bev = bev._image

    imgs = [
        images['CAM_FRONT_LEFT'], images['CAM_FRONT'], images['CAM_FRONT_RIGHT'], 
        bev, 
        images_gt['CAM_FRONT_LEFT'], images_gt['CAM_FRONT'], images_gt['CAM_FRONT_RIGHT'],
        images['CAM_BACK_LEFT'].transpose(Image.Transpose.FLIP_LEFT_RIGHT), images['CAM_BACK'].transpose(Image.Transpose.FLIP_LEFT_RIGHT), images['CAM_BACK_RIGHT'].transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        images_gt['CAM_BACK_LEFT'].transpose(Image.Transpose.FLIP_LEFT_RIGHT), images_gt['CAM_BACK'].transpose(Image.Transpose.FLIP_LEFT_RIGHT), images_gt['CAM_BACK_RIGHT'].transpose(Image.Transpose.FLIP_LEFT_RIGHT)]

    dst = create_grid(2, 7, imgs, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13], padding, nuscenes_special=True)
    return dst

def gen_figure(
    folder: Path = Path('archive/figures/images'), 
    output_name: str = 'output', 
    folder_: Path = None,
    output_name_: str = None,
    output_folder = Path(__file__).parent.resolve(),
    add_car: bool = True,
    dataset: Dataset = Dataset.NUSCENES
    ):

    camera_list = Cameras.NUSCENES_CAMERAS if dataset == Dataset.NUSCENES else Cameras.ARGOVERSE_FRONT_CAMERAS

    images = {image_name: Image.open(folder / f'{image_name}.jpg') for image_name in camera_list}
    images_ = {image_name: Image.open(folder_ / f'{image_name}.jpg') for image_name in camera_list}

    if exists(folder / 'bev.npz'):
        bev = np.load(folder / 'bev.npz')['arr_0']
        if dataset == Dataset.NUSCENES:
            bev = bev[..., :12]
    else:
        bev = None

    viz_func = nuscenes_camera_bev_grid_gt if dataset == Dataset.NUSCENES else argoverse_camera_bev_grid_gt
    dst = viz_func(images, images_, bev, keep_bev_space=True, add_car=add_car)
    dst.save(output_folder / f'{output_name}.png')


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def create_site(folder: Path, dataset: Dataset = Dataset.NUSCENES, shuffle: bool =True):
    site_path = folder / f'{folder.name}_site_compare'
    gen_path = site_path / 'gen_images_compare'
    os.makedirs(site_path, exist_ok=True)
    os.makedirs(gen_path, exist_ok=True)

    paths = list(Path(folder / 'sample').iterdir())
    random.shuffle(paths)

    for _, path in tqdm(enumerate(paths)):
        if path.is_dir():
            path_ = path.parent.parent / 'sample_gt' / path.name
            gen_figure(path, path.name, path_, path_.name, gen_path, dataset=dataset)


    code = """<html><head></head><body>"""
    end_code = """</body></html>"""

    gen_paths = list(gen_path.iterdir())
    if shuffle:
        random.shuffle(gen_paths)
    else:
        gen_paths = sorted(gen_paths)

    for path1 in gen_paths:
        if path1.suffix == '.png' and exists(gen_path / path1.name):
            code += f' <img src="{Path("gen_images_compare") / path1.name}" width="100%" style="margin: 10px 0 5px 0px;"/> \n'

    with open(folder / f'{folder.name}_site_compare' / "index.html","w") as f:
        f.write(code + end_code)

    make_tarfile(folder / f'{folder.name}_compare.tar.gz', folder / f'{folder.name}_site_compare')

def gen_bev(folder: Path):
    for path in Path(folder).iterdir():
        if path.is_dir():
            if exists(path / 'bev.npz'):
                bev = np.load(path / 'bev.npz')['arr_0'][..., :12]
                bev = viz_bev(bev, Dataset.NUSCENES).pil
                bev.save(path / 'bev.png')


def main(
    folder: Path = Path('archive/figures/images'),
    output_name: str = 'output',
    make_site: bool = False,
    make_bev: bool = False,
    argoverse: bool = False,
    shuffle: bool = True,
    ):
    if make_site:
        create_site(folder, dataset=Dataset.ARGOVERSE if argoverse else Dataset.NUSCENES, shuffle=shuffle)
    elif make_bev:
        gen_bev(folder)
    else:
        gen_figure(folder, output_name)

if __name__ == "__main__":
    typer.run(main)
