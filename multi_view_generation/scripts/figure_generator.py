import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from multi_view_generation.bev_utils import Cameras
from multi_view_generation.bev_utils.visualize import argoverse_camera_bev_grid, viz_bev
from multi_view_generation.bev_utils import Dataset
from multi_view_generation.modules.transformer.mingpt_sparse import GPTConfig
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

def gen_figure(
    folder: Path = Path('archive/figures/images'), 
    output_name: str = 'output', 
    output_folder = Path(__file__).parent.resolve(),
    add_car: bool = True,
    dataset: Dataset = Dataset.NUSCENES
    ):

    camera_list = Cameras.NUSCENES_CAMERAS if dataset == Dataset.NUSCENES else Cameras.ARGOVERSE_FRONT_CAMERAS
    images = {image_name: Image.open(folder / f'{image_name}.jpg') for image_name in camera_list}

    if exists(folder / 'bev.npz'):
        bev = np.load(folder / 'bev.npz')['arr_0']
        if dataset == Dataset.NUSCENES:
            bev = bev[..., :12]
    else:
        bev = None

    viz_func = camera_bev_grid if dataset == Dataset.NUSCENES else argoverse_camera_bev_grid
    dst = viz_func(images, bev, keep_bev_space=True, add_car=add_car)
    dst.save(output_folder / f'{output_name}.png')

figures = [
    ('figure_2', 'images')
]

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def create_site(folder: Path, dataset: Dataset = Dataset.NUSCENES, shuffle: bool =True):
    site_path = folder / f'{folder.name}_site'
    gen_path = site_path / 'gen_images'
    gt_path = site_path / 'gt_images'
    os.makedirs(site_path, exist_ok=True)
    os.makedirs(gen_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)

    paths = list(Path(folder / 'sample').iterdir())
    random.shuffle(paths)

    num_images = 1000

    for _, path in tqdm(enumerate(paths[:num_images])):
        if path.is_dir():
            gen_figure(path, path.name, gen_path, dataset=dataset)

    if exists(Path(folder / 'sample_gt')):
        for _, path in tqdm(enumerate(paths[:num_images])):
            path_ = path.parent.parent / 'sample_gt' / path.name
            if path_.is_dir():
                gen_figure(path_, path_.name, gt_path, dataset=dataset)

    code = """<html><head></head><body>"""
    end_code = """</body></html>"""

    gen_paths = list(gen_path.iterdir())
    if shuffle:
        random.shuffle(gen_paths)
    else:
        gen_paths = sorted(gen_paths)

    for path1 in gen_paths:
        if path1.suffix == '.png' and exists(gen_path / path1.name):
            code += f' <img src="{Path("gen_images") / path1.name}" width="100%" style="margin: 10px 0 5px 0px;"/> \n'
            if exists(gt_path / path1.name):
                code += f' <img src="{Path("gt_images") / path1.name}" width="100%" style="margin: 0px 0px 10px 0px;"/> \n'

    with open(folder / f'{folder.name}_site' / "index.html","w") as f:
        f.write(code + end_code)

    make_tarfile(folder / f'{folder.name}.tar.gz', folder / f'{folder.name}_site')

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
    use_list: bool = False,
    process_all: bool = False,
    make_site: bool = False,
    make_bev: bool = False,
    argoverse: bool = False,
    shuffle: bool = True,
    ):
    if use_list:
        for fig_name, fig_folder in figures:
            gen_figure(fig_folder, fig_name)
    elif process_all:
        os.makedirs(folder / 'images', exist_ok=True)
        for path in Path(folder).iterdir():
            if path.is_dir():
                gen_figure(path, path.name, folder / 'images')
    elif make_site:
        create_site(folder, dataset=Dataset.ARGOVERSE if argoverse else Dataset.NUSCENES, shuffle=shuffle)
    elif make_bev:
        gen_bev(folder)
    else:
        gen_figure(folder, output_name)

if __name__ == "__main__":
    typer.run(main)
