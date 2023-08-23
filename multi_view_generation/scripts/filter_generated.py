from einops import rearrange
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from PIL import Image
from torchvision.transforms.functional import to_tensor
import typer
from pathlib import Path
from cleanfid import fid
from multi_view_generation.bev_utils.nuscenes_dataset import NuScenesDataset
from multi_view_generation.scripts.lpip import LearnedPerceptualImagePatchSimilarity
from multi_view_generation.bev_utils import Cameras
from multi_view_generation.bev_utils import SAVE_DATA_DIR
from os.path import exists
import torch
import shutil
import os
from functools import partial
from multiprocessing import Pool
import random

NUSCENES_CONFIG = {
    'split': 0,
    'return_cam_img': True,
    'return_bev_img': True,
    'return_all_cams': True,
    'stage_2_training': True,
    'metadrive_compatible_v2': True,
    'non_square_images': True,
    'mini_dataset': False,
    'only_keyframes': True,
}

from multiprocessing import current_process, Pool

def copy_imgs(image_idx, dataset_dir):
    dataset = NuScenesDataset(**NUSCENES_CONFIG)
    images = dataset.images

    from tqdm import tqdm
    pbar = tqdm(image_idx, unit="instances")
    for idx in pbar:
        sample_data_token = images[idx]
        cam_record = dataset.nusc.get("sample_data", sample_data_token)
        sample_token = cam_record["sample_token"]
        sample_record = dataset.nusc.get("sample", sample_token)

        sample_path = dataset_dir / 'sample' / sample_token
        os.makedirs(sample_path, exist_ok=True)

        sample_gt_path = dataset_dir / 'sample_gt' / sample_token
        os.makedirs(sample_gt_path, exist_ok=True)

        for cam_name in Cameras.NUSCENES_CAMERAS:
            cam_token = sample_record['data'][cam_name]
            cam_record = dataset.nusc.get('sample_data', cam_token)
            path = cam_record["filename"]

            if exists(dataset_dir / 'gen' / path):
                 shutil.copy(dataset_dir / 'gen' / path, sample_path / f'{cam_name}.jpg')

            if exists(dataset_dir / 'gt' / path):
                 shutil.copy(dataset_dir / 'gt' / path, sample_gt_path / f'{cam_name}.jpg')


def compute_overlap(image_idx, dataset_dir):

    p = current_process()._identity[0]
    idx_to_gpu = {1:'0', 2:'4', 3:'5', 4:'0', 5:'4', 6:'5'}

    device = torch.device(f'cuda:{idx_to_gpu[p]}')
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction='none').to(device)

    dataset = NuScenesDataset(**NUSCENES_CONFIG)
    images = dataset.images
    
    output_dir = dataset_dir.parent / f"{dataset_dir.name}_filtered_v2"

    samples_removed = 0
    samples_kept = 0
    from tqdm import tqdm
    pbar = tqdm(image_idx, unit="instances")
    batch_gt_imgs, batch_gen_imgs, batch_paths, batch_tokens = [], [], [], []

    os.makedirs(output_dir / 'sample_gt', exist_ok=True)
    os.makedirs(output_dir / 'sample', exist_ok=True)

    for idx in pbar:
        sample_data_token = images[idx]
        cam_record = dataset.nusc.get("sample_data", sample_data_token)
        sample_token = cam_record["sample_token"]
        sample_record = dataset.nusc.get("sample", sample_token)

        skip_sample = False
        gt_imgs, gen_imgs = [], []
        paths = []
        for cam_name in Cameras.NUSCENES_CAMERAS:
            cam_token = sample_record['data'][cam_name]
            cam_record = dataset.nusc.get('sample_data', cam_token)
            path = cam_record["filename"]

            if not exists(dataset_dir / 'gt' / path) or not exists(dataset_dir / 'gen' / path):
                print("Skipping", path)
                skip_sample = True
                break

            gt_imgs.append(to_tensor(Image.open(dataset_dir / 'gt' / path)))
            gen_imgs.append(to_tensor(Image.open(dataset_dir / 'gen' / path)))
            paths.append(path)
        
        if skip_sample:
            continue

        batch_gt_imgs.append(torch.stack(gt_imgs).to(device))
        batch_gen_imgs.append(torch.stack(gen_imgs).to(device))
        batch_paths.append(paths)
        batch_tokens.append(sample_token)

        batch_size = 6
        if (len(batch_gt_imgs) == batch_size) or (idx == len(image_idx) - 1 and len(batch_gt_imgs) > 0):
            gen, gt = torch.stack(batch_gen_imgs), torch.stack(batch_gt_imgs)
            gen, gt = rearrange(gen, 'b c ... -> (b c) ...'), rearrange(gt, 'b c ... -> (b c) ...')
            psim = lpips(gen, gt)
            psim = rearrange(psim, '(b c) -> b c', c=6)
            for i in range(psim.shape[0]):
                if (psim[i, :] > 0.58).all().item() or psim[i, :].mean().item() > 0.59:
                    samples_kept += 1
                    
                    for path in batch_paths[i]:
                        os.makedirs((output_dir / 'gt' / path).parent, exist_ok=True)
                        os.makedirs((output_dir / 'gen' / path).parent, exist_ok=True)
                        shutil.copy(dataset_dir / 'gt' / path, output_dir / 'gt' / path)
                        shutil.copy(dataset_dir / 'gen' / path, output_dir / 'gen' / path)

                    shutil.copytree(dataset_dir / 'sample_gt' / batch_tokens[i], output_dir / 'sample_gt' / batch_tokens[i])
                    shutil.copytree(dataset_dir / 'sample' / batch_tokens[i], output_dir / 'sample' / batch_tokens[i])
                else:
                    samples_removed += 1

            batch_gt_imgs, batch_gen_imgs, batch_paths, batch_tokens = [], [], [], []

        if samples_kept + samples_removed > 0:
            pbar.set_postfix(kept=f"{(samples_kept / (samples_kept + samples_removed)) * 100:.2f}%")

    print(f"Removed {samples_removed} samples, kept {samples_kept}, {samples_removed / len(images) * 100:.2f}%")

app = typer.Typer(pretty_exceptions_show_locals=False)

def split_range(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

@app.command()
def main(
    dataset_dir: Path = (SAVE_DATA_DIR / "baseline" / "multi_view_simple"),
    mini_dataset: bool = False,
    rec_dataset: bool = False):  

    dataset = NuScenesDataset(**NUSCENES_CONFIG)
    images = dataset.images

    # copy_imgs(range(len(images)), dataset_dir)
    # exit()
    # compute_overlap(range(len(images)), dataset_dir)
    
    num_processes = 6
    with Pool(num_processes) as p:
        num_arr = list(range(len(images)))
        random.shuffle(num_arr)
        for _ in p.imap_unordered(
            partial(
                compute_overlap,
                dataset_dir=dataset_dir,
            ),
            split_range(num_arr, num_processes),
        ):
            pass

if __name__ == "__main__":
    app()