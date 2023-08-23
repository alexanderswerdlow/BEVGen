from os.path import exists
from pathlib import Path
import hashlib

import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import torch
import typer
from kornia.feature import LoFTR
from kornia_moons.feature import *
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from multi_view_generation.bev_utils import SAVE_DATA_DIR


class GeneratedDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        mini_dataset: bool = False,
        rec_dataset: bool = False,
        **kwargs,
    ):
        self.mini_dataset = mini_dataset
        self.dataset_dir = dataset_dir
        self.comp_dataset = 'rec' if rec_dataset else 'gen'
        gt_samples = list((dataset_dir / 'sample_gt').glob('*'))
        gen_samples = list((dataset_dir / 'sample').glob('*'))
        max_samples = max(len(gt_samples), len(gen_samples))

        gt_samples = set([gt_path.relative_to(Path(dataset_dir / 'sample_gt')) for gt_path in gt_samples])
        gen_samples = set([gen_path.relative_to(Path(dataset_dir / 'sample')) for gen_path in gen_samples])
        self.samples = list(gt_samples.intersection(gen_samples))
        if (min_removed := max_samples - len(self.samples)) > 0:
            print(f'Removed at least {min_removed}')

        sha_gt = hashlib.sha1(f"{','.join(sorted(list(map(lambda x: str(x), gt_samples))))}".encode("utf-8")).hexdigest()
        sha_gen = hashlib.sha1(f"{','.join(sorted(list(map(lambda x: str(x), gen_samples))))}".encode("utf-8")).hexdigest()
        assert sha_gt == sha_gen

        self.samples = list(self.samples)

    def __getitem__(self, idx):
        gt = Path(self.dataset_dir) / 'sample_gt' / self.samples[idx]
        ret = Path(self.dataset_dir) / 'sample' / self.samples[idx]
        cams = ("CAM_FRONT", "CAM_BACK", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK_RIGHT", "CAM_BACK_LEFT")

        gt = torch.stack([to_tensor(Image.open((gt / cam_name).with_suffix('.jpg'))) for cam_name in cams])
        ret = torch.stack([to_tensor(Image.open((ret / cam_name).with_suffix('.jpg'))) for cam_name in cams])

        return gt, ret

    def __len__(self):
        if self.mini_dataset:
            return 10
        else:
            return len(self.samples)

    def get_all_image_paths(self):
        ret_samples = self.samples[:10] if self.mini_dataset else self.samples
        return list(map(lambda x: Path(self.dataset_dir) / 'sample_gt' / x, ret_samples)), list(map(lambda x: Path(self.dataset_dir) / 'sample' / x, ret_samples))


def save_ax(ax, filename, **kwargs):
    ax.axis("off")
    ax.figure.canvas.draw()
    trans = ax.figure.dpi_scale_trans.inverted()
    bbox = ax.bbox.transformed(trans)
    plt.savefig(filename, dpi="figure", bbox_inches=bbox,  **kwargs)
    ax.axis("on")
    im = plt.imread(filename)
    return im


def viz_correspondences(img1, img2, correspondences, img_name):
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    try:
        Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
        inliers = inliers > 0
    except:
        return

    ax = draw_LAF_matches(
        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1, -1, 2),
                                     torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                                     torch.ones(mkpts0.shape[0]).view(1, -1, 1)),

        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1, -1, 2),
                                     torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                                     torch.ones(mkpts1.shape[0]).view(1, -1, 1)),
        torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
        K.tensor_to_image(img1),
        K.tensor_to_image(img2),
        inliers,
        draw_dict={'inlier_color': (0.2, 1, 0.2),
                   'tentative_color': None,
                   'feature_color': (0.2, 0.5, 1), 'vertical': False}, return_axis=True)
    save_ax(ax, img_name)


def get_edge_window(left_img, right_img, num_pix):
    return left_img[:, :, -num_pix:], right_img[:, :, :num_pix]


def compute_overlap(dataset: GeneratedDataset, visualize: bool):
    gt_sum, gen_sum = 0, 0

    device = 'cuda:0'
    matcher = LoFTR(pretrained="outdoor").to('cuda:0')

    gt_paths, ret_paths = dataset.get_all_image_paths()
    for idx, path_ in tqdm(enumerate(ret_paths)):
        for dataset_type in (0, 1):
            if dataset_type == 0:
                path = path_.parent.parent / 'sample_gt' / path_.name
            else:
                path = path_

            if exists((Path(path) / "CAM_BACK").with_suffix('.jpg')):
                imgs = {cam_name: to_tensor(Image.open((Path(path) / cam_name).with_suffix('.jpg')))
                        for cam_name in ("CAM_FRONT", "CAM_BACK", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK_RIGHT", "CAM_BACK_LEFT")}

                pairs = (("CAM_FRONT_LEFT", "CAM_FRONT"), ("CAM_FRONT", "CAM_FRONT_RIGHT"), ("CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"),
                         ("CAM_BACK_RIGHT", "CAM_BACK"), ("CAM_BACK", "CAM_BACK_LEFT"), ("CAM_BACK_LEFT", "CAM_FRONT_LEFT"))
            elif exists((Path(path) / "ring_front_center").with_suffix('.jpg')):
                imgs = {cam_name: to_tensor(Image.open((Path(path) / cam_name).with_suffix('.jpg'))) for cam_name in ("ring_front_center", "ring_front_right", "ring_front_left")}

                pairs = (("ring_front_left", "ring_front_center"), ("ring_front_center", "ring_front_right"))
            else:  # For ablations, we only have front 3 cameras
                imgs = {cam_name: to_tensor(Image.open((Path(path) / cam_name).with_suffix('.jpg'))) for cam_name in ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT")}

                pairs = (("CAM_FRONT_LEFT", "CAM_FRONT"), ("CAM_FRONT", "CAM_FRONT_RIGHT"))

            confidence_sum = 0
            for i, j in pairs:
                img1, img2 = get_edge_window(imgs[i], imgs[j], 50)
                img1, img2 = img1[None], img2[None]

                input = {"image0": K.color.rgb_to_grayscale(img1).to(device), "image1": K.color.rgb_to_grayscale(img2).to(device)}
                with torch.inference_mode():
                    correspondences = matcher(input)
                    confidence_sum += correspondences['confidence'].sum()
                    
                if visualize:
                    viz_correspondences(img1, img2, correspondences, f"{path.name}_{dataset_type}_{i}_{j}.png")

                if dataset_type == 0:
                    gt_sum += confidence_sum / len(pairs)
                else:
                    gen_sum += confidence_sum / len(pairs)

        print(f"GT: {gt_sum / (idx + 1)}, Gen: {gen_sum / (idx + 1)}")


def custom_collate(batch):
    gt, ret = zip(*batch)
    return torch.cat(gt), torch.cat(ret)


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    dataset_dir: Path = (SAVE_DATA_DIR / "baseline" / "multi_view_simple"),
    mini_dataset: bool = False,
    rec_dataset: bool = False,
    visualize: bool = False
):
    dataset = GeneratedDataset(dataset_dir=dataset_dir, mini_dataset=mini_dataset, rec_dataset=rec_dataset)
    compute_overlap(dataset, visualize)


if __name__ == "__main__":
    app()
