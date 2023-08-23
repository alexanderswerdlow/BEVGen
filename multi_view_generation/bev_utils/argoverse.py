import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import torchvision.transforms as T
from einops import rearrange
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import ImageStat, Image
from torch.utils.data import DataLoader, Dataset
from enum import Enum
import albumentations as A
from numpy.linalg import inv
import pandas as pd
from tqdm import tqdm
from av2.datasets.sensor.constants import RingCameras
import torchvision.transforms.functional as F
from collections.abc import Iterable
from multi_view_generation.bev_utils import Cameras, raw_output_data_bev_grid, ARGOVERSE_DIR
from multi_view_generation.bev_utils.argoverse_multi_sensor_dataloader import MultiSensorData, MultiSynchronizedSensorData
from multi_view_generation.bev_utils.argoverse_sensor_dataloader import SensorDataloader
from multi_view_generation.bev_utils.argoverse_helper import load_synchronized_cams, populate_image_records
from multi_view_generation.bev_utils.nuscenes_helper import NusceneCamGeometry
import os
import random
import pickle

class Split(Enum):
    TRAIN = 0
    VAL = 1


class Argoverse(Dataset):
    def __init__(
        self,
        split: Split,
        return_cam_img: bool = True,
        return_bev_img: bool = False,
        return_calib: bool = False,
        return_meta: bool = False,
        merge_val: bool = False,
        merge_test: bool = False,
        dataset_dir: Path = ARGOVERSE_DIR,
        bev_dir_name: Path = "bev_seg_full_11_14", # Contains pre-generated bev representation
        cam_res: int = (256, 336),
        augment_cam_img: bool = False,
        augment_bev_img: bool = False,
        flip_imgs: bool = False,
        normalize_cam_img: bool = True,
        multi_camera: bool = False,
        specific_cameras: Union[bool, List[str]] = False,
        specific_frames: Union[bool, Path] = False,
        only_keyframes: bool = False,
        num_same_instances: Optional[int] = None,
        raw_data: bool = False,
        square_image: bool = False,
        **kwargs,
    ):
        self.split = Split(split)
        self.return_cam_img = return_cam_img
        self.return_bev_img = return_bev_img
        self.return_calib = return_calib
        self.dataset_dir = Path(dataset_dir)
        self.sensor_dir = self.dataset_dir / 'sensor'
        self.bev_dir = self.dataset_dir / bev_dir_name
        self.return_meta = return_meta
        self.flip_imgs = flip_imgs
        self.multi_camera = multi_camera
        self.kwargs = kwargs
        self.only_keyframes = only_keyframes
        self.specific_frames = specific_frames
        self.num_same_instances = num_same_instances
        self.raw_data = raw_data
        self.square_image = square_image

        if self.specific_frames:
            if os.path.exists(self.specific_frames):
                with open(self.specific_frames, "rb") as f:
                    self.specific_frames = pickle.load(f)
            else:
                raise Exception()

        if not specific_cameras:
            self.cameras = list(RingCameras)
        else:
            self.cameras = list(map(lambda x: RingCameras(x), specific_cameras))
        
        # Some Argoverse methods want the actual enum, not the string name and vice-versa
        self.camera_names = list(map(lambda x: x.value, self.cameras))
        self.augment_cam_img = augment_cam_img
        self.cam_transform = None
        self.cam_res = cam_res if isinstance(cam_res, Iterable) else (cam_res, cam_res)
        self.cam_intrinsic_aug = NusceneCamGeometry()

        if self.multi_camera:
            if normalize_cam_img:
                self.cam_transform = T.Compose([
                    T.Normalize(
                        mean=(0.4265, 0.4489, 0.4769),
                        std=(0.2053, 0.2206, 0.2578),
                    ),
                ])
            else:
                self.cam_transform = T.Compose([])

            if augment_bev_img:
                self.bev_transform = A.Compose([A.ShiftScaleRotate(shift_limit=0.001, scale_limit=0.01, rotate_limit=0, p=0.5), A.HorizontalFlip(p=0.5)])
            else:
                self.bev_transform = A.Compose([])

            # Allows us to obtain data between keyframes
            self.dataset = MultiSensorData(
                dataset_dir=self.sensor_dir, split=self.split.name.lower(), with_annotations=True, with_cache=True, cam_names=self.cameras, all_camera_frames=(not self.specific_frames and not only_keyframes)
            )

            self.color_transform = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        else:
            # cam_records contains records for each camera frame
            if merge_test or merge_val:
                splits_to_use = [self.split.name.lower()]
                if merge_test:
                    splits_to_use.append("test")

                if merge_val:
                    splits_to_use.append("val")

                self.cam_records = pd.concat(
                    [populate_image_records(self.camera_names, self.sensor_dir, split_dir, ignore_pickle=False) for split_dir in splits_to_use], ignore_index=False
                )

            else:
                self.cam_records = populate_image_records(
                    self.camera_names,
                    self.sensor_dir,
                    self.split.name.lower(),
                    ignore_pickle=False,
                    bev_dir=None,
                )

            self.dataset = SensorDataloader(dataset_dir=self.sensor_dir, split=self.split.name.lower(), with_annotations=True, with_cache=True, cam_names=self.cameras)

            if self.augment_cam_img:
                self.cam_transform = T.Compose([
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(0),
                ])
            else:
                self.cam_transform = T.Compose([])

            self.cam_transform = T.Compose([self.cam_transform,
                                            T.Normalize(
                                                mean=(0.4265, 0.4489, 0.4769),
                                                std=(0.2053, 0.2206, 0.2578),
                                            ),
                                            ])

            if augment_bev_img:
                self.bev_transform = A.Compose([self.bev_transform, A.ShiftScaleRotate(shift_limit=0.075, scale_limit=0.075, rotate_limit=10, p=0.5)])
            else:
                self.bev_transform = A.Compose([])

            if return_bev_img:
                # We require that all cameras have a corresponding lidar scan and hence BEV representation
                df = self.dataset.synchronization_cache.reset_index()
                rows_to_remove = df[df.lidar.isnull()]
                for cam_name_ in self.cameras:
                    cam_nat_records = (
                        rows_to_remove[rows_to_remove.sensor_name == cam_name_][["split", "log_id", "sensor_name", cam_name_]]
                        .rename(columns={cam_name_: "timestamp_ns"})
                        .astype({"timestamp_ns": "int64"})
                    )
                    self.cam_records = pd.merge(self.cam_records, cam_nat_records, indicator=True, how="outer").query('_merge=="left_only"').drop("_merge", axis=1)

        print(f'Argoverse has {len(self)} samples')

    def process_img(self, img, color_params=None):
        output_height, output_width = self.cam_res

        self.cam_intrinsic_aug = NusceneCamGeometry()
        self.cam_intrinsic_aug.rescale_first = False
        im_to_process = T.ToPILImage()(img)
        if self.augment_cam_img:
            if color_params is not None:
                fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = color_params
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness_factor is not None:
                        im_to_process = F.adjust_brightness(im_to_process, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        im_to_process = F.adjust_contrast(im_to_process, contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        im_to_process = F.adjust_saturation(im_to_process, saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        im_to_process = F.adjust_hue(im_to_process, hue_factor)

            if self.multi_camera:
                scale_max = 0.1
            else:
                scale_max = 0.25
            desired_scale = np.random.uniform(1.0 - scale_max, 1.0)
            new_width, new_height = int(im_to_process.size[0] * desired_scale), int(im_to_process.size[1] * desired_scale)
            i, j, h, w = T.RandomCrop.get_params(im_to_process, (new_height, new_width))
            im_to_process = F.crop(im_to_process, i, j, h, w)
            self.cam_intrinsic_aug.set_scale(output_width/new_width, output_height/new_height)
            self.cam_intrinsic_aug.set_crop(i, j)
        else:
            self.cam_intrinsic_aug.set_scale(output_width/im_to_process.size[0], output_height/im_to_process.size[1])

        im_to_process = im_to_process.resize((output_width, output_height), resample=Image.BICUBIC)
        im_to_process = F.to_tensor(im_to_process)
        im_to_process = self.cam_transform(im_to_process)
        return np.asarray(rearrange(im_to_process, 'c h w -> h w c'))

    def __getitem__(self, idx: int):
        """Load the lidar point cloud and optionally the camera imagery and annotations.

        Grab the lidar sensor data and optionally the camera sensor data and annotations at the lidar record
            corresponding to the specified index.

        Args:
            idx: Index in [0, self.num_sweeps - 1].

        Returns:
            Mapping from sensor name to data for the lidar record corresponding to the specified index.
        """

        ret = {"dataset": "argoverse"}

        if self.multi_camera:
            if self.specific_frames:
                if self.num_same_instances is not None:
                    idx = idx % self.num_same_instances
                data: MultiSynchronizedSensorData = self.dataset.get_from_record(*self.specific_frames[idx])
            elif self.only_keyframes:
                data: MultiSynchronizedSensorData = self.dataset.regular_get_item(idx)
            else:
                data: MultiSynchronizedSensorData = self.dataset[idx]

            log_id, timestamp_ns, split, lidar_timestamp_ns = (data.log_id, data.timestamp_ns, data.split, data.lidar_timestamp_ns)
            if len(data.synchronized_imagery) != len(self.cameras):
                raise ValueError()

            if self.return_bev_img:
                bev_image_path = self.dataset_dir / self.bev_dir / split / log_id / f"{lidar_timestamp_ns}.npz"
                bev_image = np.load(bev_image_path)
                bev_img = self.bev_transform(image=bev_image.f.arr_0)["image"].astype(np.float32)
                ret["segmentation"] = bev_img

            if self.raw_data:
                ret["data"] = data
                # ret["raw_images"] = data.synchronized_imagery
                ret["weight"] = sum([ 1 / (np.linalg.norm(x.xyz_center_m)**2) * np.prod(x.dims_lwh_m) for x in data.annotations])
                ret["key"] = (split, log_id, timestamp_ns)
                

            if self.kwargs.get('fake_load', False):
                return {}

            if self.return_cam_img:  # 1550, 2048, 3 (uint8, numpy) 256, 336, h, w, c
                try:
                    imgs = np.stack([data.synchronized_imagery[k.value].img.transpose(1, 0, 2) if k.value == 'ring_front_center' else data.synchronized_imagery[k.value].img for k in self.cameras], axis=0)
                except:
                    print(log_id, timestamp_ns, split)

                color_params = T.ColorJitter.get_params(brightness=self.color_transform.brightness, contrast=self.color_transform.contrast, saturation=self.color_transform.saturation, hue=self.color_transform.hue)

                transformed_imgs, intrinsics, extrinsics, intrinsics_inv, extrinsics_inv = [], [], [], [], []
                for cam_name, img in zip(self.camera_names, imgs):
                    if self.square_image:
                        if cam_name in "ring_front_left":
                            img = img[:, img.shape[1] - img.shape[0]:]
                        elif cam_name == "ring_front_right":
                            img = img[:, :-(img.shape[1] - img.shape[0])]
                        elif cam_name == "ring_front_center":
                            img = img.transpose(1, 0, 2)[(img.shape[1] - img.shape[0]):]
                        else:
                            raise Exception()

                    transformed_imgs.append(self.process_img(img, color_params))
                    cam_intrinsic = data.synchronized_imagery[cam_name].camera_model.intrinsics.K
                    cam_intrinsic = np.float32(self.cam_intrinsic_aug.apply(cam_intrinsic))
                    cam_extrinsic = data.synchronized_imagery[cam_name].camera_model.ego_SE3_cam.transform_matrix

                    intrinsics.append(cam_intrinsic)
                    intrinsics_inv.append(inv(cam_intrinsic))

                    extrinsics.append(cam_extrinsic)
                    extrinsics_inv.append(inv(cam_extrinsic))

                imgs = np.stack(transformed_imgs, axis=0)
                assert imgs.shape[0] == len(self.cameras)
                ret["image"] = imgs
                ret['intrinsics'] = np.stack(intrinsics).astype(np.float32)
                ret['extrinsics'] = np.stack(extrinsics).astype(np.float32)
                ret['intrinsics_inv'] = np.stack(intrinsics_inv).astype(np.float32)
                ret['extrinsics_inv'] = np.stack(extrinsics_inv).astype(np.float32)

            ret['cam_name'] = self.camera_names
            ret['sample_token'] = f'{data.log_id}_{data.timestamp_ns}'

        else:
            idx = idx.item() if torch.is_tensor(idx) else idx
            record: Tuple[str, str, str, int] = self.cam_records.iloc[idx]
            split, log_id, cam_name, timestamp_ns = record

            if self.return_cam_img:
                cam_img = load_synchronized_cams(self.sensor_dir, split, log_id, cam_name, timestamp_ns)
                im_to_process = cam_img.img
                im_to_process = im_to_process.transpose(1, 0, 2) if cam_img.camera_model.cam_name == 'ring_front_center' else im_to_process
                ret["image"] = self.process_img(im_to_process)[None, ...]

            if self.return_bev_img:
                lidar_path = self.dataset.get_closest_lidar_fpath(split, log_id, cam_name, timestamp_ns)
                bev_image_path = self.dataset_dir / self.bev_dir / split / log_id / f"{lidar_path.stem}.npz"
                bev_image = np.load(bev_image_path)
                bev_img = self.bev_transform(image=bev_image.f.arr_0)["image"].astype(np.float32)
                ret["segmentation"] = bev_img

            if self.return_calib:
                ret["calib"] = cam_img.camera_model

            if self.flip_imgs:
                if "segmentation" in ret:
                    ret["segmentation"] = A.hflip(ret["segmentation"])

            ret['cam_name'] = [cam_name]

        return ret

    def get_index(self, record):
        return self.cam_records.loc[(self.cam_records[list(record)] == pd.Series(record)).all(axis=1)].index.item()

    def __len__(self):
        if self.specific_frames:
            if self.num_same_instances is not None:
                return len(self.specific_frames) * self.num_same_instances
            return len(self.specific_frames)
        elif self.multi_camera:
            return len(self.dataset)
        else:
            return len(self.cam_records)

    def viz_item(self, batch: dict, idx):
        imgs = raw_output_data_bev_grid(batch)
        for i in range(len(imgs)):
            os.makedirs(f'output', exist_ok=True)
            imgs[i].save(f'output/{idx}_{i}.png')

    def save_cam_data(self, batch: dict):
        torch.save({k: v for k, v in batch.items() if "intrinsics" in k or "extrinsics" in k}, 'cam_data_argoverse.pt')


class Stats(ImageStat.Stat):
    def __add__(self, other):
        return Stats(list(np.add(self.h, other.h)))


def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

    Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for i, data in tqdm(enumerate(loader)):
        data = data.permute(0, 3, 1, 2)
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

        if i % 100 == 0:
            print(fst_moment, torch.sqrt(snd_moment - fst_moment ** 2))

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

def find_interesting():
    STAGE_2_ONLY_KEYFRAMES = {
        'split': 1,
        'return_cam_img': True,
        'return_bev_img': True,
        'multi_camera': True,
        'specific_cameras': Cameras.ARGOVERSE_CAMERAS,
        'only_keyframes': True
    }

    dataset = Argoverse(**STAGE_2_ONLY_KEYFRAMES)
    batch_iter = DataLoader(dataset, batch_size=12, num_workers=12, shuffle=False, pin_memory=True)
    weights = []
    splits, log_ids, timestamp_nss = [], [], []
    for i, data in enumerate(tqdm(batch_iter)):
        weights.extend(data['weight'].tolist())
        splits.extend(data['key'][0])
        log_ids.extend(data['key'][1])
        timestamp_nss.extend(data['key'][2])
    comb_instances = tuple(zip(splits, log_ids, timestamp_nss))
    image_tokens = random.choices(comb_instances, weights, k=500)
    image_tokens = list(map(lambda x: (x[0], x[1], x[2].item()), image_tokens))
    frozen_image_tokens = 'data/interesting_argoverse_image_tokens.pkl'
    with open(frozen_image_tokens, "wb") as f:
        pickle.dump(image_tokens, f)
    exit()

def get_raw_data(config):
    dataset = Argoverse(**config)
    indexes = list(range(len(dataset)))
    random.shuffle(indexes)
    for i in indexes:
        data = dataset[i]
        from multi_view_generation.scripts.argoverse_ground_truth_viz import run_data
        run_data(data, i)
    exit()

if __name__ == "__main__":
    # find_interesting()
    STAGE_2_TRAIN = {
        'split': 1,
        'return_cam_img': True,
        'return_bev_img': True,
        'multi_camera': True,
        'augment_cam_img': False,
        'specific_cameras': Cameras.ARGOVERSE_FRONT_CAMERAS,
        'fake_load': False,
        'square_image': True,
        'cam_res': (256, 256),
    }

    STAGE_2_ONLY_KEYFRAMES = {
        **STAGE_2_TRAIN,
        'only_keyframes': True
    }

    STAGE_2_INTERESTING = {
        **STAGE_2_TRAIN,
        'specific_frames': 'data/interesting_argoverse_image_tokens.pkl',
        'cam_res': (512, 672),
    }

    STAGE_2_ALL_DATASET = {
        **STAGE_2_TRAIN,
        'split': 0,
        'specific_frames': False,
        'specific_cameras': False,
        'raw_data': True,
    }

    STAGE_1_TRAIN = {
        'split': 1,
        'return_cam_img': True,
        'return_bev_img': True,
        'multi_camera': False,
        'augment_cam_img': True,
        'specific_cameras': Cameras.ARGOVERSE_CAMERAS
    }

    STAGE_1_VAL = {
        'split': 1,
        'return_cam_img': True,
        'return_bev_img': True,
        'multi_camera': False,
        'specific_cameras': Cameras.ARGOVERSE_CAMERAS
    }

    # get_raw_data(STAGE_2_ALL_DATASET)

    dataset = Argoverse(**STAGE_2_TRAIN)
    batch_iter = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=True, pin_memory=False)
    for i, data in enumerate(batch_iter):
        dataset.viz_item(data, i)
        breakpoint()
        from multi_view_generation.scripts.argoverse_ground_truth_viz import run_data
        run_data(data, i)
        