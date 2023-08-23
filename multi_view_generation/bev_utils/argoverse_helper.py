import hashlib
from os.path import exists
from typing import Dict, Optional

import numpy as np
import pandas as pd
from av2.datasets.sensor.utils import convert_path_to_named_record
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.structures.cuboid import CuboidList
from av2.structures.timestamped_image import TimestampedImage
from av2.utils.constants import HOME
from av2.utils.io import read_feather, read_img
from av2.datasets.sensor.constants import AnnotationCategories
from rich.progress import track
from pathlib import Path
from enum import Enum, unique
import torch
from PIL import Image

@unique
class StandardCategories(str, Enum):
    """Sensor dataset annotation categories."""

    VEHICLE = "VEHICLE"
    LARGE_VEHICLE = "LARGE_VEHICLE"
    PEDESTRIAN = "PEDESTRIAN"
    OTHER = "OTHER"


standard_to_argoverse: Dict[StandardCategories, AnnotationCategories] = {
    StandardCategories.VEHICLE: (AnnotationCategories.REGULAR_VEHICLE,),
    StandardCategories.LARGE_VEHICLE: (
        AnnotationCategories.ARTICULATED_BUS,
        AnnotationCategories.BOX_TRUCK,
        AnnotationCategories.BUS,
        AnnotationCategories.LARGE_VEHICLE,
        AnnotationCategories.TRAFFIC_LIGHT_TRAILER,
        AnnotationCategories.TRUCK,
        AnnotationCategories.TRUCK_CAB,
        AnnotationCategories.VEHICULAR_TRAILER,
    ),
    StandardCategories.PEDESTRIAN: (AnnotationCategories.PEDESTRIAN,),
}

standard_colormap = np.array([[255, 0, 0], [255, 0, 127], [0, 255, 0], [255, 255, 255]], dtype=np.uint8)


argoverse_to_standard: Dict[AnnotationCategories, StandardCategories] = {argo_cat: k for k, v in standard_to_argoverse.items() for argo_cat in v}


def argo_to_standard_category(img):
    standard_imgs = {k: [] for k in StandardCategories}
    for i in range(len(AnnotationCategories)):
        argo_cat = list(AnnotationCategories)[i]
        if argo_cat in argoverse_to_standard:
            standard_imgs[argoverse_to_standard[argo_cat]].append(img[..., i])
        else:
            standard_imgs[StandardCategories.OTHER].append(img[..., i])
    return [np.logical_or.reduce(v) for _, v in standard_imgs.items()]


def argoverse_to_standard_img(img):
    return np.stack([*argo_to_standard_category(img), img[..., len(AnnotationCategories)]], axis=-1)

def get_closest(array, values):
    # make sure array is a numpy array
    array = np.array(array)

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = (idxs == len(array)) | (np.fabs(values - array[np.maximum(idxs - 1, 0)]) < np.fabs(values - array[np.minimum(idxs, len(array) - 1)]))
    idxs[prev_idx_is_less] -= 1
    return array[idxs]

def populate_image_records(cam_names, sensor_dir: Path, split: str, ignore_pickle=False, bev_dir: Path = None) -> pd.DataFrame:
    """Obtain (log_id, sensor_name, timestamp_ns) 3-tuples for all images in the dataset.

    Returns:
        DataFrame of shape (N,3) with `log_id`, `sensor_name`, and `timestamp_ns` columns.
            N is the total number of images for all logs in the dataset, and the `sensor_name` column
            should be populated with the name of the camera that captured the corresponding image in
            every entry.
    """

    av2_split_dir = sensor_dir / split
    av2_bev_dir = bev_dir / split if bev_dir is not None else None
    sha = hashlib.sha1(f"{av2_split_dir.resolve()}_{cam_names}_{bev_dir}".encode("utf-8"))
    pickle_name = HOME / ".cache" / "av2" / f"{sha.hexdigest()}.p"
    pickle_name.parent.mkdir(parents=True, exist_ok=True)

    if exists(pickle_name) and not ignore_pickle:
        print(f"Loading from file: {pickle_name}")
        cam_records = pd.read_pickle(pickle_name)
    else:
        if bev_dir:
            all_bev_images = []
            all_cam_images = []
            for cam_name in cam_names:
                all_bev_images.extend(
                    [
                        str(av2_split_dir / str(name.resolve()).replace(".npz", ".jpg").partition(str(av2_bev_dir.resolve()))[2][1:])
                        for name in av2_bev_dir.glob(f"**/sensors/cameras/{cam_name}/*.npz")
                    ]
                )
                all_cam_images.extend([str(name.resolve()) for name in av2_split_dir.glob(f"**/sensors/cameras/{cam_name}/*.jpg")])

            all_cam_images = [Path(p) for p in set(all_cam_images).intersection(set(all_bev_images))]
        else:
            # Get sorted list of camera paths.
            all_cam_images = []
            for cam_name in cam_names:
                all_cam_images.extend(av2_split_dir.glob(f"**/sensors/cameras/{cam_name}/*.jpg"))

        cam_paths = sorted(all_cam_images, key=lambda x: int(x.stem))

        # Load entire set of camera records.
        cam_record_list = [convert_path_to_named_record(x) for x in track(cam_paths, description="Loading camera records ...")]

        # Concatenate into single dataframe (list-of-dicts to DataFrame).
        cam_records = pd.DataFrame(cam_record_list)
        cam_records.to_pickle(pickle_name.resolve())
        print(f"Writing to file with params: {pickle_name.resolve()}, \n {av2_split_dir.resolve()}_{cam_names}_{bev_dir}")

    return cam_records


def load_synchronized_cams(dataset_dir, split: str, log_id: str, cam_name: str, sweep_timestamp_ns: int) -> Optional[Dict[str, TimestampedImage]]:
    """Load the synchronized imagery for a lidar sweep.

    Args:
        split: Dataset split.
        sensor_dir: Sensor directory.
        log_id: Log unique id.
        sweep_timestamp_ns: Nanosecond timestamp.

    Returns:
        Mapping between camera names and synchronized images.

    Raises:
        RuntimeError: if the synchronization database (sync_records) has not been created.
    """

    sensor_dir = dataset_dir / split / log_id / "sensors"
    p = sensor_dir / "cameras" / cam_name / f"{sweep_timestamp_ns}.jpg"

    log_dir = sensor_dir.parent
    cam_img = TimestampedImage(
        img=read_img(p, channel_order="RGB"), camera_model=PinholeCamera.from_feather(log_dir=log_dir, cam_name=cam_name), timestamp_ns=int(p.stem),
    )
    return cam_img


def load_annotations(dataset_dir, split: str, log_id: str, sweep_timestamp_ns: int) -> CuboidList:
    """Load the sweep annotations at the provided timestamp.

    Args:
        split: Split name.
        log_id: Log unique id.
        sweep_timestamp_ns: Nanosecond timestamp.

    Returns:
        Cuboid list of annotations.
    """
    annotations_feather_path = dataset_dir / split / log_id / "annotations.feather"

    data = read_feather(annotations_feather_path)
    timestamps = np.unique(data.loc[:, "timestamp_ns"].to_numpy())
    annotation_timestamp = get_closest(timestamps, [sweep_timestamp_ns]).item()

    # Load annotations from disk.
    # NOTE: This contains annotations for the ENTIRE sequence.
    # The sweep annotations are selected below.
    cuboid_list = CuboidList.from_feather(annotations_feather_path)
    cuboids = list(filter(lambda x: x.timestamp_ns == annotation_timestamp, cuboid_list.cuboids))
    return CuboidList(cuboids=cuboids), annotation_timestamp


def make_grid2d(grid_size, grid_offset, grid_res):
    """
    Constructs an array representing the corners of an orthographic grid
    """
    depth, width = grid_size
    xoff, zoff = grid_offset
    xcoords = torch.arange(0.0, width, grid_res) + xoff
    zcoords = torch.arange(0.0, depth, grid_res) + zoff

    zz, xx = torch.meshgrid(zcoords, xcoords)
    return torch.stack([xx, zz], dim=-1)