# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Example script for loading data from the AV2 sensor dataset."""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from pathlib import Path
from typing import Final, List, Tuple, Union

import click
import numpy as np
from rich.progress import track
from PIL import Image

from multi_view_generation.bev_utils.util import Cameras
from av2.datasets.sensor.constants import RingCameras, StereoCameras
from av2.datasets.sensor.sensor_dataloader import SensorDataloader
from av2.rendering.color import ColorFormats, create_range_map
from av2.rendering.rasterize import draw_points_xy_in_img
from av2.rendering.video import write_video
from av2.structures.ndgrid import BEVGrid
from av2.utils.typing import NDArrayByte, NDArrayInt
from enum import Enum, unique
from pathlib import Path
from typing import Dict, Final, Mapping, Optional, Set, Union
from multi_view_generation.bev_utils import viz_bev, Im
from multi_view_generation.bev_utils import Dataset
from multi_view_generation.modules.transformer.mingpt_sparse import GPTConfig
import av
import numpy as np
import pandas as pd

from av2.rendering.color import ColorFormats
from av2.utils.typing import NDArrayByte


# Bird's-eye view parameters.
MIN_RANGE_M: Tuple[float, float] = (-102.4, -77.5)
MAX_RANGE_M: Tuple[float, float] = (+102.4, +77.5)
RESOLUTION_M_PER_CELL: Tuple[float, float] = (+0.1, +0.1)

# Model an xy grid in the Bird's-eye view.
BEV_GRID: Final[BEVGrid] = BEVGrid(
    min_range_m=MIN_RANGE_M, max_range_m=MAX_RANGE_M, resolution_m_per_cell=RESOLUTION_M_PER_CELL
)

def tile_cameras(
    named_sensors: Mapping[str, Union[NDArrayByte, pd.DataFrame]],
    bev_img: Optional[NDArrayByte] = None,
) -> NDArrayByte:
    """Combine ring cameras into a tiled image.

    NOTE: Images are expected in BGR ordering.

    Layout:

        ##########################################################
        # ring_front_left # ring_front_center # ring_front_right #
        ##########################################################
        # ring_side_left  #                   #  ring_side_right #
        ##########################################################
        ############ ring_rear_left # ring_rear_right ############
        ##########################################################

    Args:
        named_sensors: Dictionary of camera names to the (width, height, 3) images.
        bev_img: (H,W,3) Bird's-eye view image.

    Returns:
        Tiled image.
    """
    landscape_height = 2048
    landscape_width = 1550
    for _, v in named_sensors.items():
        landscape_width = max(v.shape[0], v.shape[1])
        landscape_height = min(v.shape[0], v.shape[1])
        break

    padding_ = 5
    height = landscape_width + landscape_height + landscape_height + padding_ + padding_# landscape_height + landscape_height + landscape_height
    width = landscape_width + landscape_height + landscape_width + padding_ + padding_
    tiled_im_bgr: NDArrayByte = 255 * np.ones((height, width, 3), dtype=np.uint8)

    padding = ((landscape_width + landscape_height) - (2 * landscape_height)) // 2
    

    if "ring_front_left" in named_sensors:
        ring_front_left = named_sensors["ring_front_left"]
        tiled_im_bgr[2 * padding:2 * padding + landscape_height, :landscape_width] = ring_front_left

    if "ring_front_center" in named_sensors:
        ring_front_center = named_sensors["ring_front_center"]
        tiled_im_bgr[:landscape_width, padding_ + landscape_width : padding_ + landscape_width + landscape_height] = ring_front_center

    if "ring_front_right" in named_sensors:
        ring_front_right = named_sensors["ring_front_right"]
        tiled_im_bgr[2 * padding:2 * padding + landscape_height, 2 * padding_ + landscape_width + landscape_height :] = ring_front_right

    if "ring_side_left" in named_sensors:
        ring_side_left = named_sensors["ring_side_left"]
        tiled_im_bgr[2 * padding + landscape_height + padding_: 2 * padding + 2 * landscape_height+ padding_, :landscape_width] = ring_side_left

    if "ring_side_right" in named_sensors:
        ring_side_right = named_sensors["ring_side_right"]
        tiled_im_bgr[2 * padding + landscape_height + padding_: 2 * padding + 2 * landscape_height+ padding_, 2 * padding_ + landscape_width + landscape_height :] = ring_side_right

    if bev_img is not None:
        tiled_im_bgr[
            landscape_width + padding_: landscape_width + landscape_height+ padding_, padding_ + landscape_width : padding_ + landscape_width + landscape_height
        ] = bev_img

    if "ring_rear_left" in named_sensors:
        ring_rear_left = named_sensors["ring_rear_left"]
        tiled_im_bgr[landscape_width + landscape_height +2 *  padding_ : landscape_width + landscape_height + landscape_height+2 *  padding_, padding_//2 + landscape_height//2:padding_//2 +landscape_width+landscape_height//2] = ring_rear_left

    if "ring_rear_right" in named_sensors:
        ring_rear_right = named_sensors["ring_rear_right"]
        tiled_im_bgr[landscape_width + landscape_height +2 *  padding_: landscape_width + landscape_height + landscape_height+2 *  padding_, padding_//2 + padding_ + landscape_width+landscape_height//2: padding_//2 + padding_ + landscape_width+landscape_height//2+landscape_width] = ring_rear_right
    return tiled_im_bgr

def white_to_transparency(img):
    x = np.asarray(img.convert('RGBA')).copy()

    x[:, :, 3] = (255 * (x[:, :, :3] != 255).any(axis=2)).astype(np.uint8)

    return Image.fromarray(x)

from PIL import Image, ImageDraw
def run_data(batch, idx):
    bev = viz_bev(batch['segmentation'], dataset=Dataset.ARGOVERSE).pil
    bev_width, bev_height = bev.size[0], bev.size[1]
    bev = ImageDraw.Draw(bev)
    bev.rectangle((bev_width // 2 - 4, bev_height // 2 - 8, bev_width // 2 + 4, bev_height // 2 + 8), fill="#00FF11")
    bev = bev._image
    bev = Im(bev.resize((1550, 1550), Image.Resampling.LANCZOS)).np

    img_dict = {cam_name: cam.img for cam_name, cam in batch['raw_images'].items()}
    tiled_img = tile_cameras(img_dict, bev_img=bev)
    tiled_img = white_to_transparency(Image.fromarray(tiled_img))
    tiled_img = tiled_img.resize((tiled_img.size[0]//2,tiled_img.size[1]//2), Image.Resampling.LANCZOS)
    file_path = Path(f'output/{idx}.png')
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tiled_img.save(file_path, quality=95)

def generate_sensor_dataset_visualizations(
    dataset_dir: Path,
    with_annotations: bool,
    cam_names: Tuple[Union[RingCameras, StereoCameras], ...],
) -> None:
    """Create a video of a point cloud in the ego-view. Annotations may be overlaid.

    Args:
        dataset_dir: Path to the dataset directory.
        with_annotations: Boolean flag to enable loading of annotations.
        cam_names: Set of camera names to render.
    """
    dataset = SensorDataloader(
        dataset_dir,
        with_annotations=True,
        with_cache=True,
        cam_names=cam_names,
    )

    tiled_cams_list: List[NDArrayByte] = []
    for idx, datum in enumerate(dataset):
        if idx % 10 != 0:
            continue
        sweep = datum.sweep
        annotations = datum.annotations
        timestamp_city_SE3_ego_dict = datum.timestamp_city_SE3_ego_dict
        synchronized_imagery = datum.synchronized_imagery
        if synchronized_imagery is not None:
            cam_name_to_img = {}
            for cam_name, cam in synchronized_imagery.items():
                rot = datum.synchronized_imagery[cam_name].camera_model.ego_SE3_cam.rotation; v = np.dot(rot, np.array([1, 0, 0])); print(cam_name, np.rad2deg(np.arctan2(v[1], v[0])))
                
                if (
                    cam.timestamp_ns in timestamp_city_SE3_ego_dict
                    and sweep.timestamp_ns in timestamp_city_SE3_ego_dict
                ):
                    city_SE3_ego_cam_t = timestamp_city_SE3_ego_dict[cam.timestamp_ns]
                    city_SE3_ego_lidar_t = timestamp_city_SE3_ego_dict[sweep.timestamp_ns]

                    uv, points_cam, is_valid_points = cam.camera_model.project_ego_to_img_motion_compensated(
                        sweep.xyz,
                        city_SE3_ego_cam_t=city_SE3_ego_cam_t,
                        city_SE3_ego_lidar_t=city_SE3_ego_lidar_t,
                    )

                    uv_int: NDArrayInt = np.round(uv[is_valid_points]).astype(int)  # type: ignore
                    colors = create_range_map(points_cam[is_valid_points, :3])
                    img = draw_points_xy_in_img(
                        cam.img, uv_int, colors=colors, alpha=0.85, diameter=5, sigma=1.0, with_anti_alias=True
                    )
                    if annotations is not None:
                        img = annotations.project_to_cam(
                            img, cam.camera_model, city_SE3_ego_cam_t, city_SE3_ego_lidar_t
                        )

                cam_name_to_img[cam_name] = cam.img
                    
            if len(cam_name_to_img) < len(cam_names):
                continue
            tiled_img = tile_cameras(cam_name_to_img, bev_img=None)
            tiled_img = white_to_transparency(Image.fromarray(tiled_img[:,:,::-1]))
            tiled_img = tiled_img.resize((tiled_img.size[0]//6,tiled_img.size[1]//6), Image.Resampling.LANCZOS)
            tiled_img.save(f'output/{idx}.png', quality=70)
            # tiled_cams_list.append(tiled_img)

        # print(datum.sweep_number, datum.num_sweeps_in_log)
        # if datum.sweep_number == datum.num_sweeps_in_log - 1:
        #     video: NDArrayByte = np.stack(tiled_cams_list)
        #     dst_path = Path("videos") / f"{datum.log_id}.mp4"
        #     dst_path.parent.mkdir(parents=True, exist_ok=True)
        #     write_video(video, dst_path, crf=30, color_format=ColorFormats.RGB)
        #     tiled_cams_list = []


@click.command(help="Generate visualizations from the Argoverse 2 Sensor Dataset.")
@click.option(
    "-d",
    "--dataset-dir",
    required=True,
    help="Path to local directory where the Argoverse 2 Sensor Dataset is stored.",
    type=click.Path(exists=True),
)
@click.option(
    "-a",
    "--with-annotations",
    default=False,
    help="Boolean flag to return annotations from the dataloader.",
    type=bool,
)
@click.option(
    "-c",
    "--cam_names",
    default=tuple(x.value for x in RingCameras),
    help="List of cameras to load for each lidar sweep.",
    multiple=True,
    type=str,
)
def run_generate_sensor_dataset_visualizations(
    dataset_dir: str, with_annotations: bool, cam_names: Tuple[str, ...]
) -> None:
    """Click entry point for Argoverse Sensor Dataset visualization.

    Args:
        dataset_dir: Dataset directory.
        with_annotations: Boolean flag to return annotations.
        cam_names: Tuple of camera names to load.

    Raises:
        ValueError: If no valid camera names are provided.
    """
    valid_ring_cams = set([x.value for x in RingCameras])
    valid_stereo_cams = set([x.value for x in StereoCameras])

    cam_enums: List[Union[RingCameras, StereoCameras]] = []
    for cam_name in cam_names:
        if cam_name in valid_ring_cams:
            cam_enums.append(RingCameras(cam_name))
        elif cam_name in valid_stereo_cams:
            cam_enums.append(StereoCameras(cam_name))
        else:
            raise ValueError("Must provide _valid_ camera names!")

    generate_sensor_dataset_visualizations(
        Path(dataset_dir),
        with_annotations,
        tuple(cam_enums),
    )


if __name__ == "__main__":
    run_generate_sensor_dataset_visualizations()
