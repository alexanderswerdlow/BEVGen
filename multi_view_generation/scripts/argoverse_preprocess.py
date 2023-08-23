from typing import Optional
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from multi_view_generation.bev_utils.argoverse_helper import argo_to_standard_category
from multi_view_generation.bev_utils.util import Cameras, split_range
from multi_view_generation.bev_utils import ARGOVERSE_DIR
import torchvision.transforms.functional as T
from av2.utils.constants import HOME
from multi_view_generation.bev_utils import util, viz_bev
from multi_view_generation.bev_utils import Dataset
from multi_view_generation.modules.transformer.mingpt_sparse import GPTConfig
import albumentations as A
from tqdm import tqdm
from PIL import Image
from av2.map.lane_segment import LaneMarkType, LaneType
from av2.geometry.se3 import SE3
from av2.datasets.sensor.constants import AnnotationCategories
import typer
import torch
import numpy as np
import av2.utils.raster as raster_utils
import av2.rendering.map as map_rendering_utils
from av2.datasets.sensor.constants import RingCameras
from multi_view_generation.bev_utils.argoverse_sensor_dataloader import SensorDataloader, SynchronizedSensorData
from multi_view_generation.bev_utils.argoverse_multi_sensor_dataloader import MultiSensorData, MultiSynchronizedSensorData
import random
from av2.rendering.video import tile_cameras
from pathlib import Path
from os.path import exists
from multiprocessing import Pool
from functools import partial
import shutil
import os


def cam_to_bev(polygon):
    polygon = (polygon - np.array(extents[:2])) / resolution
    polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
    return polygon


def city_to_bev(xyz, ego_SE3_cam, city_SE3_ego_cam_t):
    return cam_to_bev(ego_SE3_cam.inverse().transform_from(city_SE3_ego_cam_t.inverse().transform_from(xyz))[:, [0, 2]])


def bev_to_city(xy, ego_SE3_cam, city_SE3_ego_cam_t):
    xyz = np.hstack(((xy + np.array(extents[:2])) * resolution, np.zeros((xy.shape[0], 1))))
    return (city_SE3_ego_cam_t.compose(ego_SE3_cam)).transform_point_cloud(xyz)


def square_pad(image, h, w):
    h_1, w_1 = image.shape[-2:]
    ratio_f = w / h
    ratio_1 = w_1 / h_1

    # check if the original and final aspect ratios are the same within a margin
    if round(ratio_1, 2) != round(ratio_f, 2):

        # padding to preserve aspect ratio
        hp = int(w_1 / ratio_f - h_1)
        wp = int(ratio_f * h_1 - w_1)
        if hp > 0 and wp < 0:
            hp = hp // 2
            image = T.pad(image, (0, hp, 0, hp), 0, "constant")
            return T.resize(image, [h, w])

        elif hp < 0 and wp > 0:
            wp = wp // 2
            image = T.pad(image, (wp, 0, wp, 0), 0, "constant")
            return T.resize(image, [h, w])

    else:
        return T.resize(image, [h, w])


# min_, max_ = -2.87206, 4.41666
img_range = 40
extents = [-img_range, -img_range, img_range, img_range]
desired_resolution = 256
resolution = (2 * img_range) / desired_resolution


def process_item(indices, dataset, save_dir, visualize, specific_frames):
    if specific_frames is not None and exists(specific_frames):
        import pickle
        with open(specific_frames, "rb") as f:
            specific_frames = pickle.load(f)
    elif specific_frames is not None:
        split, log_id, timestamp_ns = str(specific_frames).split("_")
        timestamp_ns = int(timestamp_ns)
        specific_frames = [(split, log_id, timestamp_ns)]

        dataset_ = MultiSensorData(
            Path('/data/datasets/av2/sensor'),
            'val',
            with_cache=True,
            cam_names=list(map(lambda x: RingCameras(x), Cameras.ARGOVERSE_FRONT_CAMERAS)),
        )

    for idx in tqdm(specific_frames if specific_frames else indices):
        if specific_frames:
            data: SynchronizedSensorData = dataset_.get_from_record(*idx)
        else:
            data: SynchronizedSensorData = dataset[idx]

        if data.annotations is None:
            print("No annotations", data.log_id, data.timestamp_ns)
            continue

        log_id, annotations, avm, timestamp_ns, split = (data.log_id, data.annotations, data.avm, data.timestamp_ns, data.split)

        # if len(data.synchronized_imagery) == 7:
        #     # from scipy.spatial.transform import Rotation as R
        #     # angles = {cam_name.value : np.mod(-data.synchronized_imagery[cam_name].camera_model.egovehicle_yaw_cam_rad, 2 * np.pi) for cam_name in RingCameras}
        #     # print(repr(angles))
        #     # np.rad2deg(data.synchronized_imagery['ring_front_center'].camera_model.compute_pixel_ray_directions(np.array([[0, data.synchronized_imagery['ring_front_center'].camera_model.intrinsics.height_px / 2]]))[0])
        # else:
        #     print(len(data.synchronized_imagery))
        #     continue

        output_folder = save_dir / split / log_id
        output_filename = output_folder / f"{timestamp_ns}.npz"

        if exists(output_filename):
            continue
        else:
            print("Creating: ", data.log_id, data.timestamp_ns)

        city_SE3_ego_cam_t = data.timestamp_city_SE3_ego_dict[timestamp_ns]

        output_shape = (desired_resolution, desired_resolution)

        ego_SE3_cam: SE3 = SE3(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]), np.zeros(3))
        # ego_SE3_cam: SE3 = SE3(np.rint(data.synchronized_imagery['ring_front_center'].camera_model.ego_SE3_cam.rotation), np.zeros(3))

        da_polygons_img = []
        for da_polygon_city in list(avm.vector_drivable_areas.values()):
            da_polygon_img = city_to_bev(da_polygon_city.xyz, ego_SE3_cam, city_SE3_ego_cam_t)
            da_polygon_img = np.round(da_polygon_img).astype(np.int32)  # type: ignore
            da_polygons_img.append(da_polygon_img)

        driveable_img = raster_utils.get_mask_from_polygons(da_polygons_img, *output_shape)

        cuboid_categories = {category: [] for category in AnnotationCategories}
        for cuboid in annotations.cuboids:
            # cam_img.camera_model.project_ego_to_img(points_cam_time)
            da_polygon_img = cam_to_bev(ego_SE3_cam.inverse().transform_point_cloud(cuboid.vertices_m[[0, 1, 5, 4]])[:, [0, 2]])
            cuboid_categories[cuboid.category].append(da_polygon_img)

        category_imgs = [raster_utils.get_mask_from_polygons(v, *output_shape) for k, v in cuboid_categories.items()]

        lane_imgs = {
            (lane_type, lane_mark_type): np.ascontiguousarray(np.zeros(output_shape).astype(np.uint8))
            for lane_mark_type in LaneMarkType
            for lane_type in LaneType
        }
        stopline_img = np.ascontiguousarray(np.zeros(output_shape).astype(np.uint8))

        for lane_segment in avm.vector_lane_segments.values():
            map_rendering_utils.draw_visible_polyline_segments_cv2(
                line_segments_arr=city_to_bev(lane_segment.left_lane_boundary.xyz, ego_SE3_cam, city_SE3_ego_cam_t),
                valid_pts_bool=np.full((lane_segment.left_lane_boundary.xyz.shape[0],), True),
                image=lane_imgs[(lane_segment.lane_type, lane_segment.left_mark_type)],
                color=1,
                thickness_px=1,
            )

            map_rendering_utils.draw_visible_polyline_segments_cv2(
                line_segments_arr=city_to_bev(
                    lane_segment.right_lane_boundary.xyz,
                    ego_SE3_cam,
                    city_SE3_ego_cam_t,
                ),
                valid_pts_bool=np.full((lane_segment.right_lane_boundary.xyz.shape[0],), True),
                image=lane_imgs[(lane_segment.lane_type, lane_segment.left_mark_type)],
                color=1,
                thickness_px=1,
            )

            if lane_segment.is_intersection:
                stopline = np.vstack(
                    [
                        lane_segment.right_lane_boundary.xyz[0],
                        lane_segment.left_lane_boundary.xyz[0],
                    ]
                )
                map_rendering_utils.draw_visible_polyline_segments_cv2(
                    line_segments_arr=city_to_bev(stopline, ego_SE3_cam, city_SE3_ego_cam_t),
                    valid_pts_bool=np.full((stopline.shape[0],), True),
                    image=stopline_img,
                    color=1,
                    thickness_px=1,
                )

        da_polygons_img = [city_to_bev(ped_xing.polygon, ego_SE3_cam, city_SE3_ego_cam_t) for ped_xing in avm.vector_pedestrian_crossings.values()]
        ped_xing_img = raster_utils.get_mask_from_polygons(da_polygons_img, *output_shape)
        cuboids = argo_to_standard_category(np.stack(category_imgs, axis=-1))
        lane_lines = np.logical_or.reduce(list(lane_imgs.values()))
        stopline_and_ped_xing_img = np.logical_or.reduce([ped_xing_img, stopline_img])
        layers = np.stack([*cuboids, driveable_img, lane_lines, stopline_and_ped_xing_img], axis=-1)
        masked_layers = np.flipud(layers)

        if visualize:
            cust = np.zeros_like(driveable_img)
            dist = cust.shape[0] // 2
            cust[dist - 4: dist + 4, dist - 4: dist + 4] = 1
            custom_layers = np.stack([cust, *category_imgs, *lane_imgs.values(), ped_xing_img, stopline_img], axis=-1)
            os.makedirs(Path("output"), exist_ok=True)
            custom_layers = np.flipud(custom_layers)
            rgb_img = viz_bev(masked_layers, dataset=Dataset.ARGOVERSE).pil

            if data.synchronized_imagery is not None:
                cam_name_to_img = {}
                for cam_name, cam in data.synchronized_imagery.items():
                    cam_name_to_img[cam_name] = cam.img

                if len(cam_name_to_img) < 1:
                    continue
            
            bev_img = util.torch_to_numpy(square_pad(T.to_tensor(rgb_img), 2048, 1550))
            tiled_img = tile_cameras(cam_name_to_img, bev_img=bev_img)
            Image.fromarray(tiled_img).save(f"output/{timestamp_ns}.png")
        else:
            os.makedirs(output_folder, exist_ok=True)
            np.savez_compressed(output_filename, masked_layers)


app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    dataset_dir: Path = ARGOVERSE_DIR / 'sensor',
    save_dir: Path = Path("output"),
    multiprocess: bool = False,
    visualize: bool = False,
    val_set: bool = True,
    specific_frames: Optional[Path] = None,
):
    if save_dir == Path("output") and save_dir.exists() and save_dir.is_dir():
        print(f"Deleting {save_dir}")
        shutil.rmtree(save_dir)

    dataset = SensorDataloader(
        dataset_dir,
        'val' if val_set else 'train',
        with_cache=True,
        cam_names=list(map(lambda x: RingCameras(x), Cameras.ARGOVERSE_FRONT_CAMERAS)),
    )
    if multiprocess:
        num_processes = 24
        with Pool(num_processes) as p:
            num_arr = list(range(len(dataset)))
            random.shuffle(num_arr)
            for _ in p.imap_unordered(
                partial(
                    process_item,
                    dataset=dataset,
                    save_dir=save_dir,
                    visualize=visualize,
                    specific_frames=specific_frames
                ),
                split_range(num_arr, num_processes),
            ):
                pass
    else:
        process_item(
            range(len(dataset)),
            dataset=dataset,
            save_dir=save_dir,
            visualize=visualize,
            specific_frames=specific_frames
        )


if __name__ == "__main__":
    torch.manual_seed(1)
    colorize = torch.randn(3, 8, 1, 1)
    colorize.share_memory_()
    app()