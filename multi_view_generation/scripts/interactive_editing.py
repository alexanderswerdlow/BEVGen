import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


from multi_view_generation.bev_utils import camera_bev_grid, argoverse_camera_bev_grid
import av2.rendering.map as map_rendering_utils
import av2.utils.raster as raster_utils
import typer
from av2.datasets.sensor.constants import AnnotationCategories
from av2.geometry.se3 import SE3
from av2.map.lane_segment import LaneMarkType, LaneType
from PIL import Image
from tqdm import tqdm
from multi_view_generation.bev_utils import Dataset
from av2.utils.constants import HOME
import torchvision.transforms.functional as T
from multi_view_generation.bev_utils.argoverse_helper import argo_to_standard_category
from image_utils import Im
import gradio as gr
from hydra import compose, initialize
import albumentations as A
from multi_view_generation import utils
from pytorch_lightning import LightningModule
import hydra
from typing import Dict, Tuple
from image_utils import library_ops
from torch.utils.data import Dataset
import torch
from av2.structures.cuboid import CuboidList, Cuboid
from multi_view_generation.bev_utils import batched_camera_bev_grid
from multi_view_generation.bev_utils.nuscenes_helper import DIVIDER, DYNAMIC, STATIC, get_annotations_by_category, get_dynamic_layers, get_dynamic_objects, get_line_layers, get_static_layers, parse_pose
import numpy as np
from functools import partial

# Run with python -m multi_view_generation.scripts.interactive_editing

log = utils.get_pylogger(__name__)


def get_bev(dataset, cam_record, sample_token, sample_record, lidar_record, map_name):

    egolidar = dataset.nusc.get("ego_pose", lidar_record["ego_pose_token"])

    world_from_egolidarflat = parse_pose(egolidar, flat=True)
    egolidarflat_from_world = parse_pose(egolidar, flat=True, inv=True)

    cam_record_data = {}
    # cam_record_data["token"] = sample_token
    cam_record_data["pose"] = world_from_egolidarflat.tolist()
    cam_record_data["pose_inverse"] = egolidarflat_from_world.tolist()
    cam_record_data["scene"] = dataset.nusc.get("scene", sample_record["scene_token"])["name"]
    cam_record_data["cam_token"] = cam_record['token']

    bev = get_metadrive_compatible_bev_v2(dataset, cam_record_data)

    return A.Compose([])(image=bev)["image"]


def get_metadrive_compatible_bev_v2(dataset, cam_record_data: Dict):
    """Return BEV image from nuScenes data in 21-channel format"""
    anns_dynamic = get_annotations_by_category(dataset.nusc, cam_record_data, DYNAMIC)
    anns_vehicle = get_annotations_by_category(dataset.nusc, cam_record_data, ["vehicle"])[0]

    dynamic = get_dynamic_layers(cam_record_data, anns_dynamic)  # 200 200 8

    static = get_static_layers(dataset.nusc_map, cam_record_data, STATIC)  # 200 200 2
    dividers = get_line_layers(dataset.nusc_map, cam_record_data, DIVIDER)  # 200 200 2

    aux, visibility = get_dynamic_objects(cam_record_data, anns_vehicle)

    bev = np.concatenate((static, dividers, dynamic, visibility[..., None]), -1)  # 200 200 14
    bev = (bev / 255.0).astype(np.float32)
    bev = np.concatenate((bev, aux), -1)

    return bev


def raw_output_data_bev_grid(batch):
    images = Im(batch['image']).denormalize().torch
    ret_images = []

    if batch['dataset'] == 'nuscenes':
        viz_func = camera_bev_grid
    else:
        viz_func = argoverse_camera_bev_grid
    image_dict = {batch['cam_name'][k]: images[k] for k in range(images.shape[0])}
    ret_images.append(viz_func(images=image_dict, bev=batch["segmentation"]))

    return ret_images


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


def get_bev(dataset, data):
    log_id, annotations, avm, timestamp_ns, split = (data.log_id, data.annotations, data.avm, data.timestamp_ns, data.split)
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
    return masked_layers


num_to_edit = 10
num_images = 5


def transfer_batch_to_gpu(batch, gpu_id):
    # base case
    if isinstance(batch, torch.Tensor):
        return batch.cuda(gpu_id)

    # when list
    elif isinstance(batch, list):
        for i, x in enumerate(batch):
            batch[i] = transfer_batch_to_gpu(x, gpu_id)
        return batch

    # when dict
    elif isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = transfer_batch_to_gpu(v, gpu_id)

        return batch


def predict(idx, modified_annotations, ret, image_0, image_1, image_2, image_3, image_4, model, dataset):
    if ret is None:
        return [None for _ in range(num_images)]

    data = ret["data"]
    log_id, annotations, avm, timestamp_ns, split = (data.log_id, data.annotations, data.avm, data.timestamp_ns, data.split)

    for i in range(len(data.annotations)):
        modified, cuboid = modified_annotations[i], annotations.cuboids[i]
        annotations.cuboids[i] = Cuboid(
            dst_SE3_object=SE3(cuboid.dst_SE3_object.rotation, np.array([float(modified[2]), -float(modified[1]), cuboid.dst_SE3_object.translation[2]])),
            length_m=cuboid.length_m,
            width_m=cuboid.width_m,
            height_m=cuboid.height_m,
            category=cuboid.category,
            timestamp_ns=cuboid.timestamp_ns,
        )

    bev = get_bev(dataset, data)
    ret['segmentation'] = bev.copy()
    keep_data = ret['data']
    ret['data'] = 0
    from torch.utils.data._utils.collate import default_collate
    input_data = default_collate([ret])
    input_data = transfer_batch_to_gpu(input_data, 0)

    with torch.no_grad():
        output = model(input_data)

    ret['data'] = keep_data

    new_image = Im(batched_camera_bev_grid(model.cfg, output['gen'][0], input_data['segmentation'][0].to(dtype=torch.float).detach().cpu().numpy()[None,])[0]).pil

    return new_image, image_0, image_1, image_2, image_3


def get_annotations(idx, dataset):
    ret = dataset[idx]
    data = ret["data"]

    data.annotations = CuboidList(cuboids=sorted(data.annotations, key=lambda x: np.linalg.norm(x.dst_SE3_object.translation[:2])))
    data.annotations = CuboidList(cuboids=list(filter(lambda x: x.category != "PEDESTRIAN", data.annotations)))
    #  np.linalg.norm(x.dst_SE3_object.translation[:2]) < 50 and
    data.annotations = CuboidList(cuboids=data.annotations.cuboids[:num_to_edit])

    value = []
    for i in range(len(data.annotations)):
        category, cuboid = data.annotations.categories[i], data.annotations.cuboids[i]
        value.append((category, *list(np.round(np.array([-cuboid.dst_SE3_object.translation[1], cuboid.dst_SE3_object.translation[0]]), 2))))

    return gr.update(value=value), ret


def run_editing(model, dataset) -> Tuple[dict, dict]:
    with gr.Blocks() as demo:
        instance = gr.Slider(0, len(dataset), label="Choose any Argoverse 2 instance", value=21000)
        annotations = gr.Dataframe(
            headers=["annotation type", "x", "y"],
            datatype=["str", "number", "number"],
            col_count=(3, "fixed"),
            type="array",
            label="The nearest 10 annotations (sorted). Double click to edit."
        )
        btn = gr.Button(value="Generate!")
        instance_ret = gr.State()

        get_annotations_ = partial(get_annotations, dataset=dataset)
        instance.change(fn=get_annotations_, inputs=instance, outputs=[annotations, instance_ret])
        demo.load(fn=get_annotations_, inputs=instance, outputs=[annotations, instance_ret])

        images = []
        for _ in range(num_images):
            images.append(gr.Image(type="pil"))

        predict_ = partial(predict, model=model, dataset=dataset)
        btn.click(fn=predict_, inputs=[instance, annotations, instance_ret, *images], outputs=images)

    demo.launch(share=True)


def main() -> None:
    initialize(version_base="1.2", config_path="../../configs", job_name="interactive_editing")
    cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=["experiment=muse_stage_two_multi_view", "datamodule=stage_2_argoverse_generate",
                  "modes=argoverse", "model.debug_viz=False", "+model.ckpt_path=logs/default/runs/2023-03-04_15-07-39/checkpoints/last.ckpt", "+datamodule.test.raw_data=True"])  # ,

    log.info(f"Instantiating datamodule <{cfg.datamodule.test._target_}>")
    dataset: Dataset = hydra.utils.instantiate(cfg.datamodule.test)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    cfg.paths.output_dir = None
    model: LightningModule = hydra.utils.instantiate(cfg.model).to("cuda:0")
    model.eval()

    run_editing(model, dataset)


if __name__ == "__main__":
    main()
