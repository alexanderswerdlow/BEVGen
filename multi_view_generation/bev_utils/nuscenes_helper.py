from enum import Enum
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
import torch
from matplotlib.pyplot import get_cmap
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from shapely.geometry import MultiPolygon
import numpy as np
from multi_view_generation.bev_utils import Cameras

STATIC = ["lane", "road_segment"]
DIVIDER = ["road_divider", "lane_divider"]
DYNAMIC = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction",
    "pedestrian",
    "motorcycle",
    "bicycle",
]

CLASSES = STATIC + DIVIDER + DYNAMIC
NUM_CLASSES = len(CLASSES)
INTERPOLATION = cv2.LINE_8

VIZ_TOKENS = [
    "250018ac46314ca4873919f3cde82a8c",
    "e855c339d1ee4a1bb3fb96d165349af3",
]

class Split(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

bev = {"h": 256, "w": 256, "h_meters": 80, "w_meters": 80, "offset": 0.0}
bev_shape = (bev["h"], bev["w"])

def split_range(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    sh = h / h_meters
    sw = w / w_meters

    return np.float32([[0.0, -sw, w / 2.0], [-sh, 0.0, h * offset + h / 2.0], [0.0, 0.0, 1.0]])

view = get_view_matrix(**bev)

def get_transformation_matrix(R, t, inv=False):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R if not inv else R.T
    pose[:3, -1] = t if not inv else R.T @ -t

    return pose


class NusceneCamGeometry:
    """
    A class to handle the change of camera intrinsics for data augmentation.
    Currenly mainly for cropping and resizing. TODO: Add rotation later.
    """
    def __init__(self):
        self.x_scale = 0
        self.y_scale = 0

        # the origin of image is on the left up corner,
        # so cropping on the bottom and right side won't influence the x0 y0
        # in intrinsics
        self.top_crop = 0
        self.left_crop = 0

        # the order of image augment
        self.rescale_first = True

    def set_scale(self, x_scale, y_scale):
        self.x_scale = x_scale
        self.y_scale = y_scale

    def set_crop(self, top_crop, left_crop):
        self.top_crop = top_crop
        self.left_crop = left_crop

    def apply(self, cam_intrinc):
        """
        Apply the change to camera intrinsic.

        Parameters
        ----------
        camera_intrinsic : list or np.array
        """
        camera_intrinsic = deepcopy(cam_intrinc)
        if self.rescale_first:
            if isinstance(camera_intrinsic, list):
                camera_intrinsic[0][0] *= self.x_scale
                camera_intrinsic[0][2] *= self.x_scale
                camera_intrinsic[1][1] *= self.y_scale
                camera_intrinsic[1][2] *= self.y_scale
                camera_intrinsic[1][2] -= self.top_crop
                camera_intrinsic[0][2] -= self.left_crop

            else:
                camera_intrinsic[0, 0] *= self.x_scale
                camera_intrinsic[0, 2] *= self.x_scale
                camera_intrinsic[1, 1] *= self.y_scale
                camera_intrinsic[1, 2] *= self.y_scale
                camera_intrinsic[1, 2] -= self.top_crop
                camera_intrinsic[0, 2] -= self.left_crop

        else:
            if isinstance(camera_intrinsic, list):
                camera_intrinsic[1][2] -= self.top_crop
                camera_intrinsic[0][2] -= self.left_crop
                camera_intrinsic[0][0] *= self.x_scale
                camera_intrinsic[0][2] *= self.x_scale
                camera_intrinsic[1][1] *= self.y_scale
                camera_intrinsic[1][2] *= self.y_scale

            else:
                camera_intrinsic[1, 2] -= self.top_crop
                camera_intrinsic[0, 2] -= self.left_crop
                camera_intrinsic[0, 0] *= self.x_scale
                camera_intrinsic[0, 2] *= self.x_scale
                camera_intrinsic[1, 1] *= self.y_scale
                camera_intrinsic[1, 2] *= self.y_scale

        return camera_intrinsic


def resize_intrinsic(camera_intrinsic, x_scale, y_scale):
    """
    Adjust the intrinsic
    """

    # modify the camera intrinsic according to resize
    camera_intrinsic[0][0] *= x_scale
    camera_intrinsic[0][2] *= x_scale
    camera_intrinsic[1][1] *= y_scale
    camera_intrinsic[1][2] *= y_scale

    return camera_intrinsic


def get_pose(rotation, translation, inv=False, flat=False):
    if flat:
        yaw = Quaternion(rotation).yaw_pitch_roll[0]
        R = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).rotation_matrix
    else:
        R = Quaternion(rotation).rotation_matrix

    t = np.array(translation, dtype=np.float32)

    return get_transformation_matrix(R, t, inv=inv)


def encode(x):
    """
    (h, w, c) np.uint8 {0, 255}
    """
    n = x.shape[2]

    # assert n < 16
    assert x.ndim == 3
    assert x.dtype == np.uint8
    assert all(x in [0, 255] for x in np.unique(x))

    shift = np.arange(n, dtype=np.int32)[None, None]

    binary = x > 0
    binary = (binary << shift).sum(-1)
    binary = binary.astype(np.int32)

    return binary

def decode_binary_labels(labels, nclass):
    bits = torch.pow(2, torch.arange(nclass))
    return (labels & bits.view(-1, 1, 1)) > 0

def decode(img, n):
    """
    returns (h, w, n) np.int32 {0, 1}
    """
    shift = np.arange(n, dtype=np.int32)[None, None]

    x = np.array(img)[..., None]
    x = (x >> shift) & 1

    return x


def get_split(split):
    split_path = Path(__file__).parent / "splits" / f"{split}.txt"
    return split_path.read_text().strip().split("\n")


def quaternion_yaw(rotation_matrix) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def compute_pixel_ray_directions(uv, fx, fy, img_w, img_h):
    """Given (u,v) coordinates and intrinsics, generate pixel rays in the camera coordinate frame.

    Assume +z points out of the camera, +y is downwards, and +x is across the imager.

    Args:
        uv: Numpy array of shape (N,2) with (u,v) coordinates

    Returns:
        Array of shape (N,3) with ray directions to each pixel, provided in the camera frame.

    Raises:
        ValueError: If input (u,v) coordinates are not (N,2) in shape.
        RuntimeError: If generated ray directions are not (N,3) in shape.
    """
    # fx and fy can be more different now due to the horizontal resize
    if not np.isclose(fx, fy, atol=5):
        raise ValueError(f"Focal lengths in the x and y directions must match: {fx} != {fy}")

    if uv.shape[1] != 2:
        raise ValueError("Input (u,v) coordinates must be (N,2) in shape.")

    # Approximation for principal point
    px = img_w / 2
    py = img_h / 2

    u = uv[:, 0]
    v = uv[:, 1]
    num_rays = uv.shape[0]
    ray_dirs = np.zeros((num_rays, 3))
    # x center offset from center
    ray_dirs[:, 0] = u - px
    # y center offset from center
    ray_dirs[:, 1] = v - py
    ray_dirs[:, 2] = fx

    # elementwise multiplication of scalars requires last dim to match
    ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=1, keepdims=True)  # type: ignore
    if ray_dirs.shape[1] != 3:
        raise RuntimeError("Ray directions must be (N,3)")
    return ray_dirs

def parse_scene(nusc, scene_record, camera_rigs):
    data = []
    sample_token = scene_record["first_sample_token"]

    while sample_token:
        sample_record = nusc.get("sample", sample_token)

        for camera_rig in camera_rigs:
            data.append(parse_sample_record(nusc, scene_record, sample_record, camera_rig))

        sample_token = sample_record["next"]

    return data

def parse_pose(record, *args, **kwargs):
    return get_pose(record["rotation"], record["translation"], *args, **kwargs)

def parse_sample_record(nusc, scene_record, sample_record, camera_rig):
    lidar_record = nusc.get("sample_data", sample_record["data"]["LIDAR_TOP"])
    egolidar = nusc.get("ego_pose", lidar_record["ego_pose_token"])

    world_from_egolidarflat = parse_pose(egolidar, flat=True)
    egolidarflat_from_world = parse_pose(egolidar, flat=True, inv=True)

    cam_channels = []
    images = []
    intrinsics = []
    extrinsics = []

    for cam_idx in camera_rig:
        cam_channel = Cameras.NUSCENES_CAMERAS[cam_idx]
        cam_token = sample_record["data"][cam_channel]

        cam_record = nusc.get("sample_data", cam_token)
        egocam = nusc.get("ego_pose", cam_record["ego_pose_token"])
        cam = nusc.get("calibrated_sensor", cam_record["calibrated_sensor_token"])

        cam_from_egocam = parse_pose(cam, inv=True)
        egocam_from_world = parse_pose(egocam, inv=True)

        E = cam_from_egocam @ egocam_from_world @ world_from_egolidarflat
        I = cam["camera_intrinsic"]

        full_path = Path(nusc.get_sample_data_path(cam_token))
        image_path = str(full_path.relative_to(nusc.dataroot))

        cam_channels.append(cam_channel)
        intrinsics.append(I)
        extrinsics.append(E.tolist())
        images.append(image_path)

    return {
        "scene": scene_record["name"],
        "token": sample_record["token"],
        "pose": world_from_egolidarflat.tolist(),
        "pose_inverse": egolidarflat_from_world.tolist(),
        "cam_ids": list(camera_rig),
        "cam_channels": cam_channels,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "images": images,
    }

def get_dynamic_objects(sample, annotations):
    h, w = bev_shape[:2]

    segmentation = np.zeros((h, w), dtype=np.uint8)
    center_score = np.zeros((h, w), dtype=np.float32)
    center_offset = np.zeros((h, w, 2), dtype=np.float32)
    center_ohw = np.zeros((h, w, 4), dtype=np.float32)
    buf = np.zeros((h, w), dtype=np.uint8)

    visibility = np.full((h, w), 255, dtype=np.uint8)

    coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32)

    for ann, p in zip(annotations, convert_to_box(sample, annotations)):
        box = p[:2, :4]
        center = p[:2, 4]
        front = p[:2, 5]
        left = p[:2, 6]

        buf.fill(0)
        cv2.fillPoly(buf, [box.round().astype(np.int32).T], 1, INTERPOLATION)
        mask = buf > 0

        if not np.count_nonzero(mask):
            continue

        sigma = 1
        segmentation[mask] = 1.0
        center_offset[mask] = (((center[None] - coords[mask]) / h) + 1) / 2
        center_score[mask] = np.exp(-(center_offset[mask] ** 2).sum(-1) / (sigma**2))

        # orientation, h/2, w/2
        center_ohw[mask, 0:2] = ((((front - center) / (np.linalg.norm(front - center) + 1e-6))[None]) + 1) / 2
        center_ohw[mask, 2:3] = np.linalg.norm(front - center) / h
        center_ohw[mask, 3:4] = np.linalg.norm(left - center) / h

        visibility[mask] = ann["visibility_token"]

    segmentation = np.float32(segmentation[..., None])
    center_score = center_score[..., None]

    result = np.concatenate((segmentation, center_score, center_offset, center_ohw), 2)

    # (h, w, 1 + 1 + 2 + 2)
    return result, visibility

def convert_to_box(sample, annotations):
    # Import here so we don't require nuscenes-devkit unless regenerating labels
    from nuscenes.utils import data_classes

    V = view
    M_inv = np.array(sample["pose_inverse"])
    S = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    )

    for a in annotations:
        box = data_classes.Box(a["translation"], a["size"], Quaternion(a["rotation"]))

        corners = box.bottom_corners()  # 3 4
        center = corners.mean(-1)  # 3
        front = (corners[:, 0] + corners[:, 1]) / 2.0  # 3
        left = (corners[:, 0] + corners[:, 3]) / 2.0  # 3

        p = np.concatenate((corners, np.stack((center, front, left), -1)), -1)  # 3 7
        p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)  # 4 7
        p = V @ S @ M_inv @ p  # 3 7

        yield p  # 3 7

def get_category_index(name, categories):
    """
    human.pedestrian.adult
    """
    tokens = name.split(".")

    for i, category in enumerate(categories):
        if category in tokens:
            return i

    return None

def get_annotations_by_category(nusc, sample, categories):
    result = [[] for _ in categories]

    for ann_token in nusc.get_boxes(sample['cam_token']):
        a = nusc.get("sample_annotation", ann_token.token)
        idx = get_category_index(a["category_name"], categories)

        if idx is not None:
            result[idx].append(a)

    return result

def get_line_layers(nusc_map, sample, layers, patch_radius=150, thickness=1):
    h, w = bev_shape[:2]
    V = view
    M_inv = np.array(sample["pose_inverse"])
    S = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    )

    box_coords = (
        sample["pose"][0][-1] - patch_radius,
        sample["pose"][1][-1] - patch_radius,
        sample["pose"][0][-1] + patch_radius,
        sample["pose"][1][-1] + patch_radius,
    )
    records_in_patch = nusc_map[sample["scene"]].get_records_in_patch(box_coords, layers, "intersect")

    result = list()

    for layer in layers:
        render = np.zeros((h, w), dtype=np.uint8)

        for r in records_in_patch[layer]:
            polygon_token = nusc_map[sample["scene"]].get(layer, r)
            line = nusc_map[sample["scene"]].extract_line(polygon_token["line_token"])

            p = np.float32(line.xy)  # 2 n
            p = np.pad(p, ((0, 1), (0, 0)), constant_values=0.0)  # 3 n
            p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)  # 4 n
            p = V @ S @ M_inv @ p  # 3 n
            p = p[:2].round().astype(np.int32).T  # n 2

            cv2.polylines(render, [p], False, 1, thickness=thickness)

        result.append(render)

    return 255 * np.stack(result, -1)

def get_static_layers(nusc_map, sample, layers, patch_radius=150):
    h, w = bev_shape[:2]
    V = view
    M_inv = np.array(sample["pose_inverse"])
    S = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    )

    box_coords = (
        sample["pose"][0][-1] - patch_radius,
        sample["pose"][1][-1] - patch_radius,
        sample["pose"][0][-1] + patch_radius,
        sample["pose"][1][-1] + patch_radius,
    )
    records_in_patch = nusc_map[sample["scene"]].get_records_in_patch(box_coords, layers, "intersect")

    result = list()

    for layer in layers:
        render = np.zeros((h, w), dtype=np.uint8)

        for r in records_in_patch[layer]:
            polygon_token = nusc_map[sample["scene"]].get(layer, r)

            if layer == "drivable_area":
                polygon_tokens = polygon_token["polygon_tokens"]
            else:
                polygon_tokens = [polygon_token["polygon_token"]]

            for p in polygon_tokens:
                polygon = nusc_map[sample["scene"]].extract_polygon(p)
                polygon = MultiPolygon([polygon])

                exteriors = [np.array(poly.exterior.coords).T for poly in polygon.geoms]
                exteriors = [np.pad(p, ((0, 1), (0, 0)), constant_values=0.0) for p in exteriors]
                exteriors = [np.pad(p, ((0, 1), (0, 0)), constant_values=1.0) for p in exteriors]
                exteriors = [V @ S @ M_inv @ p for p in exteriors]
                exteriors = [p[:2].round().astype(np.int32).T for p in exteriors]

                cv2.fillPoly(render, exteriors, 1, INTERPOLATION)

                interiors = [np.array(pi.coords).T for poly in polygon.geoms for pi in poly.interiors]
                interiors = [np.pad(p, ((0, 1), (0, 0)), constant_values=0.0) for p in interiors]
                interiors = [np.pad(p, ((0, 1), (0, 0)), constant_values=1.0) for p in interiors]
                interiors = [V @ S @ M_inv @ p for p in interiors]
                interiors = [p[:2].round().astype(np.int32).T for p in interiors]

                cv2.fillPoly(render, interiors, 0, INTERPOLATION)

        result.append(render)

    return 255 * np.stack(result, -1)

def get_dynamic_layers(sample, anns_by_category):
    h, w = bev_shape[:2]
    result = list()

    for anns in anns_by_category:
        render = np.zeros((h, w), dtype=np.uint8)
        for p in convert_to_box(sample, anns):
            p = p[:2, :4]
            cv2.fillPoly(render, [p.round().astype(np.int32).T], 1, INTERPOLATION)

        result.append(render)

    return 255 * np.stack(result, -1)

class NuScenesSingleton:
    """
    Wraps both nuScenes and nuScenes map API

    This was an attempt to sidestep the 30 second loading time in a "clean" manner
    """

    def __init__(self, dataset_dir, version):
        """
        dataset_dir: /path/to/nuscenes/
        version: v1.0-trainval
        """
        self.dataroot = str(dataset_dir)
        self.nusc = NuScenes(version=version, dataroot=self.dataroot)

    def get_scenes(self):
        for scene_record in self.nusc.scene:
            yield scene_record["name"], scene_record

    @lru_cache(maxsize=16)
    def get_map(self, log_token):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        map_name = self.nusc.get("log", log_token)["location"]
        nusc_map = NuScenesMap(dataroot=self.dataroot, map_name=map_name)

        return nusc_map


def colorize(x, colormap=None):
    """
    x: (h w) np.uint8 0-255
    colormap
    """
    try:
        return (255 * get_cmap(colormap)(x)[..., :3]).astype(np.uint8)
    except:
        pass

    if x.dtype == np.float32:
        x = (255 * x).astype(np.uint8)

    if colormap is None:
        return x[..., None].repeat(3, 2)

    return cv2.applyColorMap(x, getattr(cv2, f"COLORMAP_{colormap.upper()}"))

def local_resize(src, dst=None, shape=None, idx=0):
    if dst is not None:
        ratio = dst.shape[idx] / src.shape[idx]
    elif shape is not None:
        ratio = shape[idx] / src.shape[idx]

    width = int(ratio * src.shape[1])
    height = int(ratio * src.shape[0])

    return cv2.resize(src, (width, height), interpolation=cv2.INTER_CUBIC)

class Sample(dict):
    def __init__(self, token, scene, intrinsics, extrinsics, images, view, bev, **kwargs):
        super().__init__(**kwargs)

        # Used to create path in save/load
        self.token = token
        self.scene = scene

        self.view = view
        self.bev = bev

        self.images = images
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    def __getattr__(self, key):
        return super().__getitem__(key)

    def __setattr__(self, key, val):
        self[key] = val

        return super().__setattr__(key, val)

def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


def polar_transform(signal):
    S = signal.shape[0]  # Original size of the aerial image
    height = signal.shape[0] - 1  # Height of polar transformed aerial image
    width = signal.shape[1] - 1   # Width of polar transformed aerial image
    fov_x = 1.5593362604608414

    i = np.arange(0, height + 1)
    j = np.arange(0, width + 1)
    jj, ii = np.meshgrid(j, i)

    y = S - (height-ii) * np.cos(fov_x * ((jj/width) - 0.5))
    x = (((height-ii)/2 * (np.sin(fov_x * ((jj/width) - 0.5)))) + 100)

    image = np.stack([bilinear_interpolate(signal[..., i], x, y) for i in range(signal.shape[2])], axis=-1)

    return image

