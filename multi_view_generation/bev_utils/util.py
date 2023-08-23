from enum import Enum
import numpy as np
import torch
from PIL import Image, ImageOps
import colorsys
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from einops import rearrange
import torchvision.transforms.functional as T
import random
import string
from random import randint

"""
This file contains a variety of utility functions for the Argoverse 2/nuScenes Dataloaders as well as various models used for visualizations, etc.
"""


class Cameras(Enum):
    NUSCENES_FRONT = ("CAM_FRONT",)
    NUSCENES_CAMERAS = ("CAM_FRONT", "CAM_BACK", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK_RIGHT", "CAM_BACK_LEFT")
    NUSCENES_ABLATION_CAMERAS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT")
    ARGOVERSE_CAMERAS = ("ring_side_left", "ring_front_left", "ring_front_right", "ring_side_right")
    ARGOVERSE_FRONT_CAMERAS = ("ring_front_left", "ring_front_center", "ring_front_right")
    ARGOVERSE_ALL_CAMERAS = ("ring_side_left", "ring_front_left", "ring_front_center", "ring_front_right", "ring_side_right")

    def __getitem__(self, index):
        return self._value_[index]

    def __len__(self):
        return len(self._value_)

    def index(self, index):
        return self._value_.index(index)

class Dataset(Enum):
    NUSCENES = 0
    ARGOVERSE = 1

def random_bbox(bbox):
    v = [randint(0, v) for v in bbox]
    left = min(v[0], v[2])
    upper = min(v[1], v[3])
    right = max(v[0], v[2])
    lower = max(v[1], v[3])
    return [left, upper, right, lower]


def get_fake_stage_2_data(cam_h=224, cam_w=400, seg_channels=21, cam_names=Cameras.NUSCENES_CAMERAS):
    num_cams = len(cam_names)
    return {
        'image': torch.randn(([num_cams, cam_h, cam_w, 3]), dtype=torch.float32),
        'segmentation': torch.randn(([256, 256, seg_channels]), dtype=torch.float32),
        'angle': torch.pi,
        'dataset': 'nuscenes',
        'token': '17c2936e57db48809ba36e21e466aba9',
        'channel': 'CAM_FRONT_LEFT',
        'cam_idx': torch.tensor([0, 1, 2, 3, 4, 5]),
        'intrinsics_inv': torch.randn([num_cams, 3, 3]),
        'extrinsics_inv': torch.randn([num_cams, 4, 4]),
        'intrinsics': torch.randn([6, 3, 3]),
        'extrinsics': torch.randn([6, 4, 4]),
        'view': torch.randn([3, 3]),
        'center': torch.randn([1, 200, 200]),
        'pose': torch.randn([4, 4]),
        'bbx': torch.tensor([[random_bbox([0, 0, cam_w, cam_h]) for _ in range(5)] for _ in range(6)]),
        'sample_token': ''.join(random.choices(string.ascii_lowercase, k=5)),
        'image_paths': [(''.join(random.choices(string.ascii_lowercase, k=5))) + ".png" for _ in range(6)],
        'cam_name': list(cam_names)
    }


def denormalize_tensor_norm(x, keep_tensor=False):
    """Requires B, C, H, W"""
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    if len(x.shape) == 3:
        ten = x.unsqueeze(0).clone().permute(1, 2, 3, 0)
    else:
        # 3, H, W, B
        ten = x.clone().permute(1, 2, 3, 0)

    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)

    ret = torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)  # B, 3, H, W
    if len(x.shape) == 3:
        ret = ret.squeeze()

    if not keep_tensor:
        ret = torch_to_numpy_img(ret)

    return ret
    
def denormalize_tensor(x, keep_tensor=False):
    """Requires B, C, H, W"""
    mean = [0.4265, 0.4489, 0.4769]
    std = [0.2053, 0.2206, 0.2578]

    if len(x.shape) == 3:
        ten = x.unsqueeze(0).clone().permute(1, 2, 3, 0)
    else:
        # 3, H, W, B
        ten = x.clone().permute(1, 2, 3, 0)

    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)

    ret = torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)  # B, 3, H, W
    if len(x.shape) == 3:
        ret = ret.squeeze()

    if not keep_tensor:
        ret = torch_to_numpy_img(ret)

    return ret


def torch_to_numpy_img(arr):
    if len(arr.shape) == 3:
        return arr.permute(1, 2, 0).cpu().detach().numpy()
    elif len(arr.shape) == 4:
        return arr.permute(0, 2, 3, 1).cpu().detach().numpy()


def chw_to_hwc(arr):
    if torch.is_tensor(arr):
        return arr.permute(1, 2, 0)
    else:
        return arr.transpose(1, 2, 0)


def get_concat_h(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, max(im2.height, im1.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_h_space(im1, im2):
    dst = Image.new("RGBA", (im1.width + im2.width + 5, max(im2.height, im1.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width + 10, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def torch_to_numpy(arr):
    return (chw_to_hwc(arr) * 255).cpu().detach().numpy().astype(np.uint8)


def numpy_to_pil(arr):
    return Image.fromarray((arr * 255).astype(np.uint8))


def get_layered_image_from_binary_mask(masks, flip=False):
    if torch.is_tensor(masks):
        masks = masks.cpu().detach().numpy()
    if flip:
        masks = np.flipud(masks)

    masks = masks.astype(np.bool_)

    colors = np.asarray(list(getDistinctColors(masks.shape[2])))
    img = np.zeros((*masks.shape[:2], 3))
    for i in range(masks.shape[2]):
        img[masks[..., i]] = colors[i]

    return Image.fromarray(img.astype(np.uint8))


def get_img_from_binary_masks(masks, flip=False):
    """H W C"""
    arr = encode_binary_labels(masks)
    if flip:
        arr = np.flipud(arr)

    colors = np.asarray(list(getDistinctColors(2 ** masks.shape[2])))
    return Image.fromarray(colors[arr].astype(np.uint8))


def encode_binary_labels(masks):
    if torch.is_tensor(masks):
        masks = masks.cpu().detach().numpy()

    masks = masks.transpose(2, 0, 1)
    bits = np.power(2, np.arange(len(masks), dtype=np.int32))
    return (masks.astype(np.int32) * bits.reshape(-1, 1, 1)).sum(0)


def HSVToRGB(h, s, v):
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
    return (int(255 * r), int(255 * g), int(255 * b))


def getDistinctColors(n):
    huePartition = 1.0 / (n + 1)
    return (HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n))


colorize_weights = {}


def colorize(x):
    if x.shape[0] not in colorize_weights:
        colorize_weights[x.shape[0]] = torch.randn(3, x.shape[0], 1, 1)

    x = F.conv2d(x, weight=colorize_weights[x.shape[0]])
    x = (x-x.min())/(x.max()-x.min())
    return x


def visualize_score(scores, heatmaps, grid, image, iou, record=None):
    """Visualization for translating images to maps"""
    # cuboids, driveable, lane_lines, stopline_and_ped_xing_img
    indices = torch.LongTensor([4, 6, 5, 0, 1, 2, 3])
    scores = scores[indices]
    heatmaps = heatmaps[indices]
    # Condese scores and ground truths to single map
    class_idx = torch.arange(len(scores)) + 1
    logits = scores.clone().cpu() * class_idx.view(-1, 1, 1)
    logits, _ = logits.max(dim=0)

    scores_ = (scores.detach().clone().cpu() > 0.5).float() * class_idx.view(-1, 1, 1)

    cls_idx = scores_.clone()
    cls_idx = cls_idx.argmax(dim=0)
    cls_idx = cls_idx.numpy() * 20
    cls_idx = cv2.applyColorMap(cv2.convertScaleAbs(cls_idx, alpha=1), cv2.COLORMAP_JET)

    # Visualize score
    fig = plt.figure(num="score", figsize=(8, 6))
    fig.clear()

    import matplotlib as mpl
    gs = mpl.gridspec.GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1:, 1])
    ax4 = fig.add_subplot(gs[1:, 2])

    image = ax1.imshow(torch_to_numpy_img(image))
    ax1.grid(which="both")
    ax2.imshow(np.flipud(cls_idx))
    ax3.imshow(get_layered_image_from_binary_mask(chw_to_hwc(scores > 0.5), flip=True))
    ax4.imshow(get_layered_image_from_binary_mask(chw_to_hwc(heatmaps), flip=True))

    ax2.set_title("Model output logits", size=11)
    ax3.set_title("Model prediction = logits" + r"$ > 0.5$", size=11)
    ax4.set_title("Ground truth", size=11)

    output_str = "IoUs "
    for k, v in iou.items():
        output_str += f"{k}: {v:.2f} "
    plt.suptitle(output_str, size=8, wrap=True)
    if record is not None:
        plt.figtext(0.025, 0.925, record["log_id"], fontsize="small")
        plt.figtext(0.025, 0.875, record["sensor_name"])
        plt.figtext(0.025, 0.825, record["timestamp_ns"])

    gs.tight_layout(fig)
    gs.update(top=0.9)

    return fig


def square_pad(image, h, w):
    h_1, w_1 = image.shape[-2:]
    ratio_f = w / h
    ratio_1 = w_1 / h_1

    # check if the original and final aspect ratios are the same within a margin
    if round(ratio_1, 2) != round(ratio_f, 2):

        # padding to preserve aspect ratio
        hp = int(w_1/ratio_f - h_1)
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


dict_name_to_filter = {"PIL": {"bicubic": Image.BICUBIC,
                               "bilinear": Image.BILINEAR,
                               "nearest": Image.NEAREST,
                               "lanczos": Image.LANCZOS,
                               "box": Image.BOX}}


def resize(img, size):
    return make_resizer("PIL", "bilinear", size)(img)


def smallest_max_size(img, max_size):
    def py3round(number):
        """Unified rounding in all python versions."""
        if abs(round(number) - number) == 0.5:
            return int(2.0 * round(number / 2.0))

        return int(round(number))

    height, width = img.shape[:2]

    scale = max_size / float(min(width, height))

    if scale != 1.0:
        size = tuple(py3round(dim * scale) for dim in (height, width))
        img = resize(img, size[::-1])

    return img


def make_resizer(library, filter, output_size):
    if library == "PIL":
        s1, s2 = output_size

        def resize_single_channel(x_np):
            img = Image.fromarray(x_np.astype(np.float32), mode="F")
            img = img.resize(output_size, resample=dict_name_to_filter[library][filter])
            return np.asarray(img)[..., None]

        def func(x):
            x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
            x = np.concatenate(x, axis=2).astype(np.float32)
            return x

    elif library == "PyTorch":
        import warnings

        # ignore the numpy warnings
        warnings.filterwarnings("ignore")

        def func(x):
            x = torch.Tensor(x.transpose((2, 0, 1)))[None, ...]
            x = F.interpolate(x, size=output_size, mode=filter, align_corners=False)
            x = x[0, ...].cpu().data.numpy().transpose((1, 2, 0)).clip(0, 255)
            return x

    else:
        raise NotImplementedError("library [%s] is not include" % library)
    return func


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    See ``ToTensor`` for more details.
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """
    if not (_is_numpy_image(pic)):
        raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))

    # handle numpy array
    img = torch.from_numpy(pic.transpose((2, 0, 1)))
    # backward compatibility
    if isinstance(img, torch.ByteTensor) or img.dtype == torch.uint8:
        return img.float().div(255)
    else:
        return img

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def rot_img(x, theta, dtype):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)
    return x

def split_range(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n))