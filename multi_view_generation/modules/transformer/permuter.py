import torch
import torch.nn as nn
from multi_view_generation.bev_utils import Cameras
from multi_view_generation.bev_utils import Dataset
from multi_view_generation.modules.transformer.mingpt_sparse import GPTConfig
from einops import rearrange, repeat
import itertools
from multi_view_generation.bev_utils.nuscenes_helper import compute_pixel_ray_directions
import numpy as np
class AbstractPermuter(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, reverse=False):
        raise NotImplementedError


class Identity(AbstractPermuter):
    def __init__(self):
        super().__init__()

    def forward(self, x, reverse=False):
        return x


def get_seq_pixel_mappings(cfg: GPTConfig):
    pixel_to_seq = torch.zeros((cfg.num_cams, cfg.cam_latent_h, cfg.cam_latent_w), dtype=torch.long)
    seq_to_pixel = rearrange(torch.stack(torch.meshgrid(torch.arange(cfg.num_cams), torch.arange(cfg.cam_latent_h), torch.arange(cfg.cam_latent_w)), -1), 'cam h w d -> (cam h w) d')
    pixel_to_seq[seq_to_pixel[:, 0], seq_to_pixel[:, 1], seq_to_pixel[:, 2]] = torch.arange(seq_to_pixel.shape[0])
    return pixel_to_seq, seq_to_pixel


class CustomPermuter(AbstractPermuter):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

        pixel_to_seq, _ = get_seq_pixel_mappings(cfg)
        center_pixel = cfg.cam_latent_w // 2

        if cfg.dataset == Dataset.NUSCENES:
            if cfg.num_cams == 3:
                cams = [("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")]
                camera_index = Cameras.NUSCENES_ABLATION_CAMERAS
            else:
                cams = [("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"), ("CAM_BACK_RIGHT", "CAM_BACK",  "CAM_BACK_LEFT")]
                camera_index = Cameras.NUSCENES_CAMERAS

            indices = []
            for i in range(cfg.cam_latent_h):
                dir_idxs = []

                for l_cam, c_cam, r_cam in cams:
                    list_to_append = []

                    left_sequence_left = pixel_to_seq[camera_index.index(l_cam), i, :].tolist()[::-1]
                    right_sequence_right = pixel_to_seq[camera_index.index(r_cam), i, :].tolist()
                    left_sequence_center = pixel_to_seq[camera_index.index(c_cam), i, :center_pixel].tolist()[::-1]

                    if cfg.cam_latent_w % 2 == 0:
                        right_sequence_center = pixel_to_seq[camera_index.index(c_cam), i, center_pixel:].tolist()
                    else:
                        center_index = pixel_to_seq[camera_index.index(c_cam), i, center_pixel].item()
                        list_to_append.append(center_index)
                        right_sequence_center = pixel_to_seq[camera_index.index(c_cam), i, center_pixel + 1:].tolist()

                    dir_idxs.append([*list_to_append, *list(itertools.chain.from_iterable(zip([*left_sequence_center, *left_sequence_left], [*right_sequence_center, *right_sequence_right])))])

                row_idxs = list(itertools.chain.from_iterable(zip(*dir_idxs)))
                indices.extend(row_idxs)
        else:
            indices = []
            for i in range(cfg.cam_latent_h):
                for j, cam in enumerate(cfg.cam_names):
                    sequence_right = pixel_to_seq[j, i, :].tolist()
                    indices.extend(sequence_right)

        if cfg.causal_order:
            indices = torch.tensor(indices)
        else:
            indices = torch.arange(cfg.num_img_tokens)
        self.register_buffer('forward_shuffle_idx', indices)
        self.register_buffer('backward_shuffle_idx', torch.argsort(indices))

    def forward(self, x, reverse=False):
        if not reverse:
            return x[:, self.forward_shuffle_idx]
        else:
            return x[:, self.backward_shuffle_idx]


def layout_to_pattern(layout: torch.Tensor, block_size: int):
    r"""
    create a pattern of shape [heads, seq, seq] out of a blocksparse
    layout of shape [heads, seq/block_size, seq/block_size]
    """
    return torch.kron(layout, torch.ones(block_size, block_size))

def pattern_to_layout(mask: torch.Tensor, block_size: int) -> torch.Tensor:
    r"""
    Given a mask pattern and blocksize, return the corresponding layout
    which makes sure that all the positives in the mask are covered
    """
    assert mask.ndim >= 2, "We're expecting [Heads, Seq, Seq] or [Seq, Seq]"
    _should_squeeze = False

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
        _should_squeeze = True

    assert (
        mask.shape[1] % block_size == 0 and mask.shape[2] % block_size == 0
    ), "We're only handling masks divisible by block_size"

    # Now mark the mask
    layout = torch.nn.functional.max_pool2d(
        mask.to(torch.float), kernel_size=block_size, stride=block_size
    )
    layout = layout.to(torch.long)

    if _should_squeeze:
        layout.squeeze_(0)

    return layout

def random_pattern_from_probability_matrix(dist_matrix, nnz):
    att = torch.zeros_like(dist_matrix, dtype=torch.bool)
    # PyTorch multinomial wrongly doesn't support sampling when number of categories
    # is > 2^24, arguing that it's because it's the max representable consecutive element
    # in fp32 and that the kernels use float32. This is actually not true, and the kernels
    # should work fine if double tensor is passed on CPU. This is a bug that was introduced
    # in https://github.com/pytorch/pytorch/commit/bf04c2ca2f591d98ce57816f0ef0cd20a21bbf66
    # when unifying the checks between CPU and CUDA. For now, just fall-back to numpy
    if dist_matrix.numel() > 2**24:
        dist_matrix = dist_matrix.double()
        dist_matrix /= dist_matrix.sum()
        idxs = np.random.choice(
            dist_matrix.numel(), nnz, p=dist_matrix.flatten(), replace=False
        )
        idxs = torch.as_tensor(idxs)
    else:
        idxs = torch.multinomial(dist_matrix.flatten(), nnz, replacement=False)
    att.view(-1)[idxs] = True
    return att

def _generate_2d_grid(H, W):
    i = torch.arange(H)
    j = torch.arange(W)
    i, j = torch.meshgrid(i, j)
    return i, j

CAM_DATA = {'CAM_FRONT': (1266.417203046554, 1266.417203046554, 0.005684811144346602), 'CAM_BACK': (809.2209905677063, 809.2209905677063, 3.1391709219861887), 'CAM_FRONT_RIGHT': (1260.8474446004698, 1260.8474446004698, 5.298742851167251), 'CAM_FRONT_LEFT': (1272.5979470598488, 1272.5979470598488, 0.9627404474321728), 'CAM_BACK_RIGHT': (1259.5137405846733, 1259.5137405846733, 4.349372983905386), 'CAM_BACK_LEFT': (1256.7414812095406, 1256.7414812095406, 1.895431863668132)}

def get_col_angles(cfg: GPTConfig):
    col_angles = []
    for cam_idx in range(len(Cameras.NUSCENES_CAMERAS)):
        fx, fy, cam_angle = CAM_DATA[Cameras.NUSCENES_CAMERAS[cam_idx]]
        img_w, img_h = 1600, 900
        col_angles_ = [-compute_pixel_ray_directions(np.array([[img_w * ((i + 0.5) / cfg.cam_latent_w), img_h / 2]]), fx, fy, img_h, img_w)[0, 0] for i in range(cfg.cam_latent_w)]
        col_angles_ = [np.mod(cam_angle + x, 2 * np.pi).astype(np.float32) for x in col_angles_]
        col_angles.append(col_angles_)

    return np.array(col_angles)
