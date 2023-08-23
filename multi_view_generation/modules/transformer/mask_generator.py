import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from functools import lru_cache
from scipy.spatial import distance
import numpy as np
from einops import repeat, rearrange
import torch.nn.functional as F
import torch
from multi_view_generation.bev_utils import util, save_binary_as_image
from multi_view_generation.modules.transformer.permuter import get_seq_pixel_mappings, layout_to_pattern, pattern_to_layout, random_pattern_from_probability_matrix, _generate_2d_grid, get_col_angles
from multi_view_generation.bev_utils import Dataset
from multi_view_generation.modules.transformer.mingpt_sparse import GPTConfig
from multi_view_generation.bev_utils import Cameras
from multi_view_generation.modules.transformer.mingpt_sparse import get_bev_grid
from multi_view_generation.modules.transformer.mingpt_sparse import generate_grid
import torch.nn.functional as F

def plot_pattern_multi_cam(cfg: GPTConfig, pattern, middle_point, split_sequence_order=False, ignore_cond=False, filename=None):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    
    pattern = pattern.detach().cpu().numpy()
    fig, axs = plt.subplots(nrows=1 + cfg.num_cams // 3, ncols=3, figsize=(30, 20))
    extent = (0, pattern.shape[0], pattern.shape[1], 0)
    axs[0, 0].axis('off')
    axs[0, 2].axis('off')
    axs[0, 1].imshow(pattern, extent=extent, cmap='hot', vmin=0, vmax=1)
    axs[0, 1].grid(linewidth=2)
    axs[0, 1].set_frame_on(False)
    axs[0, 1].xaxis.set_major_locator(mticker.MultipleLocator(cfg.num_cam_tokens))
    axs[0, 1].yaxis.set_major_locator(mticker.MultipleLocator(cfg.num_cam_tokens))
    axs[0, 1].set_title("Full attn_mask matrix")

    all_cam_names = [["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]]
    if cfg.num_cams == 6:
        all_cam_names.append(["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"])
    for row_num, cam_side in enumerate(all_cam_names):
        for i, cam_name in zip(range(cfg.num_cams), cam_side):
            j = cfg.cam_name_to_idx[cam_name]
            extent = (0, cfg.cam_latent_w, cfg.cam_latent_h, 0)
            if split_sequence_order:
                axs[row_num + 1, i].imshow(pattern[cfg.num_cond_tokens + middle_point, cfg.num_cond_tokens + j::cfg.num_cams].reshape(cfg.cam_latent_h, cfg.cam_latent_w), extent=extent, vmin=0, vmax=1)
            elif ignore_cond:
                axs[row_num + 1, i].imshow(pattern[middle_point, j * cfg.num_cam_tokens: (j + 1) * cfg.num_cam_tokens].reshape(cfg.cam_latent_h, cfg.cam_latent_w), extent=extent, cmap='hot', vmin=0, vmax=1)
            else:
                axs[row_num + 1, i].imshow(pattern[cfg.num_cond_tokens + middle_point, cfg.num_cond_tokens + j * cfg.num_cam_tokens: cfg.num_cond_tokens +  (j + 1) * cfg.num_cam_tokens].reshape(cfg.cam_latent_h, cfg.cam_latent_w), extent=extent, cmap='hot', vmin=0, vmax=1)
            axs[row_num + 1, i].grid(linewidth=2)
            axs[row_num + 1, i].set_frame_on(False)
            axs[row_num + 1, i].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            axs[row_num + 1, i].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            axs[row_num + 1, i].set_title(cam_name)

    fig.suptitle('Attention Mask', fontsize=16)
    if filename is not None:
        plt.savefig(filename, dpi=350)
    elif split_sequence_order:
        plt.savefig('lower_triangular_sequnce.png', dpi=350)
    else:
        plt.savefig('mixed_sequence.png', dpi=350)
    plt.close()

def pad_with_conf(pattern, bev_cond_seq_len, value=True):
    pattern = F.pad(pattern, (0, 0, bev_cond_seq_len, 0), mode='constant', value=False)
    pattern = F.pad(pattern, (bev_cond_seq_len, 0, 0, 0), mode='constant', value=value)
    return pattern

def get_bev_weights(cfg: GPTConfig, angles):
    bev_latent_h, bev_latent_w = cfg.bev_latent_res[0], cfg.bev_latent_res[1]
    assert cfg.num_cond_tokens == bev_latent_h * bev_latent_w
    pixel_to_seq = torch.zeros((bev_latent_h, bev_latent_w), dtype=torch.long)
    seq_to_pixel = rearrange(torch.stack(torch.meshgrid(torch.arange(bev_latent_h), torch.arange(bev_latent_w)), -1), 'h w d -> (h w) d')
    pixel_to_seq[seq_to_pixel[:, 0], seq_to_pixel[:, 1]] = torch.arange(seq_to_pixel.shape[0])
    seq_to_pixel = seq_to_pixel.float()
    center_point_h, center_point_w = (bev_latent_h // 2) - 0.5, (bev_latent_w // 2) - 0.5
    seq_to_pixel[:, 0] *= -1
    seq_to_pixel[:, 0] += center_point_h
    seq_to_pixel[:, 1] -= center_point_w
    bev_angle = torch.remainder(torch.atan2(seq_to_pixel[:, 0], seq_to_pixel[:, 1]) - torch.pi / 2, 2 * torch.pi)
    sim_matrix = torch.from_numpy(1 - distance.cdist(np.stack([np.cos(angles), np.sin(angles)], 1), np.stack([np.cos(bev_angle), np.sin(bev_angle)], 1), metric='cosine'))
    return (sim_matrix + 1) / 2


def get_image_direction_vectors(cfg: GPTConfig):
    data = torch.load(f'pretrained/cam_data_{cfg.dataset_name}.pt')
    mapper = {k:v for v,k in enumerate(Cameras.NUSCENES_CAMERAS if cfg.dataset == Dataset.NUSCENES else cfg.cam_names)}

    image_plane = generate_grid(cfg.cam_latent_h, cfg.cam_latent_w)[None]
    image_plane[:, :, 0] *= 1600
    image_plane[:, :, 1] *= 900

    E_inv = data['extrinsics'][[0]].inverse()
    I_inv = data['intrinsics'][[0]].inverse()

    pixel_flat = rearrange(image_plane, '... h w -> 1 ... (h w)')
    cam = I_inv @ pixel_flat                                                # b n 3 (h w)
    cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                     # b n 4 (h w)
    d = E_inv @ cam                                                         # b n 4 (h w)
    c = E_inv[..., -1:, None]
    output = d.squeeze() - repeat(c.squeeze(), 'n c -> n c (h w)', h=cfg.cam_latent_h, w=cfg.cam_latent_w)
    output = rearrange(output, 'n c (h w) -> n (h w) c', h=cfg.cam_latent_h, w=cfg.cam_latent_w)[..., :3]
    output = torch.stack([output[mapper[cam_name]] for cam_name in cfg.cam_names], dim=0)
    output = rearrange(output, 'n (h w) c -> (n h w) c', h=cfg.cam_latent_h, w=cfg.cam_latent_w)
    output = F.normalize(output, dim=1)
    return output

def get_cosine_similarity(a, b):
    cos_sim = F.normalize(a) @ F.normalize(b).t()
    return (cos_sim + 1) / 2

def get_image_similarity(cfg: GPTConfig):
    output = get_image_direction_vectors(cfg)
    return get_cosine_similarity(output, output)

def get_bev_sim(cfg: GPTConfig):
    bev_grid = get_bev_grid(cfg)
    bev_grid = rearrange(bev_grid, 'c h w -> (h w) c')
    bev_grid[:, 2] = 0
    bev_grid = F.normalize(bev_grid, dim=1)

    output = get_image_direction_vectors(cfg)

    return get_cosine_similarity(output, bev_grid)

@lru_cache(maxsize=1000)
def outward_pattern(cfg: GPTConfig, return_camera_bias_matrix=False):
    i, j = _generate_2d_grid(cfg.num_img_tokens, cfg.num_img_tokens)
    sliding_window_start = repeat(torch.where(torch.arange(cfg.num_img_tokens) - cfg.window_len >= 0, torch.arange(cfg.num_img_tokens) - cfg.window_len, 0),  '... -> ... d', d=cfg.num_img_tokens)
    sliding_window_end = repeat(torch.arange(cfg.num_img_tokens) + 1, '... -> ... d', d=cfg.num_img_tokens)
    sliding_window = (sliding_window_start <= j) & (j < sliding_window_end)

    def mask_from_indices(deocde_mask):
        decode_indices = repeat(torch.arange(cfg.num_img_tokens) if cfg.causal_order else cfg.forward_shuffle_idx, '... -> d ...', d=cfg.num_img_tokens).clone()  # forward_shuffle_idx contains decoding order, repeat over rows
        decode_indices[~deocde_mask] = -1  # We only want to look at certain decode indices
        decode_indices = decode_indices[torch.arange(cfg.num_img_tokens) if cfg.causal_order else cfg.backward_shuffle_idx]  # We use arange to create rows passed to this func so backward to the original indices
        row_col_decode = torch.stack((i, decode_indices), dim=-1)  # (orig_seq_len, orig_seq_len, 2)
        row_col_decode = row_col_decode[row_col_decode[:, :, 1] >= 0]  # (n, 2), only include elements we wish to decode
        output_mask = torch.full((cfg.num_img_tokens, cfg.num_img_tokens), False, dtype=torch.bool)
        output_mask[row_col_decode[:, 0], row_col_decode[:, 1]] = True  # We keep the row index from above and select the column from the value
        return output_mask

    window_pattern = mask_from_indices(sliding_window)
    allowed_pattern = mask_from_indices(j < sliding_window_end)  # Maintain causality

    if cfg.legacy_prob_matrix:
        i = get_seq_pixel_mappings(cfg)[1][:, 1]  # seq_to_pixel: ((num_cams x cam_seq_len), (cam, h, w)) [select row]
        j = get_seq_pixel_mappings(cfg)[1][:, [0, 2]]
        angles = get_col_angles(cfg)[j[:, 0], j[:, 1]]
        ii = torch.stack([i.flatten(), torch.zeros_like(i.flatten())], 1).float()
        jj = np.stack([np.cos(angles), np.sin(angles)], 1)
        d = torch.from_numpy(np.rad2deg(distance.cdist(jj, jj, metric='cosine'))) # BUG!!!

        horiz_d = torch.cdist(ii, ii, p=2.0)
        sigma = 4.0
        prob_matrix = torch.exp(-0.5 * sigma ** (-2.0) * (d + horiz_d))
    else:
        prob_matrix = get_image_similarity(cfg)
        # save_binary_as_image(output, f'{cfg.output_dir}/cosine_dist.png')
    if cfg.causal_order:
        prob_matrix = prob_matrix[:, cfg.forward_shuffle_idx][cfg.forward_shuffle_idx, :]

    prob_matrix[~allowed_pattern] = 0

    if cfg.plot and cfg.dataset == Dataset.NUSCENES:
        plot_pattern_multi_cam(cfg, prob_matrix[cfg.backward_shuffle_idx, :][:, cfg.backward_shuffle_idx], get_seq_pixel_mappings(cfg)[0][cfg.cam_names.index('CAM_FRONT_RIGHT'), cfg.cam_latent_h // 2, cfg.cam_latent_w // 2], ignore_cond=True, filename=f'{cfg.output_dir}/prob_plot_cam_order.png')

    prob_matrix_ = prob_matrix.clone()
    prob_matrix = F.pad(prob_matrix, (0, cfg.num_pad_tokens, 0, cfg.num_pad_tokens), mode='constant', value=0)
    prob_matrix = torch.clamp(prob_matrix, min=0, max=1)
    if return_camera_bias_matrix:
        prob_matrix = prob_matrix.clone()
        prob_matrix = pad_with_conf(prob_matrix, cfg.num_cond_tokens, value=1.0)
        if cfg.legacy_prob_matrix:
            sim_matrix = get_bev_weights(cfg, angles[cfg.forward_shuffle_idx])
        else:
            sim_matrix = get_bev_sim(cfg)[cfg.forward_shuffle_idx, :]
        if cfg.num_pad_tokens == 0:
            prob_matrix[cfg.num_cond_tokens:, :cfg.num_cond_tokens] = sim_matrix
        else:
            prob_matrix[cfg.num_cond_tokens:-cfg.num_pad_tokens, :cfg.num_cond_tokens] = sim_matrix
        if cfg.plot:
            idx = cfg.backward_shuffle_idx[get_seq_pixel_mappings(cfg)[0][0, cfg.cam_latent_h // 2, cfg.cam_latent_w // 2]]
            save_binary_as_image(prob_matrix, f'{cfg.output_dir}/camera_bias_prob_matrix.png')
            save_binary_as_image(sim_matrix[idx].reshape(cfg.bev_latent_res[0], cfg.bev_latent_res[1]), f'{cfg.output_dir}/bev_to_cam_bias.png')
        return prob_matrix

    prob_matrix = pad_with_conf(prob_matrix, cfg.num_cond_tokens, value=0.5)
    prob_layout = torch.nn.functional.avg_pool2d(prob_matrix.unsqueeze(0).to(torch.float), kernel_size=cfg.sparse_block_size, stride=cfg.sparse_block_size).squeeze()

    window_pattern = F.pad(window_pattern, (0, cfg.num_pad_tokens, 0, cfg.num_pad_tokens), mode='constant', value=0)
    static_pattern = pad_with_conf(window_pattern, cfg.num_cond_tokens, value=False) # Adds a column of all 1s to the left
    if cfg.num_pad_tokens != 0:
        static_pattern[-cfg.num_pad_tokens:, 0] = 1
        static_pattern[-cfg.num_pad_tokens:, 1:] = 0 # To avoid NaN, at least 1 element must be unmasked
    static_layout = pattern_to_layout(static_pattern, cfg.sparse_block_size)

    allowed_pattern = F.pad(allowed_pattern, (0, cfg.num_pad_tokens, 0, cfg.num_pad_tokens), mode='constant', value=False)
    allowed_pattern = pad_with_conf(allowed_pattern, cfg.num_cond_tokens, value=True) # Adds a column of all 1s to the left
    if cfg.num_pad_tokens != 0:
        allowed_pattern[-cfg.num_pad_tokens:, 1:] = 0 # To avoid NaN, at least 1 element must be unmasked
    allowed_pattern = repeat(allowed_pattern.float(), '... -> num_heads ...', num_heads=cfg.num_heads)

    if cfg.plot:
        save_binary_as_image(allowed_pattern[0].bool(), f'{cfg.output_dir}/allowed_pattern.png')
        save_binary_as_image(static_layout.bool(), f'{cfg.output_dir}/static_layout.png')
        save_binary_as_image(prob_layout, f'{cfg.output_dir}/prob_layout.png')
        save_binary_as_image(prob_matrix_[:, cfg.backward_shuffle_idx][cfg.backward_shuffle_idx, :], f'{cfg.output_dir}/prob_matrix_cam_order.png')

    return allowed_pattern, static_layout, prob_layout, prob_matrix


def multi_outward_pattern(cfg: GPTConfig):
    allowed_pattern, static_layout, prob_layout, prob_matrix = outward_pattern(cfg)

    layouts = []
    for _ in range(cfg.num_heads):
        num_to_sample = int(((prob_layout.shape[0] * prob_layout.shape[1]) * cfg.density) - static_layout.sum())
        # int(cfg.density * ((prob_layout > 0) & (prob_layout < 1)).sum())
        sampled_layout = random_pattern_from_probability_matrix(prob_layout, num_to_sample)
        sampled_layout[prob_layout == 0] = False
        pattern = static_layout | sampled_layout
        layouts.append(pattern)
    layouts = torch.stack(layouts)

    if cfg.plot:
        computed_pattern = layout_to_pattern(layout=layouts[-1], block_size=cfg.sparse_block_size)
        result_pattern = computed_pattern.to(torch.bool) & allowed_pattern[0].bool()
        
        test = layout_to_pattern(layout=static_layout, block_size=cfg.sparse_block_size).bool() | random_pattern_from_probability_matrix(prob_matrix, int(cfg.density * (prob_matrix > 0).sum())).bool()
        test = test[cfg.num_cond_tokens:-cfg.num_pad_tokens, cfg.num_cond_tokens:-cfg.num_pad_tokens]
        test = test[cfg.backward_shuffle_idx, :][:, cfg.backward_shuffle_idx]
        plot_pattern_multi_cam(cfg, test, get_seq_pixel_mappings(cfg)[0][cfg.cam_names.index('CAM_FRONT_RIGHT'), cfg.cam_latent_h // 2, cfg.cam_latent_w // 2], ignore_cond=True, filename=f'{cfg.output_dir}/static_layout_prob_matrix_cam_order.png')

        test = result_pattern.bool()
        test = test[cfg.num_cond_tokens:-cfg.num_pad_tokens, cfg.num_cond_tokens:-cfg.num_pad_tokens]
        test = test[cfg.backward_shuffle_idx, :][:, cfg.backward_shuffle_idx]
        plot_pattern_multi_cam(cfg, test, get_seq_pixel_mappings(cfg)[0][cfg.cam_names.index('CAM_FRONT_RIGHT'), cfg.cam_latent_h // 2, cfg.cam_latent_w // 2], 
        ignore_cond=True, filename=f'{cfg.output_dir}/result_plot_cam_order.png')

        print(f'Pattern Coverage: {pattern.sum() / (pattern.shape[0] * pattern.shape[1])}, Layout Coverage: {computed_pattern.sum() / (computed_pattern.shape[0] * computed_pattern.shape[1])}, Change: {computed_pattern.sum() / pattern.sum()}x')
        save_binary_as_image(sampled_layout.bool(), f'{cfg.output_dir}/sampled_layout.png')
        save_binary_as_image(result_pattern.bool(), f'{cfg.output_dir}/result_pattern.png')
        save_binary_as_image(np.concatenate([util.return_binary_as_image(layouts[i]) for i in range(layouts.shape[0])], axis=1), filename=f'{cfg.output_dir}/all_layouts.png')
        print(round((layouts.sum() / torch.prod(torch.tensor(list(layouts.size())))).item(), 3))

    return layouts, allowed_pattern.clone()

