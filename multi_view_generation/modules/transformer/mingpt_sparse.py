"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, Any
from deepspeed.ops.sparse_attention import SparsityConfig
import logging
from multi_view_generation.bev_utils.util import Cameras
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat, reduce
import numpy as np
from multi_view_generation.modules.transformer.sparse_self_attention import SparseSelfAttention
from multi_view_generation.bev_utils import Dataset
logger = logging.getLogger(__name__)

@dataclass(unsafe_hash=True)
class GPTConfig():
    """Config for GPT and sparse transformer modules/masks"""
    embd_pdrop: float
    resid_pdrop: float
    attn_pdrop: float
    num_layers: int
    num_heads: int
    num_embed: int
    hidden_size: int
    vocab_size: int  
    cond_vocab_size: int
    num_cams: int
    window_len: int
    density: float
    sparse_block_size: int
    n_unmasked: int
    backend: str
    plot: bool
    cam_res: Tuple[int, int]
    cam_latent_res: Tuple[int, int]
    bev_latent_res: Tuple[int, int]
    camera_bias: bool
    bev_embed: bool
    image_embed: bool
    cam_names: Cameras
    cam_name_to_idx: Dict[str, Any] = field(init=False, compare=False)
    num_cond_tokens: int = field(init=False)
    num_cam_tokens: int = field(init=False)
    num_img_tokens: int = field(init=False)
    num_pad_tokens: int = field(init=False)
    gpt_block_size: int = field(init=False)
    cam_latent_h: int = field(init=False)
    cam_latent_w: int = field(init=False)
    attention_mask: torch.FloatTensor = field(init=False, compare=False) # Mask of size [gpt_block_size, gpt_block_size], 0 means do not attend to position (-inf)
    causal_order: bool = False
    forward_shuffle_idx: torch.LongTensor = field(init=False, compare=False)
    backward_shuffle_idx: torch.LongTensor = field(init=False, compare=False)
    layout: Optional[torch.Tensor] = field(init=False, compare=False, repr=False)
    only_front_cams: bool = field(init=False, compare=False, repr=False)
    dataset_name: str = field(init=False)
    output_dir: str = 'output'
    prob_matrix = None
    legacy_prob_matrix: bool = True
    cam_intrinsics: torch.FloatTensor = None
    cam_extrinsics: torch.FloatTensor = None
    dataset: Dataset = Dataset.NUSCENES

    def __post_init__(self):
        self.dataset = Dataset(self.dataset) if isinstance(self.dataset, int) else Dataset[self.dataset]
        self.dataset_name = self.dataset.name.lower()
        self.cam_names = Cameras[self.cam_names]
        assert len(self.cam_names) == self.num_cams

        self.cam_name_to_idx = {k:v for v,k in enumerate(self.cam_names)}
        
        self.cam_latent_h = self.cam_latent_res[0]
        self.cam_latent_w = self.cam_latent_res[1]
        self.num_cond_tokens = self.bev_latent_res[0] * self.bev_latent_res[1]
        self.num_cam_tokens = self.cam_latent_h * self.cam_latent_w
        self.num_img_tokens = self.num_cam_tokens * self.num_cams
        self.gpt_block_size = self.sparse_block_size * int(np.ceil((self.num_img_tokens + self.num_cond_tokens) / self.sparse_block_size))
        self.num_pad_tokens = self.gpt_block_size - (self.num_img_tokens + self.num_cond_tokens)

        from multi_view_generation.modules.transformer.permuter import CustomPermuter
        from multi_view_generation.modules.transformer.mask_generator import outward_pattern
        permuter = CustomPermuter(self)
        
        self.forward_shuffle_idx = permuter.forward_shuffle_idx
        self.backward_shuffle_idx = permuter.backward_shuffle_idx

        _, mask = self.get_mask()
        self.attention_mask = mask[0]

        if self.camera_bias:
            prob_matrix = outward_pattern(self, return_camera_bias_matrix=True)
            self.prob_matrix = prob_matrix

    def get_mask(self):
        from multi_view_generation.modules.transformer.mask_generator import multi_outward_pattern
        return multi_outward_pattern(self)


    def forward_permuter(self, x):
        return x[:, self.forward_shuffle_idx]

    def backward_permuter(self, x):
        return x[:, self.backward_shuffle_idx]


def get_bev_grid(
    cfg: GPTConfig,
    offset: int = 0,
):
    # each decoder block upsamples the bev embedding by a factor of 2
    h = cfg.bev_latent_res[0]
    w = cfg.bev_latent_res[1]

    # bev coordinates
    grid = generate_grid(h, w).squeeze(0)
    grid[0] = w * grid[0]
    grid[1] = h * grid[1]

    # map from bev coordinates to ego frame
    sh = h / 80 # 80m
    sw = w / 80 # 80m

    V = [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]
    V_inv = torch.FloatTensor(V).inverse()                                  # 3 3
    grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')                      # 3 (h w)
    grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)                    # 3 h w
    return grid

class CustomSparsityConfig(SparsityConfig):
    def __init__(self,
                 num_heads,
                 layout,
                 block,
                 different_layout_per_head=True):

        super().__init__(num_heads, block, different_layout_per_head)
        self.layout = layout

    def make_layout(self, seq_len):
        return self.layout


class CustomSparseSelfAttention(nn.Module):
    """Implements Sparse Self Attention layer of Bert model based on https://github.com/microsoft/DeepSpeedExamples/blob/master/bing_bert/nvidia/modelingpreln.py#L373"""

    def __init__(self, cfg: GPTConfig):
        super(CustomSparseSelfAttention, self).__init__()

        self.cfg = cfg
        if cfg.hidden_size % cfg.num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (cfg.hidden_size,
                                cfg.num_heads))
        self.num_attention_heads = cfg.num_heads
        self.attention_head_size = int(cfg.hidden_size / cfg.num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(cfg.hidden_size, self.all_head_size)
        self.key = nn.Linear(cfg.hidden_size, self.all_head_size)
        self.value = nn.Linear(cfg.hidden_size, self.all_head_size)
        layout, _ = self.cfg.get_mask()
        self.sparse_self_attention = SparseSelfAttention(CustomSparsityConfig(num_heads=self.cfg.num_heads, layout=layout, block=self.cfg.sparse_block_size), attn_mask_mode='mul')

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, camera_bias=None):
        """Applies forward phase of bert sparse self attention

        Arguments:
            hidden_states: required: hidden_states tensor of the bert model
            attn_mask: required: a mask tensor of size (SequenceLength X SequenceLength); currently only 2D is supported

        Return:
             context_layer: a dense tensor containing attention context
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        context_layer = self.sparse_self_attention(query_layer,
                                                   key_layer,
                                                   value_layer,
                                                   attn_mask=attention_mask,
                                                   add_mask=camera_bias)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.num_embed)
        self.ln2 = nn.LayerNorm(cfg.num_embed)
        self.cfg = cfg

        if self.cfg.backend == "deepspeed":
            self.attention = CustomSparseSelfAttention(cfg)
            self.attention_mask = self.cfg.attention_mask
        elif self.cfg.backend == "pytorch":
            self.multi_head = nn.MultiheadAttention(cfg.num_embed, cfg.num_heads, batch_first=True)
            self.attention_mask = repeat(self.cfg.attention_mask.to(torch.long), '... -> head ...', head=self.cfg.num_heads)
        else:
            raise ValueError()

        self.mlp = nn.Sequential(
            nn.Linear(cfg.num_embed, 4 * cfg.num_embed),
            nn.GELU(),
            nn.Linear(4 * cfg.num_embed, cfg.num_embed),
            nn.Dropout(cfg.resid_pdrop),
        )

    def forward(self, input_data):
        x, camera_bias = input_data
        x = self.ln1(x)

        if self.cfg.backend == "deepspeed":
            attn = self.attention(x, self.attention_mask.to(x.device), camera_bias)
        elif self.cfg.backend == "pytorch":
            attn_mask = repeat(self.attention_mask, 'head ... -> (head batch) ...', batch=x.shape[0], head=self.cfg.num_heads).to(x.device)
            attn = self.multi_head(query=x, key=x, value=x, attn_mask=attn_mask.to(torch.bool))[0]

        x = x + attn
        x = x + self.mlp(self.ln2(x))

        return (x, camera_bias)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, cfg: GPTConfig, **kwargs):
        super().__init__()

        self.cfg = cfg

        self.x_tok_emb = nn.Embedding(cfg.vocab_size + 1, cfg.num_embed)
        self.cond_tok_emb = nn.Embedding(cfg.cond_vocab_size, cfg.num_embed)

        self.x_pos_emb = nn.Parameter(torch.zeros(1, cfg.num_img_tokens, cfg.num_embed))
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, cfg.num_cond_tokens, cfg.num_embed))

        self.drop = nn.Dropout(cfg.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.num_layers)])
        # decoder head
        self.ln_f = nn.LayerNorm(cfg.num_embed)
        self.head = nn.Linear(cfg.num_embed, cfg.vocab_size, bias=False)

        if self.cfg.image_embed:
            # 1 1 3 h w
            image_plane = generate_grid(cfg.cam_latent_h, cfg.cam_latent_w)[None]
            image_plane[:, :, 0] *= cfg.cam_res[0]
            image_plane[:, :, 1] *= cfg.cam_res[1]

            self.register_buffer('image_plane', image_plane, persistent=False)
            self.img_embed = nn.Conv2d(4, cfg.num_embed, 1, bias=False)
            self.cam_embed = nn.Conv2d(4, cfg.num_embed, 1, bias=False)

        if self.cfg.bev_embed:
            self.register_buffer('bev_grid', get_bev_grid(self.cfg))
            self.bev_embed = nn.Conv2d(2, cfg.num_embed, 1)
            self.bev_cam_pos_emb = nn.Parameter(torch.zeros(1, cfg.num_cams, cfg.num_cond_tokens, cfg.num_embed))

        if self.cfg.camera_bias:
            self.camera_bias_emb = nn.Parameter(torch.zeros(1, torch.tril_indices(cfg.gpt_block_size, cfg.gpt_block_size).shape[1]))

        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, cam_indices, bev_indices, batch, sampling, **kwargs):
        # forward the GPT model
        I_inv = batch['intrinsics_inv']  # b cam 3 3
        E_inv = batch['extrinsics_inv']  # b cam 4 4

        b, num_cams = I_inv.shape[0], I_inv.shape[1]

        h, w = self.cfg.cam_latent_h, self.cfg.cam_latent_w

        if not sampling:
            cam_indices[:, -1, -1] = self.cfg.vocab_size

        x_tok_emb_ = self.x_tok_emb(cam_indices)
        if self.cfg.image_embed:
            pixel = self.image_plane                                                # b cam 3 h w
            _, _, _, h, w = pixel.shape

            c = E_inv[..., -1:]                                                     # b cam 4 1
            c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]                # (b cam) 4 1 1
            c_embed = self.cam_embed(c_flat)                                        # (b cam) d 1 1

            pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (h w)
            cam = I_inv @ pixel_flat                                                # b cam 3 (h w)
            cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                     # b cam 4 (h w)
            d = E_inv @ cam                                                         # b cam 4 (h w)
            d_flat = rearrange(d, 'b cam d (h w) -> (b cam) d h w', h=h, w=w)       # (b cam) 4 h w
            d_embed = self.img_embed(d_flat)                                        # (b cam) d h w

            img_embed = d_embed - c_embed                                           # (b cam) d h w
            img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b cam) d h w
            img_embed = rearrange(img_embed, '(b cam) d h w -> b cam (h w) d', b=b, cam=num_cams)
            x_tok_emb_ = x_tok_emb_ + img_embed

        cond_tok_emb_ = self.cond_tok_emb(bev_indices)
        if self.cfg.bev_embed:
            c_expanded = repeat(rearrange(c_embed, '(b n) d 1 1 -> b n d', b=b, n=num_cams), 'b n d -> b n cond_tokens d', cond_tokens=self.cfg.num_cond_tokens)
            grid_embed = rearrange(self.bev_embed(self.bev_grid[None, :2]), '1 d h w -> 1 (h w) d')
            bev_cam_embed = reduce(self.bev_cam_pos_emb + c_expanded, 'b n cond_tokens d -> b cond_tokens d', 'sum')
            bev_embed = grid_embed - bev_cam_embed
            cond_tok_emb_ = cond_tok_emb_ + bev_embed

        x_tok_emb_ = rearrange(x_tok_emb_, ' b cam (h w) d -> b (cam h w) d', b=b, cam=num_cams, h=h, w=w)
        x_tok_emb_ = x_tok_emb_ + self.x_pos_emb[:, :x_tok_emb_.shape[1]]
        cond_tok_emb_ = cond_tok_emb_ + self.cond_pos_emb
        ignored_output = torch.cat((cond_tok_emb_, ), dim=1)

        x_tok_emb_ = self.cfg.forward_permuter(x_tok_emb_)
        input_data = torch.cat((ignored_output, x_tok_emb_), dim=1)

        pad_len = 0
        if input_data.shape[1] < self.cfg.gpt_block_size:
            pad_len = self.cfg.gpt_block_size - input_data.shape[1]
            pad_input_ids = torch.full((input_data.shape[0], pad_len), self.cfg.vocab_size, dtype=torch.long).to(input_data.device)
            pad_inputs_embeds = self.x_tok_emb(pad_input_ids)
            input_data = torch.cat([input_data, pad_inputs_embeds], dim=1)

        camera_bias = None
        if self.cfg.camera_bias:
            idx = torch.tril_indices(self.cfg.gpt_block_size, self.cfg.gpt_block_size)
            camera_bias_emb = torch.zeros((1, self.cfg.gpt_block_size, self.cfg.gpt_block_size), dtype=input_data.dtype, device=self.camera_bias_emb.device)
            camera_bias_emb[:, idx[0, :], idx[1, :]] = self.camera_bias_emb
            camera_bias = camera_bias_emb + self.cfg.prob_matrix.to(device=self.camera_bias_emb.device, dtype=input_data.dtype)

        assert input_data.shape[1] == self.cfg.gpt_block_size, "Cannot forward, model block size is exhausted."
        assert (~input_data.isfinite()).sum() == 0
        x = self.drop(input_data)
        x = self.blocks((x, camera_bias))[0]
        x = self.ln_f(x)
        logits = self.head(x)[:, :-pad_len or None]
        assert (~logits.isfinite()).sum() == 0

        ret = logits[:, ignored_output.shape[1] - 1:-1]
        return self.cfg.backward_permuter(ret)