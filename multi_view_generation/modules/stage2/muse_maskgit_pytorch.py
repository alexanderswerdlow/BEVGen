import math
from random import random
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum

import torchvision.transforms as T

from typing import Callable, Iterable, Optional, List

from einops import rearrange, repeat, reduce
from beartype import beartype

from muse_maskgit_pytorch.vqgan_vae import VQGanVAE
from muse_maskgit_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

from tqdm.auto import tqdm

from multi_view_generation.modules.transformer.mingpt_sparse import GPTConfig

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def l2norm(t):
    return F.normalize(t, dim = -1)

# tensor helpers

def get_mask_subset_prob(mask, prob, min_mask = 0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)
    logits = torch.rand((batch, seq), device = device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim = -1).float()

    num_padding = (~mask).sum(dim = -1, keepdim = True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

# classes

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class GEGLU(nn.Module):
    """ https://arxiv.org/abs/2002.05202 """

    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return gate * F.gelu(x)

def FeedForward(dim, mult = 4):
    """ https://arxiv.org/abs/2110.09456 """

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Linear(inner_dim, dim, bias = False)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        cross_attend = False,
        scale = 8,
        cfg: Optional[GPTConfig] = None,
    ):
        super().__init__()
        self.scale = scale
        self.heads =  heads
        inner_dim = dim_head * heads

        self.cross_attend = cross_attend
        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        self.cfg = cfg

    def forward(
        self,
        x,
        context = None,
        context_mask = None,
        attn_bias = None,
    ):
        assert not (exists(context) ^ self.cross_attend)

        h, is_cross_attn = self.heads, exists(context)

        x = self.norm(x)

        kv_input = context if self.cross_attend else x

        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        q = q * self.scale

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        nk, nv = self.null_kv
        nk, nv = map(lambda t: repeat(t, 'h 1 d -> b h 1 d', b = x.shape[0]), (nk, nv))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(attn_bias):
            if is_cross_attn:
                attn_bias = attn_bias[self.cfg.num_cond_tokens:, :self.cfg.num_cond_tokens]
            else:
                attn_bias = attn_bias[self.cfg.num_cond_tokens:, self.cfg.num_cond_tokens:]
            attn_bias = F.pad(attn_bias, (1, 0), value = 0.)
            sim = sim + attn_bias
        
        if exists(context_mask):
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            context_mask = F.pad(context_mask, (1, 0), value = True)

            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~context_mask, mask_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlocks(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        cfg: Optional[GPTConfig] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, cfg = cfg),
                Attention(dim = dim, dim_head = dim_head, heads = heads, cross_attend = True, cfg = cfg),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = LayerNorm(dim)

    def forward(self, x, context = None, context_mask = None, attn_bias=None):
        for attn, cross_attn, ff in self.layers:
            x = attn(x, attn_bias=attn_bias) + x

            x = cross_attn(x, context = context, context_mask = context_mask, attn_bias=attn_bias) + x

            x = ff(x) + x

        return self.norm(x)

class TransformerMultiView(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        seq_len,
        dim_out = None,
        self_cond = False,
        add_mask_id = False,
        cfg: Optional[GPTConfig] = None,
        **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        seq_len = seq_len[0] * seq_len[1] if isinstance(seq_len, Iterable) else seq_len
        self.dim = dim

        self.mask_id = num_tokens if add_mask_id else None

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens + int(add_mask_id), dim)
        self.pos_emb = nn.Embedding(self.cfg.num_img_tokens, dim)
        self.seq_len = seq_len

        self.cond_token_emb = nn.Embedding(cfg.cond_vocab_size, dim)
        self.cond_pos_emb = nn.Embedding(cfg.num_cond_tokens, dim)

        self.transformer_blocks = TransformerBlocks(dim = dim, cfg = cfg, **kwargs)
        self.norm = LayerNorm(dim)

        self.dim_out = default(dim_out, num_tokens)
        self.to_logits = nn.Linear(dim, self.dim_out, bias = False)

        # optional self conditioning

        self.self_cond = self_cond
        self.self_cond_to_init_embed = FeedForward(dim)

        from multi_view_generation.modules.transformer.mingpt_sparse import generate_grid, get_bev_grid
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

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3.,
        return_embed = False,
        **kwargs
    ):
        if cond_scale == 1:
            return self.forward(*args, return_embed = return_embed, cond_drop_prob = 0., **kwargs)

        logits, embed = self.forward(*args, return_embed = True, cond_drop_prob = 0., **kwargs)

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)

        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    def forward(
        self,
        x,
        return_embed = False,
        return_logits = False,
        labels = None,
        ignore_index = 0,
        self_cond_embed = None,
        cond_drop_prob = 0.,
        conditioning_token_ids: Optional[torch.Tensor] = None,
        batch=None,
        weights=None,
    ):
        device, b, n = x.device, *x.shape
        assert n <= self.seq_len


        I_inv = batch['intrinsics_inv']  # b cam 3 3
        E_inv = batch['extrinsics_inv']  # b cam 4 4

        b, num_cams = I_inv.shape[0], I_inv.shape[1]

        h, w = self.cfg.cam_latent_h, self.cfg.cam_latent_w

        x = rearrange(x, '(b cam) ... -> b cam ...', cam=self.cfg.num_cams)

        x = self.token_emb(x)
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
            x = x + img_embed

        x = rearrange(x, ' b cam (h w) d -> b (cam h w) d', b=b, cam=num_cams, h=h, w=w)
        x = x + self.pos_emb(torch.arange(x.shape[1], device = device))

        context = self.cond_token_emb(conditioning_token_ids)
        if self.cfg.bev_embed:
            c_expanded = repeat(rearrange(c_embed, '(b n) d 1 1 -> b n d', b=b, n=num_cams), 'b n d -> b n cond_tokens d', cond_tokens=self.cfg.num_cond_tokens)
            grid_embed = rearrange(self.bev_embed(self.bev_grid[None, :2]), '1 d h w -> 1 (h w) d')
            bev_cam_embed = reduce(self.bev_cam_pos_emb + c_expanded, 'b n cond_tokens d -> b cond_tokens d', 'sum')
            bev_embed = grid_embed - bev_cam_embed
            context = context + bev_embed
        context = context + self.cond_pos_emb(torch.arange(context.shape[1], device = device))
        context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)

        attn_bias = None
        if self.cfg.camera_bias:
            idx = torch.tril_indices(self.cfg.gpt_block_size, self.cfg.gpt_block_size)
            camera_bias_emb = torch.zeros((self.cfg.gpt_block_size, self.cfg.gpt_block_size), dtype=context.dtype, device=self.camera_bias_emb.device)
            camera_bias_emb[idx[0, :], idx[1, :]] = self.camera_bias_emb
            attn_bias = camera_bias_emb + self.cfg.prob_matrix.to(device=self.camera_bias_emb.device, dtype=context.dtype)

        # classifier free guidance

        if self.training and cond_drop_prob > 0.:
            mask = prob_mask_like((b, 1), 1. - cond_drop_prob, device)
            context_mask = context_mask & mask

        if self.self_cond:
            if not exists(self_cond_embed):
                self_cond_embed = torch.zeros_like(x)
            x = x + self.self_cond_to_init_embed(self_cond_embed)

        embed = self.transformer_blocks(x, context = context, context_mask = context_mask, attn_bias = attn_bias)

        logits = self.to_logits(embed)
        embed = rearrange(embed, 'b (cam h w) ... -> (b cam) (h w) ...', b=b, cam=num_cams, h=h, w=w)
        logits = rearrange(logits, 'b (cam h w) ... -> (b cam) (h w) ...', b=b, cam=num_cams, h=h, w=w)

        if return_embed:
            return logits, embed

        if not exists(labels):
            return logits

        if self.dim_out == 1:
            loss = F.binary_cross_entropy_with_logits(rearrange(logits, '... 1 -> ...'), labels)
        elif weights is not None:
            loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index = ignore_index, reduction='none')
            loss = (rearrange(loss, 'b c -> (b c)') * weights).sum() / (labels != ignore_index).sum()
        else:
            loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index = ignore_index)

        if not return_logits:
            return loss

        return loss, logits

# self critic wrapper

class SelfCritic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.to_pred = nn.Linear(net.dim, 1)

    def forward_with_cond_scale(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_cond_scale(x, *args, return_embed = True, **kwargs)
        return self.to_pred(embeds)

    def forward_with_neg_prompt(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_neg_prompt(x, *args, return_embed = True, **kwargs)
        return self.to_pred(embeds)

    def forward(self, x, *args, labels = None, **kwargs):
        _, embeds = self.net(x, *args, return_embed = True, **kwargs)
        logits = self.to_pred(embeds)

        if not exists(labels):
            return logits

        logits = rearrange(logits, '... 1 -> ...')

        if logits.shape[0] != labels.shape[0]:
            logits = rearrange(logits, 'b (cam h w) -> (b cam) (h w)', b=logits.shape[0], cam=self.net.cfg.num_cams, h=self.net.cfg.cam_latent_h, w=self.net.cfg.cam_latent_w)
            
        return F.binary_cross_entropy_with_logits(logits, labels)

# specialized transformers

class MaskGitTransformerMultiView(TransformerMultiView):
    def __init__(self, *args, **kwargs):
        assert 'add_mask_id' not in kwargs
        super().__init__(*args, add_mask_id = True, **kwargs)

class TokenCritic(TransformerMultiView):
    def __init__(self, *args, **kwargs):
        assert 'dim_out' not in kwargs
        super().__init__(*args, dim_out = 1, **kwargs)

# classifier free guidance functions

def uniform(shape, min = 0, max = 1, device = None):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device = None):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return uniform(shape, device = device) < prob

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs

# noise schedules

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

# main maskgit classes

class MaskGit(nn.Module):
    def __init__(
        self,
        image_size,
        transformer: MaskGitTransformerMultiView,
        noise_schedule: Callable = cosine_schedule,
        token_critic: Optional[TokenCritic] = None,
        self_token_critic = False,
        cond_image_size = None,
        cond_drop_prob = 0.5,
        self_cond_prob = 0.9,
        no_mask_token_prob = 0.,
        critic_loss_weight = 1.
    ):
        super().__init__()
        image_size = image_size[0] * image_size[1] if isinstance(image_size, Iterable) else image_size

        self.image_size = image_size
        self.cond_image_size = cond_image_size
        self.resize_image_for_cond_image = exists(cond_image_size)

        self.cond_drop_prob = cond_drop_prob

        self.transformer = transformer
        self.self_cond = transformer.self_cond

        self.mask_id = transformer.mask_id
        self.noise_schedule = noise_schedule

        assert not (self_token_critic and exists(token_critic))
        self.token_critic = token_critic

        if self_token_critic:
            self.token_critic = SelfCritic(transformer)

        self.critic_loss_weight = critic_loss_weight

        # self conditioning
        self.self_cond_prob = self_cond_prob

        # percentage of tokens to be [mask]ed to remain the same token, so that transformer produces better embeddings across all tokens as done in original BERT paper
        # may be needed for self conditioning
        self.no_mask_token_prob = no_mask_token_prob

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        init_ids: Optional[torch.Tensor] = None,
        cond_images: Optional[torch.Tensor] = None,
        fmap_size = None,
        temperature = 1.,
        topk_filter_thres = 0.9,
        can_remask_prev_masked = False,
        force_not_use_token_critic = False,
        timesteps = 12,  # ideal number of steps is 18 in maskgit paper
        cond_scale = 3,
        critic_noise_scale = 1,
        batch=None
    ):

        # begin with all image token ids masked

        device = next(self.parameters()).device

        seq_len = fmap_size[0] * fmap_size[1]

        batch_size = len(cond_images) * self.transformer.cfg.num_cams

        shape = (batch_size, seq_len)


        scores = torch.zeros(shape, dtype = torch.float32, device = device)
        init_mask = None
        ids = torch.full(shape, self.mask_id, dtype = torch.long, device = device)

        if init_ids is not None:
            init_mask = init_ids != self.mask_id

        starting_temperature = temperature

        cond_ids = None


        demask_fn = self.transformer.forward_with_cond_scale

        # whether to use token critic for scores

        use_token_critic = exists(self.token_critic) and not force_not_use_token_critic

        if use_token_critic:
            token_critic_fn = self.token_critic.forward_with_cond_scale

        cond_ids = cond_images

        self_cond_embed = None

        for timestep, steps_until_x0 in tqdm(zip(torch.linspace(0, 1, timesteps, device = device), reversed(range(timesteps))), total = timesteps):

            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = max(int((rand_mask_prob * seq_len).item()), 1)

            masked_indices = scores.topk(num_token_masked, dim = -1).indices

            ids = ids.scatter(1, masked_indices, self.mask_id)

            if init_ids is not None:
                ids[init_mask] = init_ids[init_mask]

            logits, embed = demask_fn(
                ids,
                self_cond_embed = self_cond_embed,
                conditioning_token_ids = cond_ids,
                cond_scale = cond_scale,
                return_embed = True,
                batch=batch
            )

            self_cond_embed = embed if self.self_cond else None

            filtered_logits = top_k(logits, topk_filter_thres)

            temperature = starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

            pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

            is_mask = ids == self.mask_id

            ids = torch.where(
                is_mask,
                pred_ids,
                ids
            )

            if use_token_critic:
                scores = token_critic_fn(
                    ids,
                    conditioning_token_ids = cond_ids,
                    cond_scale = cond_scale,
                    batch=batch
                )

                scores = rearrange(scores, '... 1 -> ...')

                scores = scores + (uniform(scores.shape, device = device) - 0.5) * critic_noise_scale * (steps_until_x0 / timesteps)
            else:
                probs_without_temperature = logits.softmax(dim = -1)

                scores = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
                scores = rearrange(scores, '... 1 -> ...')

                if not can_remask_prev_masked:
                    scores = scores.masked_fill(~is_mask, -1e5)
                else:
                    assert self.no_mask_token_prob > 0., 'without training with some of the non-masked tokens forced to predict, not sure if the logits will be meaningful for these token'

        # get ids

        ids = rearrange(ids, 'b (i j) -> b i j', i = fmap_size[0], j = fmap_size[1])

        return ids

    def forward(
        self,
        images_or_ids: torch.Tensor,
        ignore_index = -1,
        cond_images: Optional[torch.Tensor] = None,
        cond_token_ids: Optional[torch.Tensor] = None,
        cond_drop_prob = None,
        train_only_generator = False,
        sample_temperature = None,
        batch=None,
        weights=None
    ):

        ids = images_or_ids

        # take care of conditioning image if specified

        # get some basic variables

        ids = rearrange(ids, 'b ... -> b (...)')

        batch_size, seq_len, device, cond_drop_prob = *ids.shape, ids.device, default(cond_drop_prob, self.cond_drop_prob)

        # tokenize conditional images if needed

        assert not (exists(cond_images) and exists(cond_token_ids)), 'if conditioning on low resolution, cannot pass in both images and token ids'

        cond_token_ids = cond_images

        # prepare mask

        rand_time = uniform((batch_size,), device = device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min = 1)

        mask_id = self.mask_id
        batch_randperm = torch.rand((batch_size, seq_len), device = device).argsort(dim = -1)
        mask = batch_randperm < rearrange(num_token_masked, 'b -> b 1')

        mask_id = self.transformer.mask_id
        labels = torch.where(mask, ids, ignore_index)

        if self.no_mask_token_prob > 0.:
            no_mask_mask = get_mask_subset_prob(mask, self.no_mask_token_prob)
            mask &= ~no_mask_mask

        x = torch.where(mask, mask_id, ids)

        # self conditioning

        self_cond_embed = None

        if self.transformer.self_cond and random() < self.self_cond_prob:
            with torch.no_grad():
                _, self_cond_embed = self.transformer(
                    x,
                    conditioning_token_ids = cond_token_ids,
                    cond_drop_prob = 0.,
                    return_embed = True,
                    batch=batch,
                    weights=weights
                )

                self_cond_embed.detach_()

        # get loss

        ce_loss, logits = self.transformer(
            x,
            self_cond_embed = self_cond_embed,
            conditioning_token_ids = cond_token_ids,
            labels = labels,
            cond_drop_prob = cond_drop_prob,
            ignore_index = ignore_index,
            return_logits = True,
            batch=batch,
            weights=weights
        )

        if not exists(self.token_critic) or train_only_generator:
            return ce_loss

        # token critic loss

        sampled_ids = gumbel_sample(logits, temperature = default(sample_temperature, random()))

        if mask.shape[0] != sampled_ids.shape[0]:
            sampled_ids = rearrange(sampled_ids, 'b (cam h w) -> (b cam) (h w)', b=sampled_ids.shape[0], cam=self.transformer.cfg.num_cams, h=self.transformer.cfg.cam_latent_h, w=self.transformer.cfg.cam_latent_w)

        critic_input = torch.where(mask, sampled_ids, x)
        critic_labels = (ids != critic_input).float()

        bce_loss = self.token_critic(
            critic_input,
            conditioning_token_ids = cond_token_ids,
            labels = critic_labels,
            cond_drop_prob = cond_drop_prob,
            batch=batch
        )

        return ce_loss + self.critic_loss_weight * bce_loss, ce_loss, bce_loss