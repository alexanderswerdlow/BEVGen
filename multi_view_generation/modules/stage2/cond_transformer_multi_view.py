from typing import Optional
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
from einops import rearrange, repeat
from multi_view_generation.bev_utils import util, return_binary_as_image
from multi_view_generation.bev_utils import camera_bev_grid, batched_camera_bev_grid
from multi_view_generation.modules.transformer.mingpt_sparse import GPTConfig
from multi_view_generation.modules.transformer.permuter import Identity, CustomPermuter
from multi_view_generation.modules.transformer.mask_generator import get_seq_pixel_mappings
from image_utils import Im
import wandb
import numpy as np
import torchvision
import time
import os
from tqdm import tqdm
from multi_view_generation import utils
import traceback
log = utils.get_pylogger(__name__)


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Net2NetTransformer(pl.LightningModule):
    def __init__(self,
                 transformer,
                 first_stage,
                 cond_stage,
                 permuter=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 unfrozen_keys=[],
                 first_stage_key="image",
                 cond_stage_key="segmentation",
                 downsample_cond_size=-1,
                 pkeep=1.0,
                 sos_token=0,
                 unconditional=False,
                 skip_sampling: bool = False,
                 bbox_ce_weight: float = 0.0,
                 reset_random_mask: int = 0,
                 debug_viz: bool = False,
                 partial_decoding: Optional[int] = None,
                 bbox_weight_epoch: int = -1,
                 top_k: Optional[int] = None,
                 warmup_steps: int = 500,
                 lr_decay: bool = False,
                 **kwargs
                 ):
        super().__init__()

        # TODO: Find a way to autoset vars while making the linters happy
        for k, v in kwargs.items():
            if k != 'self':
                setattr(self, k, v)

        self.be_unconditional = unconditional
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.skip_sampling = skip_sampling
        self.bbox_ce_weight = bbox_ce_weight
        self.reset_random_mask = reset_random_mask
        self.debug_viz = debug_viz
        self.partial_decoding = partial_decoding
        self.bbox_weight_epoch = bbox_weight_epoch
        self.top_k = top_k
        self.lr_decay = lr_decay
        self.warmup_steps = warmup_steps
        self.init_first_stage_from_ckpt(first_stage)
        self.init_cond_stage_from_ckpt(cond_stage)
        self.transformer = transformer
        self.cfg: GPTConfig = self.transformer.cfg
        if permuter is None:
            self.permuter = Identity()
        else:
            self.permuter = permuter

        if ckpt_path is not None:
            utils.init_from_ckpt(self, ckpt_path, ignore_keys=ignore_keys, unfrozen_keys=unfrozen_keys)

        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep
        self.save_hyperparameters(ignore=['transformer', 'first_stage', 'cond_stage'])

    def init_first_stage_from_ckpt(self, model):
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def init_cond_stage_from_ckpt(self, model):
        model = model.eval()
        model.train = disabled_train
        self.cond_stage_model = model

    def expand_all_images(self, arr):
        """Combines batch and camera dimensions to be compatible with various functions that only allow for a batch dim"""
        return rearrange(arr, '(b cam) ... -> b cam ...', cam=self.cfg.num_cams)

    def combine_all_images(self, arr):
        return rearrange(arr, 'b cam ... -> (b cam) ...')

    def forward(self, x, c, batch):
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x, batch)
        with torch.autocast(device_type='cuda', dtype=torch.float):
            _, c_indices = self.encode_to_c(c, batch)

        z_indices = self.expand_all_images(z_indices)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = rearrange(z_indices, 'b cam img_token -> b (cam img_token)', cam=self.cfg.num_cams, img_token=self.cfg.num_cam_tokens).clone()

        # Train:
        # initial: c_0, c_1, z_0, z_1
        #  input: c_0, c_1, z_0, pad
        #     gt: ign, z_0, z_1, ign

        # Sample:
        #  input: c_0, c_1, pad, pad
        #     gt: ign, z_0, ign, ign

        #  input: c_0, c_1, z_0, pad
        #     gt: ign, z_0, z_1, ign

        # make the prediction
        logits = self.transformer(z_indices, c_indices, batch, sampling=False)

        return logits.contiguous(), target.contiguous()

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def inference_step(self, batch):
        if not hasattr(self, 'z_indices'):
            x, c = self.get_xc(batch)
            _, self.c_indices = self.encode_to_c(c, batch)
            self.z_indices = torch.ones((c.shape[0], self.cfg.num_cams, self.cfg.num_cam_tokens), dtype=torch.int64, device=x.device) * (self.cfg.vocab_size)

        logits = self.transformer(self.z_indices, self.c_indices, batch, sampling=False)
        return logits

    @torch.no_grad()
    def sample(self, x, c, batch, temperature=1.0, sample=False, top_k=None, callback=lambda k: None, partial_decoding_idx=None):
        # We deliberately set the indices to be out of bounds to ensure we predict each index and so we know where to pad.
        x = torch.ones((x.shape[0], self.cfg.num_cams, self.cfg.num_cam_tokens), dtype=torch.int64, device=x.device) * (self.cfg.vocab_size)
        cond = c
        if self.skip_sampling:
            return x * 0
        elif partial_decoding_idx is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float):
                _, z_indices = self.encode_to_z(self.get_input(self.first_stage_key, batch), batch)
                z_indices = self.expand_all_images(z_indices)
            x[:, partial_decoding_idx, :] = z_indices[:, partial_decoding_idx]

        assert not self.transformer.training

        permuter = CustomPermuter(self.cfg)
        seq_to_pixel = get_seq_pixel_mappings(self.cfg)[1]

        for idx in tqdm(range(permuter.forward_shuffle_idx.shape[0])):
            # The indexing is confusing here.
            # We wish to decode in the permuter order which matches the transformer order.
            # However, since we backward shuffle in mingpt, returning the indices in (cam, h, w) order
            # we need to shuffle forward here to match those indices.
            j = permuter.forward_shuffle_idx[idx].item()
            i = seq_to_pixel[j, 0].item()
            k = j % self.cfg.num_cam_tokens

            if partial_decoding_idx is not None and i in partial_decoding_idx:
                continue

            if self.debug_viz and idx % 50 == 0:
                cam_imgs = []
                for cam_idx in range(len(self.cfg.cam_names)):
                    cam_imgs.append(repeat(((x[0, cam_idx].reshape(self.cfg.cam_latent_h, self.cfg.cam_latent_w) / (self.cfg.vocab_size - 1)) * 255).to(torch.uint8), '... -> c ...', c=3))

                self.logger.experiment.log({f"LATENT_CODE_IMAGES": wandb.Image(rearrange(torchvision.utils.make_grid(
                    torch.stack(cam_imgs, dim=0), nrow=3).detach().cpu().numpy(), 'c h w -> h w c')), "global_step": self.trainer.global_step})

            callback(i)
            assert x.size(1) <= self.cfg.gpt_block_size  # make sure model can see conditioning

            if self.skip_sampling:
                logits = torch.randn(x.shape[0], self.cfg.num_cams * self.cfg.num_cam_tokens, self.cfg.vocab_size, requires_grad=True).to(torch.float16).to(x.device)
            else:
                logits = self.transformer(x, cond, batch, sampling=True)

            logits = rearrange(logits, 'b (cam img_token) d -> b cam img_token d', cam=self.cfg.num_cams, img_token=self.cfg.num_cam_tokens)

            assert (~logits.isfinite()).sum() == 0
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, i, k, :] / temperature

            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            x[:, i, k] = ix.squeeze()

        if self.debug_viz:
            cam_imgs = []
            for cam_idx in range(len(self.cfg.cam_names)):
                self.logger.experiment.log({f"LATENT_CODE_{self.cfg.cam_names[cam_idx]}": wandb.Histogram(x[0, cam_idx].detach().cpu().numpy(), num_bins=512), "global_step": self.trainer.global_step})

        assert x.max() < self.cfg.vocab_size
        return x

    @torch.no_grad()
    def encode_to_z(self, x, batch):
        quant_z, _, info = self.first_stage_model.encode(x, batch)
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c, batch):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, [_, _, indices] = self.cond_stage_model.encode(c, batch)
        indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    def get_input(self, key, batch):
        x = batch[key]

        if x.dtype == torch.double or x.dtype == torch.uint8:
            x = x.float()

        x = rearrange(x, '... h w c -> ... c h w')

        if key == 'image':
            if len(x.shape) == 4:
                x = x[None, ...]
            x = self.combine_all_images(x)

        return x.to(memory_format=torch.contiguous_format)

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]

        return x, c

    def shared_step(self, batch, batch_idx, inference=False):
        x, c = self.get_xc(batch)
        logits, target = self(x, c, batch)

        if self.bbox_ce_weight > 0:
            bboxes = batch['bbx'].clone()
            bboxes = bboxes.to(torch.float)
            bboxes[:, :, :, [0, 2]] *= self.cfg.cam_latent_w
            bboxes[:, :, :, [1, 3]] *= self.cfg.cam_latent_h

            bboxes[:, :, :, [0, 1]] = torch.floor(bboxes[:, :, :, [0, 1]])
            bboxes[:, :, :, [2, 3]] = torch.ceil(bboxes[:, :, :, [2, 3]])

            bboxes[:, :, :, [0, 2]] = torch.clamp(bboxes[:, :, :, [0, 2]], max=self.cfg.cam_latent_w)
            bboxes[:, :, :, [1, 3]] = torch.clamp(bboxes[:, :, :, [1, 3]], max=self.cfg.cam_latent_h)
            bboxes = bboxes.to(torch.long)

            # Bounding boxes and points.
            batch_size = bboxes.shape[0]
            num_bboxes = bboxes.shape[-2]
            grid_y, grid_x = torch.meshgrid(torch.arange(0, self.cfg.cam_latent_h), torch.arange(0, self.cfg.cam_latent_w))
            points = torch.cat([rearrange(torch.stack((grid_x, grid_y), 2), 'h w ... -> (h w) ...'), torch.arange(self.cfg.num_cam_tokens)[..., None]], dim=-1).to(bboxes.device)

            points = repeat(points, '... -> batch num_cams ...', num_cams=self.cfg.num_cams, batch=batch_size)
            old_points = points
            points = repeat(points, 'batch num_cams cam_tokens ... -> batch num_cams cam_tokens num_bboxes ...',
                            num_bboxes=num_bboxes, cam_tokens=self.cfg.num_cam_tokens, num_cams=self.cfg.num_cams, batch=batch_size)

            # Create the conditions necessary to determine if a point is within a bounding box.
            # x >= left, x <= right, y >= top, y <= bottom
            bboxes = rearrange(bboxes, 'batch num_cams num_bboxes ... -> (batch num_cams num_bboxes) ...', num_bboxes=num_bboxes, num_cams=self.cfg.num_cams, batch=batch_size)
            points = rearrange(points, 'batch num_cams cam_tokens num_bboxes ... -> cam_tokens (batch num_cams num_bboxes) ...',
                               num_bboxes=num_bboxes, cam_tokens=self.cfg.num_cam_tokens, num_cams=self.cfg.num_cams, batch=batch_size)
            c1 = points[:, :, 0] <= bboxes[:, 2]
            c2 = points[:, :, 0] >= bboxes[:, 0]
            c3 = points[:, :, 1] <= bboxes[:, 3]
            c4 = points[:, :, 1] >= bboxes[:, 1]

            # Add all of the conditions together. If all conditions are met, sum is 4.
            # Afterwards, get all point indices that meet the condition (a.k.a. all non-zero mask-summed values)
            mask = c1.to(torch.long) + c2.to(torch.long) + c3.to(torch.long) + c4.to(torch.long)
            mask = torch.nonzero(rearrange((mask == 4), 'cam_tokens (batch num_cams num_bboxes) -> batch num_cams cam_tokens num_bboxes',
                                 num_bboxes=num_bboxes, cam_tokens=self.cfg.num_cam_tokens, num_cams=self.cfg.num_cams, batch=batch_size).sum(dim=-1))

            weight = torch.full((batch_size * self.cfg.num_img_tokens,), 1, dtype=torch.float).to(x.device)

            # Select all points that meet the condition.
            output = torch.cat([mask[:, [0, 1]], old_points[mask[:, 0], mask[:, 1], mask[:, 2], -1, None]], dim=-1)  # batch_idx, cam_idx, cam_latent_idx [0, cam_latent_h * cam_latent_w)

            seq_idx = output[:, 0] * self.cfg.num_img_tokens + (output[:, 2] + output[:, 1] * self.cfg.num_cam_tokens)
            weight[seq_idx] = self.bbox_ce_weight if self.current_epoch > self.bbox_weight_epoch else 1

            viz_weight = torch.full((batch_size * self.cfg.num_img_tokens,), 0, dtype=torch.float).to(x.device)
            viz_weight[seq_idx] = 1

            if not inference and self.debug_viz and self.trainer.global_step % 500 == 0:
                x_all = self.expand_all_images(x)
                for i in range(self.cfg.num_cams):
                    self.logger.experiment.log({f"BBOX_WEIGHT_{self.cfg.cam_names[i]}": wandb.Image(repeat(((viz_weight[i * self.cfg.num_cam_tokens: (i + 1) * self.cfg.num_cam_tokens].reshape(
                        *self.cfg.cam_latent_res).detach().cpu().numpy()) * 255).astype(np.uint8), '... -> ... c', c=3)), "global_step": self.trainer.global_step})
                    box_data = []
                    for j in range(batch['bbx'].shape[2]):
                        if batch['bbx_mask'][0, i, j].item():
                            minx, maxx, miny, maxy = batch['bbx'][0, i, j][0].item(), batch['bbx'][0, i, j][2].item(), batch['bbx'][0, i, j][1].item(), batch['bbx'][0, i, j][3].item()
                            box_data.append({"position": {"minX": minx, "maxX": maxx, "minY": miny, "maxY": maxy}, "class_id": 0, })
                    self.logger.experiment.log({f"BBOX_{self.cfg.cam_names[i]}": wandb.Image(x_all[0, i], boxes={"predictions": {"box_data": box_data}}), "global_step": self.trainer.global_step})

            logits = rearrange(logits, 'batch img_tokens d -> (batch img_tokens) d')
            target = rearrange(target, 'batch img_tokens -> (batch img_tokens)')

            return (F.cross_entropy(logits, target, reduction='none') * weight).mean()
        else:
            return sum([F.cross_entropy(logit.view(-1, logit.shape[-1]), codes.view(-1)) for logit, codes in zip(logits, target)]) / len(logits)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        if self.reset_random_mask > 0 and self.trainer.global_step % self.reset_random_mask == 0:
            import time
            start_time = time.time()
            for b in self.transformer.blocks:
                b.attention.randomize_layout()

            log.info(f'Resetting took: {time.time() - start_time}')

        return loss

    def validation_step(self, batch, batch_idx):
        log.info(f'Starting Val Step on Rank: {self.global_rank}, Batch Idx: {batch_idx}')
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        ret = {'loss': loss}
        if batch_idx == 0:
            try: # Viz code has a lot of side effects so we catch and continue training
                images = self.log_images(batch)
                ret = {**images, **ret}
            except Exception as ex:
                log.error("log_images failed", exc_info=ex)
                log.error("".join(traceback.TracebackException.from_exception(ex).format()))
        return ret

    def test_step(self, batch, batch_idx):
        log.info(f'Starting test Step on Rank: {self.global_rank}')
        # torch.save(batch, f'data/batch/{batch_idx}.pt')
        loss = self.shared_step(batch, batch_idx)
        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        images = self.log_images(batch, generate_only=True, top_k=self.top_k)
        return images

    def on_test_start(self) -> None:
        self.logger.experiment.log({
            "layout": wandb.Image(
                np.concatenate([np.concatenate([return_binary_as_image(self.transformer.blocks[j].attention.sparse_self_attention.master_layout[i]) for i in range(self.transformer.blocks[j].attention.sparse_self_attention.master_layout.shape[0])], axis=1) for j in range(len(self.transformer.blocks))], axis=0)),
            "attention_mask": wandb.Image(return_binary_as_image(self.transformer.cfg.attention_mask)),
            "global_step": self.trainer.global_step
        })

    def on_train_start(self):
        self.logger.experiment.log({
            "layout": wandb.Image(
                np.concatenate([np.concatenate([return_binary_as_image(self.transformer.blocks[j].attention.sparse_self_attention.master_layout[i]) for i in range(self.transformer.blocks[j].attention.sparse_self_attention.master_layout.shape[0])], axis=1) for j in range(len(self.transformer.blocks))], axis=0)),
            "attention_mask": wandb.Image(return_binary_as_image(self.transformer.cfg.attention_mask)),
            "global_step": self.trainer.global_step
        })

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        for param_name in ['x_pos_emb', 'cond_pos_emb', 'camera_bias_emb', 'bev_cam_pos_emb', 'camera_bias_emb']:
            if hasattr(self.transformer, param_name):
                no_decay.add(param_name)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(param_dict.keys() - union_params))], "weight_decay": 0.0},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        if not self.lr_decay:
            return optimizer

        def update_lr(*_):
            config = self.hparams

            if self.trainer.global_step < config.warmup_steps:
                # linear warmup
                lr_mult = float(self.trainer.global_step) / float(max(1, config.warmup_steps))
                lr_mult = max(lr_mult, 1e-2)  # could be that we've not seen any yet
            else:
                # cosine learning rate decay
                progress = float(self.trainer.global_step - config.warmup_steps) / float(
                    max(1, self.trainer.estimated_stepping_batches - config.warmup_steps)
                )
                lr_mult = max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))

            return lr_mult

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=[update_lr, update_lr, update_lr],
            ),
            "name": "learning_rate",
            "interval": "step",  # The unit of the scheduler's step size
            "frequency": 1,  # The frequency of the scheduler
        }

        return [optimizer], [lr_scheduler]

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, generate_only=False, **kwargs):
        log.info(f'Logging images on rank: {self.global_rank}')
        start_time = time.time()
        x, c = self.get_xc(batch)
        x = x.to(device=self.device)
        c = c.to(device=self.device)
        batch_size = batch['image'].shape[0]

        with torch.autocast(device_type='cuda', dtype=torch.float):
            quant_z, z_indices = self.encode_to_z(x, batch)
            quant_c, c_indices = self.encode_to_c(c, batch)

            original_quant_z_shape = quant_z.shape
            z_indices = self.expand_all_images(z_indices)  # b v d
            quant_z = self.expand_all_images(quant_z)
            x = self.expand_all_images(x)

            input_reconstruction = self.decode_to_img(self.combine_all_images(z_indices), original_quant_z_shape)
            input_reconstruction = util.denormalize_tensor(input_reconstruction, keep_tensor=True)
            rec_images = self.expand_all_images(input_reconstruction)

        partial_decoding_idx = None
        if self.partial_decoding:
            if self.partial_decoding == 2:
                partial_decoding_idx = torch.randint(self.transformer.cfg.num_cams, (torch.randint(1, 3, ()).item(), ))
            elif self.partial_decoding == 3:
                if torch.rand(()).item() > 0.5:
                    partial_decoding_idx = torch.tensor([0])
                else:
                    partial_decoding_idx = torch.tensor([0, 2])
            elif self.partial_decoding == 4:
                partial_decoding_idx = torch.tensor([3, 0, 2])
            else:
                partial_decoding_idx = torch.randint(self.transformer.cfg.num_cams, (1, ))

        # sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices, batch,
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None,
                                   partial_decoding_idx=partial_decoding_idx
                                   )

        gen_images = self.decode_to_img(self.combine_all_images(index_sample), original_quant_z_shape)
        gen_images = util.denormalize_tensor(gen_images, keep_tensor=True)
        source_images = util.denormalize_tensor(self.combine_all_images(x), keep_tensor=True)
        gen_images = self.expand_all_images(gen_images)
        if self.partial_decoding:
            allowed_imgs = rearrange(gen_images[:, partial_decoding_idx, ...], 'b n c h w -> (b n) c h w')
            decoded_imgs = Im(allowed_imgs).add_border(border=3, color=(0, 249, 0)).torch
            decoded_imgs = rearrange(decoded_imgs, '(b n) c h w -> b n c h w', b=batch_size)
            gen_images[:, partial_decoding_idx, ...] = decoded_imgs.to(dtype=gen_images.dtype, device=gen_images.device)
            source_images = Im(source_images).write_text([f'{partial_decoding_idx}' for _ in range(batch_size)
                                                               for _ in range(self.cfg.num_cams)]).torch.to(c_indices.device)
        else:
            source_images = Im(source_images).write_text([f'GT {cam}' for _ in range(index_sample.shape[0]) for cam in self.cfg.cam_names]).torch.to(c_indices.device)

        source_images = self.expand_all_images(source_images)
        
        ret = {'gen': gen_images, 'rec': rec_images, 'gt': source_images}  # b num_cams c h w

        if generate_only:
            return ret

        # for i in range(len(c)):
        #     cond_all_angles = cv2.cvtColor(util.Im(c[i]).np, cv2.COLOR_RGB2BGR)
        #     start_coord = (np.asarray(cond_all_angles.shape[:2]) - 1) // 2
        #     for angle in batch["angle"][i].detach().cpu().numpy():
        #         x, y = (np.sin(angle) + 1) / 2, (-np.cos(angle) + 1) / 2
        #         end_coord = np.rint(np.asarray((y * (cond_all_angles.shape[0] - 1), x * (cond_all_angles.shape[1] - 1)))).astype(np.int)
        #         cond_all_angles = cv2.arrowedLine(cond_all_angles, start_coord[::-1], end_coord[::-1], color=(174, 0, 10), thickness=2, tipLength=0.1)
        #     cond_all_angles = rearrange(torch.as_tensor((cv2.cvtColor(cond_all_angles, cv2.COLOR_BGR2RGB) / 255).astype(np.float32)), 'h w c -> c h w')
        #     c[i] = cond_all_angles

        
        cond_input_sample = batched_camera_bev_grid(self.cfg, gen_images, c)
        full_image = torch.cat([batched_camera_bev_grid(self.cfg, rec_images, batch['segmentation']), batched_camera_bev_grid(self.cfg, source_images, c)], dim=-2)

        log.info(f'Generating images took {round(time.time() - start_time, 3)} on Rank {self.global_rank}')
        return {f'cond_input_sample': cond_input_sample, f'input_reconstruction': full_image, **ret}
