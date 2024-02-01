from typing import Optional
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat
from multi_view_generation.bev_utils import util
from multi_view_generation.bev_utils import batched_camera_bev_grid
from multi_view_generation.modules.transformer.mingpt_sparse import GPTConfig
from multi_view_generation.modules.transformer.permuter import Identity, CustomPermuter
from multi_view_generation.modules.transformer.mask_generator import get_seq_pixel_mappings
from multi_view_generation import utils
from image_utils import Im
import wandb
import numpy as np
import time
import traceback

log = utils.get_pylogger(__name__)

def linear_step(warmup_steps, max_value, current_step):
    return min(max_value, current_step / warmup_steps * max_value + 1)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Net2NetTransformer(pl.LightningModule):
    def __init__(self,
                 maskgit,
                 first_stage,
                 cond_stage,
                 cfg,
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
                 bbox_warmup_steps: int = -1,
                 top_k: Optional[int] = None,
                 warmup_steps: int = 500,
                 lr_decay: bool = False,
                 sample_iterations: int = 18,
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
        self.bbox_warmup_steps = bbox_warmup_steps
        self.top_k = top_k
        self.lr_decay = lr_decay
        self.warmup_steps = warmup_steps
        self.sample_iterations = sample_iterations
        self.init_first_stage_from_ckpt(first_stage)
        self.init_cond_stage_from_ckpt(cond_stage)
        self.cfg: GPTConfig = cfg
        self.maskgit = maskgit
        if permuter is None:
            self.permuter = Identity()
        else:
            self.permuter = permuter

        if ckpt_path is not None:
            utils.init_from_ckpt(self, ckpt_path, ignore_keys=ignore_keys, unfrozen_keys=unfrozen_keys)

        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep
        self.save_hyperparameters(ignore=['transformer', 'first_stage', 'cond_stage'])

        for param in self.first_stage_model.parameters():
            param.requires_grad = False
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

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

    @torch.no_grad()
    def inference_step(self, batch):
        if not hasattr(self, 'z_indices'):
            x, c = self.get_xc(batch)
            _, self.c_indices = self.encode_to_c(c, batch)
            self.z_indices = torch.ones((c.shape[0], self.cfg.num_cams, self.cfg.num_cam_tokens), dtype=torch.int64, device=x.device) * (self.cfg.vocab_size)

        logits = self.transformer(self.z_indices, self.c_indices, batch, sampling=False)
        return logits

    @torch.no_grad()
    def sample(self, cond, batch, partial_decoding_idx=None):
        # We deliberately set the indices to be out of bounds to ensure we predict each index and so we know where to pad.
        
        init_ids = None
        if partial_decoding_idx is not None:
            init_ids = torch.full((cond.shape[0], self.cfg.num_cams, self.cfg.num_cam_tokens), self.maskgit.mask_id, dtype = torch.long, device = cond.device)
            with torch.autocast(device_type='cuda', dtype=torch.float):
                _, z_indices = self.encode_to_z(self.get_input(self.first_stage_key, batch), batch)
                z_indices = self.expand_all_images(z_indices)
            init_ids[:, partial_decoding_idx, :] = z_indices[:, partial_decoding_idx]

            init_ids = rearrange(init_ids, 'b cam tokens -> (b cam) tokens')
        ids = self.maskgit.generate(init_ids = init_ids, cond_images=cond, fmap_size=(self.cfg.cam_latent_h, self.cfg.cam_latent_w), batch=batch, timesteps=self.sample_iterations)

        if self.debug_viz:
            for cam_idx in range(len(self.cfg.cam_names)):
                self.logger.experiment.log({f"LATENT_CODE_{self.cfg.cam_names[cam_idx]}": wandb.Histogram(ids[0, cam_idx].detach().cpu().numpy(), num_bins=512), "global_step": self.trainer.global_step})

        assert ids.max() < self.cfg.vocab_size
        return ids

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

        _, z_indices = self.encode_to_z(x, batch)
        with torch.autocast(device_type='cuda', dtype=torch.float):
            _, c_indices = self.encode_to_c(c, batch)

        weights = self.get_token_weights(batch, inference, x) if self.bbox_ce_weight > 0 else None
        loss = self.maskgit.forward(z_indices, cond_images=c_indices, batch=batch, weights=weights)
        if isinstance(loss, tuple):
            return loss
        else:
            return loss, loss, 0

    def training_step(self, batch, batch_idx):
        loss, ce_loss, bce_critic_loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/ce_loss", ce_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/critic_loss", bce_critic_loss, prog_bar=True, on_step=True, on_epoch=True)
        
        if self.reset_random_mask > 0 and self.trainer.global_step % self.reset_random_mask == 0:
            import time
            start_time = time.time()
            for b in self.transformer.blocks:
                b.attention.randomize_layout()

            log.info(f'Resetting took: {time.time() - start_time}')

        return loss

    def validation_step(self, batch, batch_idx):
        log.info(f'Starting Val Step on Rank: {self.global_rank}, Batch Idx: {batch_idx}')
        loss, ce_loss, bce_critic_loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/ce_loss", ce_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/critic_loss", bce_critic_loss, prog_bar=True, on_step=False, on_epoch=True)
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
        images = self.log_images(batch, generate_only=True)
        return images

    def forward(self, batch):
        log.info(f'Starting test Step on Rank: {self.global_rank}')
        images = self.log_images(batch, generate_only=True)
        return images

    def configure_optimizers(self):        
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.learning_rate, betas = (0.9, 0.99), eps = 1e-8)
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
                lr_lambda=[update_lr],
            ),
            "name": "learning_rate",
            "interval": "step",  # The unit of the scheduler's step size
            "frequency": 1,  # The frequency of the scheduler
        }

        return [optimizer], [lr_scheduler]

    @torch.no_grad()
    def log_images(self, batch, generate_only=False, **kwargs):
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
                partial_decoding_idx = torch.randint(self.cfg.num_cams, (torch.randint(1, self.cfg.num_cams, ()).item(), ))
            elif self.partial_decoding == 3:
                if torch.rand(()).item() > 0.5:
                    partial_decoding_idx = torch.tensor([0])
                else:
                    partial_decoding_idx = torch.tensor([0, 2])
            else:
                partial_decoding_idx = torch.randint(self.cfg.num_cams, (1, ))

        # sample
        index_sample = self.sample(c_indices, batch, partial_decoding_idx=partial_decoding_idx)

        with torch.autocast(device_type='cuda', dtype=torch.float):
            gen_images = self.decode_to_img(self.combine_all_images(index_sample), original_quant_z_shape)
            gen_images = util.denormalize_tensor(gen_images, keep_tensor=True)
            gen_images = self.expand_all_images(gen_images)

            source_images = util.denormalize_tensor(self.combine_all_images(x), keep_tensor=True)
            source_images = self.expand_all_images(source_images)

            if self.partial_decoding:
                allowed_imgs = rearrange(gen_images[:, partial_decoding_idx, ...], 'b n c h w -> (b n) c h w')
                decoded_imgs = Im(allowed_imgs).add_border(border=3, color=(0, 249, 0)).torch[:, 0]
                decoded_imgs = rearrange(decoded_imgs, '(b n) c h w -> b n c h w', b=batch_size)
                gen_images[:, partial_decoding_idx, ...] = decoded_imgs.to(dtype=gen_images.dtype, device=gen_images.device)
            else:
                source_images = source_images.to(c_indices.device)

        ret = {'gen': gen_images, 'rec': rec_images, 'gt': source_images}  # b num_cams c h w

        log.info(f'Generating images took {round(time.time() - start_time, 3)} on Rank {self.global_rank}')
        return ret

    def get_token_weights(self, batch, inference, x):
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

        weight = torch.full((batch_size * self.cfg.num_img_tokens,), 1, dtype=torch.float).to(batch['image'].device)

        # Select all points that meet the condition.
        output = torch.cat([mask[:, [0, 1]], old_points[mask[:, 0], mask[:, 1], mask[:, 2], -1, None]], dim=-1)  # batch_idx, cam_idx, cam_latent_idx [0, cam_latent_h * cam_latent_w)

        seq_idx = output[:, 0] * self.cfg.num_img_tokens + (output[:, 2] + output[:, 1] * self.cfg.num_cam_tokens)
        weight[seq_idx] = linear_step(self.bbox_warmup_steps, self.bbox_ce_weight, self.trainer.global_step)

        viz_weight = torch.full((batch_size * self.cfg.num_img_tokens,), 0, dtype=torch.float).to(batch['image'].device)
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

        return weight
