# Heavily modified from https://github.com/CompVis/taming-transformers, 
# and https://github.com/thuanz123/enhancing-transformers

import os
import wandb
import numpy as np
from PIL import Image
from typing import Tuple, Generic, Dict
from pathlib import Path
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from multi_view_generation import utils
import time
from multi_view_generation.bev_utils import batched_camera_bev_grid, util, viz_bev
from einops import rearrange, repeat
from multi_view_generation.modules.stage1.vqgan import VQModel
from datetime import datetime
import random
import string
import traceback
from image_utils import Im
from multi_view_generation.bev_utils import Dataset

log = utils.get_pylogger(__name__)

def save_img(img, save_path: Path):
    os.makedirs(save_path.parents[0], exist_ok=True)
    Im(img).pil.save(save_path)


class GenerateImages(Callback):
    def __init__(self, save_dir=None, figure_format=False, rand_str=False,**kwargs):
        self.save_dir = save_dir
        self.figure_format = figure_format
        self.rand_str = rand_str

    def save_images(self, image_dict, trainer, pl_module, local_save=True):
        for k in image_dict:
            image_dict[k] = torchvision.utils.make_grid(image_dict[k].float().detach().cpu(), nrow=2)

        try:
            grids = dict()
            for k in image_dict:
                grids[f"val/{k}_{pl_module.global_rank}"] = wandb.Image(image_dict[k])
        except Exception as e:
            pass

        try:
            trainer.logger.experiment.log(grids)
        except Exception as e:
            log.error(f'Failed to log on rank: {pl_module.global_rank}')

        if local_save:
            try:
                root = os.path.join(trainer.log_dir, "results", datetime.now().strftime("%Y_%m_%d-%H_%M"))
                os.makedirs(root, exist_ok=True)
                for k in image_dict:
                    grid = image_dict[k]
                    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    grid = grid.numpy()
                    grid = (grid*255).astype(np.uint8)
                    filename = "{}_gs-{:06}_e-{:06}-{:02}.png".format(k, pl_module.global_step, pl_module.current_epoch, pl_module.global_rank)
                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    Image.fromarray(grid).save(path)
            except Exception as e:
                log.error(e)
                log.error(f"Failed to save local image, Rank: {pl_module.global_rank}")

    def save_raw_data(self, trainer: pl.trainer.Trainer, pl_module: pl.LightningModule, outputs, batch, save_nuscenes_fmt: bool = True):
        gen_outputs = outputs['gen']
        local_save_dir = self.save_dir if self.save_dir is not None else os.path.join(trainer.log_dir, "results", datetime.now().strftime("%Y_%m_%d-%H_%M"))
        for batch_num, instance_output in enumerate(gen_outputs):
            local_path = Path(local_save_dir) / 'viz' / f'{batch["sample_token"][batch_num]}.png'
            if instance_output.shape[0] == 6:
                cond_input_sample = torch.cat([
                    batched_camera_bev_grid(pl_module.cfg, gen_outputs[batch_num], batch['segmentation'][batch_num].to(dtype=torch.float).detach().cpu().numpy()[None,])[0],
                    batched_camera_bev_grid(pl_module.cfg, outputs['gt'][batch_num], batch['segmentation'][batch_num].to(dtype=torch.float).detach().cpu().numpy()[None,])[0]
                ], dim=-2)
                self.save_images({f'sample': cond_input_sample}, trainer, pl_module, local_save=False)
                save_img(cond_input_sample, local_path)

                if self.save_dir is None:
                    continue
                
            tok = (batch['sample_token'][batch_num] + '_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)))
            for cam_num in range(instance_output.shape[0]):
                if self.rand_str:
                    img_save_path = Path(local_save_dir) / 'sample' / tok / f'{batch["cam_name"][cam_num][batch_num]}.jpg'
                    save_img(instance_output[cam_num], img_save_path)

                    img_save_path = Path(local_save_dir) / 'sample_gt' / tok / f'{batch["cam_name"][cam_num][batch_num]}.jpg'
                    save_img(outputs['gt'][batch_num, cam_num], img_save_path)

                    if cam_num == 0:
                        bev_save_path = Path(local_save_dir) / 'sample' / tok / 'bev.npz'
                        os.makedirs(bev_save_path.parents[0], exist_ok=True)
                        np.savez_compressed(bev_save_path, batch['segmentation'][batch_num].to(dtype=torch.float).detach().cpu())
                        viz_bev(batch['segmentation'][batch_num].detach().cpu().numpy(), dataset=pl_module.cfg.dataset).pil.save(bev_save_path.with_suffix('.png'))

                        bev_save_path = Path(local_save_dir) / 'sample_gt' / tok / 'bev.npz'
                        os.makedirs(bev_save_path.parents[0], exist_ok=True)
                        np.savez_compressed(bev_save_path, batch['segmentation'][batch_num].to(dtype=torch.float).detach().cpu())
                else:
                    img_save_path = Path(local_save_dir) / 'sample' / batch['sample_token'][batch_num] / f'{batch["cam_name"][cam_num][batch_num]}.jpg'
                    save_img(instance_output[cam_num], img_save_path)

                    img_save_path = Path(local_save_dir) / 'sample_gt' / batch['sample_token'][batch_num] / f'{batch["cam_name"][cam_num][batch_num]}.jpg'
                    save_img(outputs['gt'][batch_num, cam_num], img_save_path)

                    if cam_num == 0:
                        bev_save_path = Path(local_save_dir) / 'sample' / batch['sample_token'][batch_num] / 'bev.npz'
                        np.savez_compressed(bev_save_path, batch['segmentation'][batch_num].to(dtype=torch.float).detach().cpu())

                        if pl_module.cfg.dataset == Dataset.NUSCENES:
                            viz_bev(batch['segmentation'][batch_num].to(dtype=torch.float).detach().cpu().numpy(), dataset=pl_module.cfg.dataset).pil.save(bev_save_path.with_suffix('.png'))

                        bev_save_path = Path(local_save_dir) / 'sample_gt' / batch['sample_token'][batch_num] / 'bev.npz'
                        np.savez_compressed(bev_save_path, batch['segmentation'][batch_num].to(dtype=torch.float).detach().cpu())

                if save_nuscenes_fmt and pl_module.cfg.dataset == Dataset.NUSCENES and 'image_paths' in batch:
                    img_save_path = Path(local_save_dir) / Path('gt') / batch['image_paths'][cam_num][batch_num]
                    save_img(outputs['gt'][batch_num, cam_num], img_save_path)

                    img_save_path = Path(local_save_dir) / Path('rec') / batch['image_paths'][cam_num][batch_num]
                    save_img(outputs['rec'][batch_num, cam_num], img_save_path)

                    img_save_path = Path(local_save_dir) / Path('gen') / batch['image_paths'][cam_num][batch_num]
                    save_img(instance_output[cam_num], img_save_path)
                    np.savez(img_save_path.with_suffix('.npz'), batch['intrinsics'][batch_num, cam_num].to(dtype=torch.float).detach().cpu())


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if hasattr(pl_module, 'debug_viz') and pl_module.debug_viz and batch_idx % 100 == 0 and 'image' in batch:
            x = batch["image"]
            x = rearrange(x, '... h w c -> ... c h w')
            x = pl_module.combine_all_images(x)
            x = x.to(memory_format=torch.contiguous_format)
            x = util.denormalize_tensor(x, keep_tensor=True)
            x = pl_module.expand_all_images(x)
            x = batched_camera_bev_grid(pl_module.cfg, x)

        if batch_idx % 100 == 0 and isinstance(pl_module, VQModel):
            images = pl_module.log_images(batch)
            for k in images:
                pl_module.logger.experiment.log({f"train/{k}": wandb.Image(torchvision.utils.make_grid(images[k]))})

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        try: # Viz code has a lot of side effects so we catch and continue training
            if isinstance(outputs, dict):
                if 'gen' in outputs:
                    self.save_raw_data(trainer, pl_module, outputs, batch)
                self.save_images({k: v for k, v in outputs.items() if torch.is_tensor(v) and len(v.shape) == 4 and k not in ["gen", "gt", "rec"]}, trainer, pl_module)
            if batch_idx % 100 == 0 and isinstance(pl_module, VQModel):
                images = pl_module.log_images(batch)
                for k in images:
                    pl_module.logger.experiment.log({f"val/{k}": wandb.Image(torchvision.utils.make_grid(images[k]))})
        except Exception as ex:
            log.error("log_images failed", exc_info=ex)
            log.error("".join(traceback.TracebackException.from_exception(ex).format()))

    def on_test_batch_end(self, trainer: pl.trainer.Trainer, pl_module: pl.LightningModule, outputs: Generic, batch: Generic, batch_idx: int, dataloader_idx: int) -> None:
        self.save_raw_data(trainer, pl_module, outputs, batch)