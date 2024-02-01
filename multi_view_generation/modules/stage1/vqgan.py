import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from einops import rearrange

from ...utils.general import initialize_from_config

from .model import Encoder, Decoder
from .quantize import VectorQuantizer2 as VectorQuantizer
from .quantize import GumbelQuantize
from .quantize import EMAVectorQuantizer
from multi_view_generation.bev_utils import util, nuscenes_helper
from multi_view_generation.modules.losses.vqperceptual import DummyLoss
from multi_view_generation import utils
from image_utils import Im
import numpy as np


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 cam_res,
                 cam_latent_res,
                 cam_emd_dim,
                 geometric_embedding=False,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 denormalize=True,
                 legacy=True,
                 **kwargs
                 ):
        super().__init__()
        self.image_key = image_key
        self.denormalize = denormalize
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = lossconfig
        self.geometric_embedding = geometric_embedding
        feat_height, feat_width = cam_latent_res
        image_height, image_width = cam_res

        if self.geometric_embedding:
            # intrinsic embedding
            image_plane = generate_grid(feat_height, feat_width)[None]
            image_plane[:, :, 0] *= image_width
            image_plane[:, :, 1] *= image_height
            self.register_buffer('image_plane', image_plane, persistent=False)
            self.img_embed = nn.Conv2d(4, cam_emd_dim, 1, bias=False)
            self.cam_embed = nn.Conv2d(4, cam_emd_dim, 1, bias=False)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape, legacy=legacy)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            utils.init_from_ckpt(self, ckpt_path, ignore_keys=ignore_keys, strict=False)
            
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def encode(self, x, batch):
        h = self.encoder(x)

        if self.geometric_embedding:
            h_flat = h[:, None, ...]
            # positional embedding part
            I_inv = rearrange(batch['intrinsics_inv'], 'b cam ... -> (b cam) ...')[:, None]
            E_inv = rearrange(batch['extrinsics_inv'], 'b cam ... -> (b cam) ...')[:, None]

            pixel = self.image_plane                                                # b n 3 h w
            _, _, _, h, w = pixel.shape

            c = E_inv[..., -1:]                                                     # b n 4 1
            c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]                # (b n) 4 1 1
            c_embed = self.cam_embed(c_flat)                                        # (b n) d 1 1

            pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (h w)
            cam = I_inv @ pixel_flat                                                # b n 3 (h w)
            cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                     # b n 4 (h w)
            d = E_inv @ cam                                                         # b n 4 (h w)
            d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)           # (b n) 4 h w
            d_embed = self.img_embed(d_flat)                                        # (b n) d h w

            img_embed = d_embed - c_embed  # (b n) d h w
            img_embed = img_embed / (img_embed.norm(dim=1,
                                                    keepdim=True) + 1e-7)  # (b n) d h w
            h_flat = rearrange(h_flat, 'b n ... -> (b n) ...')  # (b n) d h w
            h_flat += img_embed
            h = h_flat

        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, batch):
        quant, diff, _ = self.encode(input, batch)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]

        if len(x.shape) == 3:
            x = x[..., None]

        elif len(x.shape) == 5:
            x = x.squeeze()
            
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, batch)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, sync_dist=False)
            self.log_dict(log_dict_ae, sync_dist=False)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, sync_dist=False)
            self.log_dict(log_dict_disc, sync_dist=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, batch)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss, sync_dist=False)
        self.log("val/aeloss", aeloss, sync_dist=False)

        del log_dict_ae['val/rec_loss']
        self.log_dict(log_dict_ae, sync_dist=False)
        self.log_dict(log_dict_disc, sync_dist=False)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        if self.geometric_embedding:
            x, intrinsics, extrinsics = self.get_input(batch, self.image_key)
            x = x.to(self.device)
            xrec, _ = self(x, intrinsics, extrinsics)
        else:
            x = self.get_input(batch, self.image_key)
            x = x.to(self.device)
            xrec, _ = self(x, batch)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        else:
            if self.denormalize:
                x = util.denormalize_tensor(x, keep_tensor=True)
                xrec = util.denormalize_tensor(xrec, keep_tensor=True)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if x.shape[1] == 12:
            x = np.stack([nuscenes_helper.BaseViz().visualize_bev(x[i]) for i in range(len(x))], axis=0)
            x = Im(x).torch
        else:
            if not hasattr(self, "colorize"):
                self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
            if self.colorize.dtype == torch.half:
                x = x.half()
            x = F.conv2d(x, weight=self.colorize)
            x = (x-x.min())/(x.max()-x.min())
        return x


class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, batch)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        ret = {'loss': aeloss}
        if batch_idx % 500 == 0:
            ret = {**ret, **self.log_images(batch)}
        return ret

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, batch)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        ret = {'loss': aeloss}
        if batch_idx % 10 == 0:
            ret = {**ret, **self.log_images(batch)}
        return ret

    def test_step(self, batch, batch_idx):
        return self.log_images(batch)

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x, batch)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            if xrec.shape[1] == 12:
                xrec = torch.round(torch.sigmoid(xrec))
            else:
                xrec = torch.round(torch.sigmoid(xrec))
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x, intrinsics, extrinsics = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, intrinsics, extrinsics)
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x, intrinsics, extrinsics = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, intrinsics, extrinsics)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        output = pl.EvalResult(checkpoint_on=rec_loss)
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer


class GumbelVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 temperature_scheduler_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        self.quantize = GumbelQuantize(z_channels, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        self.temperature_scheduler = initialize_from_config(temperature_scheduler_config)   # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.temperature_scheduling()
        x, intrinsics, extrinsics = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, intrinsics, extrinsics)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x, intrinsics, extrinsics = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x, intrinsics, extrinsics = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(quant)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log


class EMAVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )
        self.quantize = EMAVectorQuantizer(n_embed=n_embed,
                                           embedding_dim=embed_dim,
                                           beta=0.25,
                                           remap=remap)
    def configure_optimizers(self):
        lr = self.learning_rate
        #Remove self.quantize from parameter list since it is updated via EMA
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []                                           