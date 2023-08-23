# ------------------------------------------------------------------------------------
# Modified from Taming Transformers (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
import random
import importlib
import pathlib
from typing import Optional, Tuple, List, Dict, ClassVar
import numpy as np
from omegaconf import OmegaConf
from datetime import datetime, timedelta

import torch
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
import wandb
from .callback import *
from omegaconf import DictConfig
log = utils.get_pylogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_obj_from_str(name: str, reload: bool = False) -> ClassVar:
    module, cls = name.rsplit(".", 1)

    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
        
    return getattr(importlib.import_module(module, package=None), cls)


def initialize_from_config(config: OmegaConf) -> object:
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def setup_callbacks(cfg: DictConfig) -> Tuple[List[Callback], WandbLogger, Dict]:
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{cfg.paths.output_dir}/checkpoints',
        every_n_epochs=1,
        save_last=True
    )

    checkpoint_callback_steps = ModelCheckpoint(
        dirpath=f'{cfg.paths.output_dir}/checkpoints',
        train_time_interval = timedelta(minutes=30),
        save_last=True
    )

    callbacks = [checkpoint_callback, checkpoint_callback_steps]
    
    from pytorch_lightning.callbacks import LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    return callbacks


def get_config_from_file(config_file: str) -> Dict:
    config_file = OmegaConf.load(config_file)

    if 'base_config' in config_file.keys():
        if config_file['base_config'].endswith(".yaml"):
            base_config = get_config_from_file(config_file['base_config'])

        config_file = {key: value for key, value in config_file if key != "base_config"}

        return OmegaConf.merge(base_config, config_file)
    
    return config_file



def convert_directory_chkpt_to_file(checkpoint_dir):
    from pytorch_lightning.utilities.deepspeed import ds_checkpoint_dir, get_optim_files, get_model_state_file, get_fp32_state_dict_from_zero_checkpoint
    if os.path.isdir(checkpoint_dir):
        # convert directory to single model pytorch
        CPU_DEVICE = torch.device("cpu")
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag=None)

        # additional logic to ensure we keep the lightning state dict as well from rank 0.
        deepspeed_states = [
            "module",
            "optimizer",
            "lr_scheduler",
            "csr_tensor_module_names",
            "skipped_steps",
            "global_steps",
            "dp_world_size",
            "mp_world_size",
        ]
        checkpoint_dir = ds_checkpoint_dir(checkpoint_dir)
        optim_files = get_optim_files(checkpoint_dir)
        optim_state = torch.load(optim_files[0], map_location=CPU_DEVICE)
        zero_stage = optim_state["optimizer_state_dict"]["zero_stage"]
        model_file = get_model_state_file(checkpoint_dir, zero_stage)
        client_state = torch.load(model_file, map_location=CPU_DEVICE)
        orig_state_dict = client_state["module"]
        client_state = {key: value for key, value in client_state.items() if key not in deepspeed_states}
        # State dict keys will include reference to wrapper _LightningModuleWrapperBase, Delete `module` prefix before saving.
        state_dict = {k.partition("module.")[2]: state_dict[k] for k in state_dict.keys()}
        orig_state_dict = {k.partition("module.")[2]: orig_state_dict[k] for k in orig_state_dict.keys()}

        orig_state_dict.update((k, state_dict[k]) for k in state_dict.keys() & client_state.keys())

        client_state["state_dict"] = orig_state_dict
        return client_state
    else:
        raise Exception()


def init_from_ckpt(module, path, ignore_keys=list(), unfrozen_keys=list(), strict=False):
    if os.path.isdir(path):
        sd = convert_directory_chkpt_to_file(path)
    else:
        sd = torch.load(path, map_location="cpu")

    if "state_dict" in sd.keys():
        sd = sd["state_dict"]


    for k in list(sd):
        if k.startswith('_forward_module'):
            tmp = sd[k]
            del sd[k]
            k = k.replace('_forward_module.', '')
            sd[k] = tmp

    for k in list(sd):
        for ik in ignore_keys:
            if ik in k:
                log.info("Deleting key {} from state_dict.".format(k))
                del sd[k]

    for n in module.state_dict().keys():
        if n not in sd.keys():
            print(f'Missing {n}')

    for n in sd.keys():
        if n not in module.state_dict().keys():
            print(f'Unexpected {n}')

    module.load_state_dict(sd, strict=strict)

    if len(unfrozen_keys) > 0:
        for n, p in module.named_parameters():
            p.requires_grad_ = False
            for unfrozen_name in unfrozen_keys:
                if unfrozen_name in n:
                    p.requires_grad_ = True
                    print(f'Unfreezing: {n}')

    log.info(f"Restored from {path}")