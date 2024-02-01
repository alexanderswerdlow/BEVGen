from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.logger import Logger
from typing import List, Optional
from pytorch_lightning import Callback, LightningModule, Trainer
from multi_view_generation import utils
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf import DictConfig, OmegaConf
import hydra
import glob
import wandb
import torch
from multi_view_generation.utils.general import setup_callbacks
import signal
import pytorch_lightning as pl
from pathlib import Path
import os
import pyrootutils
from image_utils import library_ops

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
log = utils.get_pylogger(__name__)
wandb.require("service")

class TorchTensorboardProfilerCallback(pl.Callback):
    def __init__(self, profiler):
        super().__init__()
        self.profiler = profiler

    def on_train_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self.profiler.step()
        if type(outputs) == list:
            for i in outputs:
                pl_module.log_dict(i)
        else:
            pl_module.log_dict({'output': outputs})  # also logging the loss, while we're here


@utils.task_wrapper
def train(cfg: DictConfig) -> None:

    pl.seed_everything(cfg.seed)

    trainer_extra_args = {}
    profile_dir = None

    try:
        log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
        datamodule = instantiate(cfg.datamodule)

        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        log.info("Instantiating callbacks...")
        callbacks = setup_callbacks(cfg)
        callbacks: List[Callback] = [*utils.instantiate_callbacks(cfg.get("callbacks")), *callbacks]

        log.info("Instantiating loggers...")
        project_name = 'debug' if cfg['task_name'] == 'debug' else os.path.basename(cfg.paths.root_dir)
        run_name = f'{os.path.basename(cfg.paths.output_dir)}_{cfg.config_name}'
        logger = hydra.utils.instantiate(cfg.get("logger"), project=project_name, name=run_name)
        wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        profiler = None

        # Profiling with this flag may generate very large files.
        if cfg.extras.profile:
            profile_dir = Path(cfg.paths.output_dir) / 'profile'
            wait, warmup, active, repeat = 3, 3, 10, 0
            total_steps = (wait + warmup + active) * (1 + repeat)
            schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)

            os.makedirs(profile_dir, exist_ok=True)
            profiler = torch.profiler.profile(
                schedule=schedule,  # see the profiler docs for details on scheduling
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
                record_shapes=True,
                with_modules=True,
                with_flops=True,
                profile_memory=True,
                with_stack=True
            )

            trainer_extra_args['limit_train_batches'] = total_steps + 5
            trainer_extra_args['limit_val_batches'] = 0

            from torch.utils.tensorboard import SummaryWriter
            summary = SummaryWriter(log_dir=profile_dir)

            profiler_callback = TorchTensorboardProfilerCallback(profiler)
            callbacks = [*callbacks, profiler_callback]

        model.learning_rate = cfg.trainer.accumulate_grad_batches * cfg.trainer.devices * cfg.base_lr

        if cfg['task_name'] == 'debug':
            logger.watch(model, log="all", log_freq=5, log_graph=True)
        else:
            logger.watch(model, log_graph=True)

        # Context required for profiling
        from contextlib import nullcontext
        with profiler if profiler else nullcontext() as profiler:
            trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, **trainer_extra_args)

            object_dict = {
                "cfg": cfg,
                "datamodule": datamodule,
                "model": model,
                "callbacks": callbacks,
                "logger": logger,
                "trainer": trainer,
                "base_lr": cfg.base_lr,
                "learning_rate": model.learning_rate,
            }
            if logger:
                log.info("Logging hyperparameters!")
                utils.log_hyperparameters(object_dict)
            
            torch.set_float32_matmul_precision("medium")
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    except Exception as e:
        log.error(e)
        if cfg['task_name'] == 'debug' and trainer.global_rank == 0:
            print('Exception...')
            import traceback
            import pdb
            import sys
            traceback.print_exc()
            pdb.post_mortem(e.__traceback__)
            sys.exit(1)
        raise
    finally:
        if trainer.global_rank == 0 and cfg.extras.profile:
            print('Finding traces')
            traces = glob.glob(f"{profile_dir}/*.pt.trace.json")
            for trace in traces:
                print(f'Adding {trace}')
                wandb.save(trace, base_path=profile_dir)
            wandb.finish()


@hydra.main(version_base="1.2", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    train(cfg)


if __name__ == "__main__":
    main()
