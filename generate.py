import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from pytorch_lightning import Callback, LightningModule, Trainer
from multi_view_generation.utils.general import setup_callbacks
import os
import wandb
from omegaconf import OmegaConf
from multi_view_generation import utils
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
import hydra
from typing import List, Tuple
from image_utils import library_ops

log = utils.get_pylogger(__name__)


def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks = setup_callbacks(cfg)
    callbacks: List[Callback] = [*utils.instantiate_callbacks(cfg.get("callbacks")), *callbacks]

    log.info("Instantiating loggers...")
    project_name = 'debug' if cfg['task_name'] == 'debug' else os.path.basename(f'{cfg.paths.root_dir}_generate')
    run_name = f'{os.path.basename(cfg.paths.output_dir)}_{cfg.config_name}'
    logger = hydra.utils.instantiate(cfg.get("logger"), project=project_name, name=run_name)
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting inference!")
    model.learning_rate = cfg.trainer.accumulate_grad_batches * cfg.trainer.devices * datamodule.batch_size * cfg.base_lr

    try:
        trainer.test(model=model, datamodule=datamodule)
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


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
