import os
from typing import cast

import hydra
import setproctitle
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.loggers import WandbLogger
from wandb.sdk.wandb_run import Run

from wind_forecast.config.register import Config, register_configs, get_tags
from wind_forecast.util.logging import log
from wind_forecast.util.rundir import setup_rundir

wandb_logger: WandbLogger


@hydra.main(config_path='config', config_name='default')
def main(cfg: Config):
    log.info(f'\\[init] Loaded config:\n{OmegaConf.to_yaml(cfg, resolve=True)}')

    pl.seed_everything(cfg.experiment.seed)

    cfg.experiment.train_parameters_config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config', 'train_parameters', cfg.experiment.train_parameters_config_file)

    RUN_NAME = os.getenv('RUN_NAME')
    log.info(f'[bold yellow]\\[init] Run name --> {RUN_NAME}')

    run: Run = wandb_logger.experiment  # type: ignore

    # Setup logging & checkpointing
    tags = get_tags(cast(DictConfig, cfg))
    run.tags = tags
    run.notes = str(cfg.notes)
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))  # type: ignore
    log.info(f'[bold yellow][{RUN_NAME} / {run.id}]: [bold white]{",".join(tags)}')

    setproctitle.setproctitle(f'{RUN_NAME} ({os.getenv("WANDB_PROJECT")})')  # type: ignore

    log.info(f'[bold white]Overriding cfg.lightning settings with derived values:')
    log.info(f' >>> num_sanity_val_steps = {-1 if cfg.experiment.validate_before_training else 0}\n')

    # Create main system (system = models + training regime)
    system: LightningModule = instantiate(cfg.experiment.system, cfg)
    log.info(f'[bold yellow]\\[init] System architecture:')
    log.info(system)
    # Prepare data using datamodules
    datamodule: LightningDataModule = instantiate(cfg.experiment.datamodule, cfg)

    trainer: pl.Trainer = instantiate(
        cfg.lightning,
        logger=wandb_logger,
        max_epochs=cfg.experiment.epochs,
        checkpoint_callback=True if cfg.experiment.save_checkpoints else False,
        num_sanity_val_steps=-1 if cfg.experiment.validate_before_training else 0
    )

    trainer.fit(system, datamodule=datamodule)
    trainer.test(system, datamodule=datamodule)

    if trainer.interrupted:  # type: ignore
        log.info(f'[bold red]>>> Training interrupted.')
        run.finish(exit_code=255)


if __name__ == '__main__':
    setup_rundir()

    wandb_logger = WandbLogger(
        project=os.getenv('WANDB_PROJECT'),
        entity=os.getenv('WANDB_ENTITY'),
        name=os.getenv('RUN_NAME'),
        save_dir=os.getenv('RUN_DIR'),
    )

    # Init logger from source dir (code base) before switching to run dir (results)
    wandb_logger.experiment  # type: ignore

    # Instantiate default Hydra config with environment variables & switch working dir
    register_configs()
    main()
