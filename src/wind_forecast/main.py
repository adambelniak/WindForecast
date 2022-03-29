import os
from typing import cast

import hydra
import pytorch_lightning as pl
from pathlib import Path
import setproctitle
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.loggers import WandbLogger
from ray.tune import SyncConfig
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from wandb.sdk.wandb_run import Run
from ray import tune

from wind_forecast.config.register import Config, register_configs, get_tags
from wind_forecast.util.callbacks import CustomCheckpointer, get_resume_checkpoint
from wind_forecast.util.logging import log
from wind_forecast.util.plots import plot_results
from wind_forecast.util.rundir import setup_rundir

from wind_forecast.util.common_util import wandb_logger
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def train_for_tune(tune_config, cfg: Config, model):
    metrics = {"loss": "ptl/val_loss"}

    for param in tune_config.keys():
        cfg.experiment.__setattr__(param, tune_config[param])

    dm = instantiate(cfg.experiment.datamodule, cfg)
    trainer: pl.Trainer = instantiate(
        cfg.lightning,
        max_epochs=cfg.experiment.epochs,
        gpus=cfg.lightning.gpus,
        callbacks=[TuneReportCallback(metrics, on="validation_end")],
        checkpoint_callback=False
    )
    trainer.fit(model, dm)


def run_tune(cfg: Config):
    config = {}

    for param in cfg.tune.params.keys():
        config[param] = tune.grid_search(list(cfg.tune.params[param]))

    # Create main system (system = models + training regime)
    system: LightningModule = instantiate(cfg.experiment.system, cfg)
    log.info(f'[bold yellow]\\[init] System architecture:')
    log.info(system)

    trainable = tune.with_parameters(
        train_for_tune,
        cfg=cfg,
        model=system)

    analysis = tune.run(
        trainable,
        resources_per_trial={
            "cpu": 4,
            "gpu": cfg.lightning.gpus
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=1,
        sync_config=SyncConfig(sync_to_driver=False),
        name="tune")

    print(analysis.best_config)


def run_training(cfg):
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

    resume_path = get_resume_checkpoint(cfg, wandb_logger)
    if resume_path is not None:
        log.info(f'[bold yellow]\\[checkpoint] [bold white]{resume_path}')

    checkpointer = CustomCheckpointer(
        dirpath='checkpoints',
        filename='{epoch}'
    )

    trainer: pl.Trainer = instantiate(
        cfg.lightning,
        logger=wandb_logger,
        max_epochs=cfg.experiment.epochs,
        callbacks=[checkpointer],
        resume_from_checkpoint=resume_path,
        checkpoint_callback=True if cfg.experiment.save_checkpoints else False,
        num_sanity_val_steps=-1 if cfg.experiment.validate_before_training else 0
    )

    if not cfg.experiment.skip_training:
        trainer.fit(system, datamodule=datamodule)

    trainer.test(system, datamodule=datamodule)

    metrics = {
        'train_dataset_length': len(datamodule.dataset_train),
        'test_dataset_length': len(datamodule.dataset_test)
    }

    mean = datamodule.dataset_test.mean
    std = datamodule.dataset_test.std

    if mean is not None:
        if type(mean) == list:
            for index in enumerate(mean):
                metrics[f"target_mean_{str(index)}"] = mean[index]
        else:
            metrics['target_mean'] = mean
    if std is not None:
        if type(std) == list:
            for index in enumerate(std):
                metrics[f"target_std_{str(index)}"] = std[index]
        else:
            metrics['target_std'] = std

    wandb_logger.log_metrics(metrics, step=system.current_epoch)

    if cfg.experiment.view_test_result:
        plot_results(system, cfg, mean, std)

    if trainer.interrupted:  # type: ignore
        log.info(f'[bold red]>>> Training interrupted.')
        run.finish(exit_code=255)


@hydra.main(config_path='config', config_name='default')
def main(cfg: Config):
    RUN_MODE = os.getenv('RUN_MODE', '').lower()
    if RUN_MODE == 'debug':
        cfg.debug_mode = True

    log.info(f'\\[init] Loaded config:\n{OmegaConf.to_yaml(cfg, resolve=True)}')

    pl.seed_everything(cfg.experiment.seed)

    cfg.experiment.train_parameters_config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                               'config', 'train_parameters',
                                                               cfg.experiment.train_parameters_config_file)

    if RUN_MODE == 'tune':
        cfg.tune_mode = True
        run_tune(cfg)
    else:
        run_training(cfg)


if __name__ == '__main__':
    setup_rundir()

    wandb.init(project=os.getenv('WANDB_PROJECT'),
               entity=os.getenv('WANDB_ENTITY'),
               name=os.getenv('RUN_NAME'))

    # Init logger from source dir (code base) before switching to run dir (results)
    wandb_logger.experiment  # type: ignore

    # Instantiate default Hydra config with environment variables & switch working dir
    register_configs()
    main()
