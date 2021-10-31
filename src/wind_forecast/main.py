import os
from typing import cast

import hydra
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import setproctitle
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.loggers import WandbLogger
from wandb.sdk.wandb_run import Run

from wind_forecast.config.register import Config, register_configs, get_tags
from wind_forecast.util.callbacks import CustomCheckpointer, get_resume_checkpoint
from wind_forecast.util.logging import log
from wind_forecast.util.rundir import setup_rundir

wandb_logger: WandbLogger
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def plot_results(system, config: Config, mean, std):
    for index in np.random.choice(np.arange(len(system.test_results['output'])), 20, replace=False):
        fig, ax = plt.subplots()
        inputs_dates = [pd.to_datetime(pd.Timestamp(d)) for d in system.test_results['inputs_dates'][index]]
        output_dates = [pd.to_datetime(pd.Timestamp(d)) for d in system.test_results['targets_dates'][index]]

        inputs_dates.extend(output_dates)
        x = inputs_dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator())
        out_series = system.test_results['output'][index].cpu() * std + mean
        truth_series = (system.test_results['inputs'][index].cpu() * std + mean).tolist()
        truth_series.extend((system.test_results['labels'][index].cpu() * std + mean).tolist())
        ax.plot(output_dates, out_series, label='prediction')
        ax.plot(x, truth_series, label='ground truth')
        ax.set_xlabel('Date')
        ax.set_ylabel(config.experiment.target_parameter)
        ax.legend(loc='best')
        plt.gcf().autofmt_xdate()
        wandb.log({'chart': ax})


@hydra.main(config_path='config', config_name='default')
def main(cfg: Config):

    cfg.experiment.train_parameters_config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                               'config', 'train_parameters',
                                                               cfg.experiment.train_parameters_config_file)

    log.info(f'\\[init] Loaded config:\n{OmegaConf.to_yaml(cfg, resolve=True)}')

    pl.seed_everything(cfg.experiment.seed)

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
        period=1,
        dirpath='checkpoints',
        filename='{epoch}',
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

    trainer.fit(system, datamodule=datamodule)
    trainer.test(system, datamodule=datamodule)

    wandb_logger.log_metrics({
        'target_mean': datamodule.dataset_test.mean,
        'target_std': datamodule.dataset_test.std
    }, step=system.current_epoch)

    mean = datamodule.dataset_test.mean
    std = datamodule.dataset_test.std

    plot_results(system, cfg, mean, std)

    if trainer.interrupted:  # type: ignore
        log.info(f'[bold red]>>> Training interrupted.')
        run.finish(exit_code=255)


if __name__ == '__main__':
    setup_rundir()

    wandb.init(project=os.getenv('WANDB_PROJECT'),
               entity=os.getenv('WANDB_ENTITY'),
               name=os.getenv('RUN_NAME'))

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
