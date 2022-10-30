import os
from typing import cast

import hydra
import optuna
import pytorch_lightning as pl
import setproctitle
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.loggers import WandbLogger, LightningLoggerBase
from wandb.sdk.wandb_run import Run

from wind_forecast.config.register import Config, register_configs, get_tags
from wind_forecast.runs_analysis import run_analysis
from wind_forecast.util.callbacks import CustomCheckpointer, get_resume_checkpoint
from wind_forecast.util.logging import log
from wind_forecast.util.plots import plot_results
from wind_forecast.util.rundir import setup_rundir

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def log_dataset_metrics(datamodule: LightningDataModule, logger: LightningLoggerBase, config: Config):
    metrics = {
        'train_dataset_length': len(datamodule.dataset_train),
        'val_dataset_length': len(datamodule.dataset_val),
        'test_dataset_length': len(datamodule.dataset_test)
    }

    mean = datamodule.dataset_test.dataset.mean
    std = datamodule.dataset_test.dataset.std

    if mean is not None:
        if type(mean) == list:
            for index, m in enumerate(mean):
                if type(m) == dict:
                    metrics[f"target_mean_{str(index)}"] = m[config.experiment.target_parameter]
                else:
                    metrics[f"target_mean_{str(index)}"] = m
        else:
            if type(mean) == dict:
                metrics[f"target_mean"] = mean[config.experiment.target_parameter]
            else:
                metrics[f"target_mean"] = mean
    if std is not None:
        if type(std) == list:
            for index, s in enumerate(std):
                if type(s) == dict:
                    metrics[f"target_std_{str(index)}"] = s[config.experiment.target_parameter]
                else:
                    metrics[f"target_std_{str(index)}"] = s
        else:
            if type(std) == dict:
                metrics[f"target_std"] = std[config.experiment.target_parameter]
            else:
                metrics[f"target_std"] = std

    logger.log_metrics(metrics)


def run_tune(cfg: Config):
    def objective(trial: optuna.trial.Trial, datamodule: LightningDataModule):
        config_exp = {}
        config_optim = {}
        for param in cfg.tune.params.keys():
            config_exp[param] = trial.suggest_categorical(param, list(cfg.tune.params[param]))

        if all(tag not in ['ARIMAX', 'SARIMAX'] for tag in cfg.experiment._tags_):
            config_exp['dropout'] = trial.suggest_uniform('dropout', 0, 0.8)
            if cfg.experiment.use_value2vec:
                config_exp['value2vec_embedding_factor'] = trial.suggest_int('value2vec_embedding_factor', 1, 20)
            if cfg.experiment.use_time2vec:
                config_exp['time2vec_embedding_factor'] = trial.suggest_int('time2vec_embedding_factor', 1, 20)

            config_optim['base_lr'] = trial.suggest_loguniform('base_lr', 0.000001, 0.01)
            if 'lambda_lr' in cfg.optim:
                config_optim['starting_lr'] = trial.suggest_loguniform('starting_lr', 0.000001, 0.01)
                config_optim['final_lr'] = trial.suggest_loguniform('final_lr', 0.000001, 0.01)
                config_optim['warmup_epochs'] = trial.suggest_int('warmup_epochs', 0, cfg.experiment.epochs)
                config_optim['decay_epochs'] = trial.suggest_int('decay_epochs', 0, cfg.experiment.epochs)

        for param in config_exp.keys():
            cfg.experiment.__setattr__(param, config_exp[param])

        for param in config_optim.keys():
            cfg.optim.__setattr__(param, config_optim[param])

        config = OmegaConf.to_container(cfg, resolve=True)
        config['trial.number'] = trial.number

        log.info(f'\\[init] Loaded config:\n{OmegaConf.to_yaml(cfg, resolve=True)}')

        RUN_NAME = os.getenv('RUN_NAME') + '-' + str(trial.number)
        WANDB_PROJECT = os.getenv('WANDB_PROJECT')
        WANDB_ENTITY = os.getenv('WANDB_ENTITY')
        wandb_logger = WandbLogger(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=RUN_NAME,
            save_dir=os.getenv('RUN_DIR')
        )
        wandb.init(project=WANDB_PROJECT,
                   entity=WANDB_ENTITY,
                   name=RUN_NAME,
                   config=config,
                   reinit=True)

        run: Run = wandb_logger.experiment  # type: ignore

        # Setup logging & checkpointing
        tags = get_tags(cast(DictConfig, cfg))
        run.tags = tags
        run.notes = str(cfg.notes)
        log.info(f'[bold yellow][{RUN_NAME} / {run.id}]: [bold white]{",".join(tags)}')

        # Create main system (system = models + training regime)
        system: LightningModule = instantiate(cfg.experiment.system, cfg)

        trainer: pl.Trainer = instantiate(
            cfg.lightning,
            logger=wandb_logger,
            max_epochs=cfg.experiment.epochs,
            gpus=cfg.lightning.gpus,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="ptl/val_mase")],
            checkpoint_callback=False,
            num_sanity_val_steps=-1 if cfg.experiment.validate_before_training else 0,
            check_val_every_n_epoch=cfg.experiment.check_val_every_n_epoch
        )
        if not cfg.experiment.skip_training:
            trainer.fit(system, datamodule)
        else:
            trainer.validate(system, datamodule)

        if trainer.interrupted:  # type: ignore
            log.info(f'[bold red]>>> Tuning interrupted.')
            run.finish(exit_code=255)

        val_loss = trainer.logged_metrics["ptl/val_mase"]
        run.summary["final loss"] = val_loss
        run.summary["state"] = "completed"
        run.finish(quiet=True)

        return val_loss

    epochs = cfg.experiment.epochs
    datamodule = instantiate(cfg.experiment.datamodule, cfg)

    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(
                                    n_warmup_steps=int(epochs * cfg.tune.prune_after_warmup_steps)),
                                    patience=int(epochs * cfg.tune.pruning_patience_factor),
                                    min_delta=cfg.tune.patient_pruning_min_delta))
    study.optimize(lambda trial: objective(trial, datamodule), n_trials=cfg.tune.trials)

    log.info("Number of finished trials: {}".format(len(study.trials)))

    log.info("Best trial:")
    trial = study.best_trial

    log.info("  Value: {}".format(trial.value))

    log.info("  Params: ")
    for key, value in trial.params.items():
        log.info("    {}: {}".format(key, value))


def run_training(cfg):
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')
    WANDB_ENTITY = os.getenv('WANDB_ENTITY')
    RUN_NAME = os.getenv('RUN_NAME')

    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=RUN_NAME,
        save_dir=os.getenv('RUN_DIR'),
        log_model='all' if os.getenv('LOG_MODEL') == 'all' else True if os.getenv('LOG_MODEL') == 'True' else False
    )
    wandb.init(project=WANDB_PROJECT,
               entity=WANDB_ENTITY,
               name=RUN_NAME)

    log.info(f'[bold yellow]\\[init] Run name --> {RUN_NAME}')

    run: Run = wandb_logger.experiment  # type: ignore

    # Setup logging & checkpointing
    tags = get_tags(cast(DictConfig, cfg))
    run.tags = tags
    run.notes = str(cfg.notes)
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))  # type: ignore
    log.info(f'[bold yellow][{RUN_NAME} / {run.id}]: [bold white]{",".join(tags)}')

    setproctitle.setproctitle(f'{RUN_NAME} ({os.getenv("WANDB_PROJECT")})')  # type: ignore

    log.info(f'\\[init] Loaded config:\n{OmegaConf.to_yaml(cfg, resolve=True)}')
    # Create main system (system = models + training regime)
    system: LightningModule = instantiate(cfg.experiment.system, cfg)
    log.info(f'[bold yellow]\\[init] System architecture:')
    log.info(system)
    wandb_logger.log_metrics({
        'n_params': sum(p.numel() for p in system.model.parameters() if p.requires_grad)
    })
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
        limit_val_batches=0 if cfg.experiment.skip_validation else 1,
        max_epochs=cfg.experiment.epochs,
        callbacks=[checkpointer],
        resume_from_checkpoint=resume_path,
        checkpoint_callback=True if cfg.experiment.save_checkpoints else False,
        num_sanity_val_steps=-1 if cfg.experiment.validate_before_training else 0,
        check_val_every_n_epoch=cfg.experiment.check_val_every_n_epoch
    )

    if not cfg.experiment.skip_training:
        trainer.fit(system, datamodule=datamodule)

    if not cfg.experiment.skip_test:
        trainer.test(system, datamodule=datamodule)

        mean = datamodule.dataset_test.dataset.mean
        std = datamodule.dataset_test.dataset.std

        if cfg.experiment.view_test_result:
            plot_results(system, cfg, mean, std)

    log_dataset_metrics(datamodule, wandb_logger, cfg)

    if trainer.interrupted:  # type: ignore
        log.info(f'[bold red]>>> Training interrupted.')
        run.finish(exit_code=255)

@hydra.main(config_path='config', config_name='default')
def main(cfg: Config):
    RUN_MODE = os.getenv('RUN_MODE', '').lower()
    if RUN_MODE == 'debug':
        cfg.debug_mode = True
    elif RUN_MODE in ['tune', 'tune_debug']:
        cfg.tune_mode = True
        if RUN_MODE == 'tune_debug':
            cfg.debug_mode = True

    pl.seed_everything(cfg.experiment.seed)

    cfg.experiment.train_parameters_config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                               'config', 'train_parameters',
                                                               cfg.experiment.train_parameters_config_file)

    if cfg.tune_mode:
        run_tune(cfg)
    elif RUN_MODE == 'analysis':
        run_analysis(cfg)
    else:
        run_training(cfg)


if __name__ == '__main__':
    setup_rundir()

    # Instantiate default Hydra config with environment variables & switch working dir
    register_configs()
    main()
