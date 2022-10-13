from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Any, Optional, Iterator, cast, Dict

import omegaconf
from hydra.conf import HydraConf, RunDir, SweepDir
from hydra.core.config_store import ConfigStore
from omegaconf import SI, MISSING, DictConfig

from wind_forecast.config.analysis import AnalysisSettings
from wind_forecast.config.experiment import ExperimentSettings
from wind_forecast.config.lightning import LightningSettings
from wind_forecast.config.optim import OptimSettings, OPTIMIZERS
from wind_forecast.config.tune import TuneSettings


@dataclass
class Hydra(HydraConf):
    run: RunDir = RunDir("${output_dir}")
    sweep: SweepDir = SweepDir(".", "${output_dir}")


@dataclass
class Config:
    """
    Top-level Hydra config class.
    """
    defaults: List[Any] = field(default_factory=lambda: [
        {'experiment': 'gfs'},
        {'lightning': 'default'},
        {'optim': 'adam'},
        {'tune': 'sarimax'},
        {'analysis': 'analysis'},
        {'override hydra/job_logging': 'rich'},
        {'override hydra/hydra_logging': 'rich'},
    ])

    # Path settings
    data_dir: str = SI("${oc.env:DATA_DIR}")
    output_dir: str = SI("${oc.env:RUN_DIR}")

    # Runtime configuration
    hydra: Hydra = Hydra()

    # Experiment settings --> experiment/*.yaml
    experiment: ExperimentSettings = MISSING

    # Lightning settings --> lightning/*.yaml
    lightning: LightningSettings = MISSING

    # Optimizer & scheduler settings --> optim/*.yaml
    optim: OptimSettings = MISSING

    tune: TuneSettings = MISSING

    analysis: AnalysisSettings = MISSING

    debug_mode: bool = False

    tune_mode: bool = False

    # wandb metadata
    notes: Optional[str] = None
    tags: Optional[List[str]] = None


def register_configs():
    cs = ConfigStore.instance()

    # Main config
    cs.store(name='default', node=DictConfig(Config()))

    # Config groups with defaults, YAML files validated by Python structured configs
    # e.g.: `python -m wind_forecast.main experiment=cnn`
    cs.store(group='experiment', name='schema_experiment', node=ExperimentSettings)
    cs.store(group='lightning', name='schema_lightning', node=LightningSettings)
    cs.store(group='optim', name='schema_optim', node=OptimSettings)
    cs.store(group='tune', name='schema_tune', node=TuneSettings)
    cs.store(group='analysis', name='schema_analysis', node=AnalysisSettings)

    for key, node in OPTIMIZERS.items():
        name = f'schema_optim_{key}'
        cs.store(group='optim', name=name, node=node, package='optim.optimizer')


def _get_tags(cfg: dict[str, Any]) -> Iterator[str]:
    for key, value in cfg.items():
        if isinstance(value, dict):
            yield from _get_tags(cast(Dict[str, Any], value))
        if key == '_tags_':
            if isinstance(value, list):
                for v in cast(List[str], value):
                    yield v
            else:
                if value is not None:
                    value = cast(str, value)
                    yield value


def get_tags(cfg: DictConfig):
    """
    Extract all tags from a nested DictConfig object.
    """
    cfg_dict = cast(Dict[str, Any], omegaconf.OmegaConf.to_container(cfg, resolve=True))
    if 'tags' in cfg_dict:
        cfg_dict['_tags_'] = cfg_dict['tags']
    return list(_get_tags(cfg_dict))
