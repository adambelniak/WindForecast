from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from hydra_configs.torch.optim import AdamConf, SGDConf
from omegaconf import MISSING

OPTIMIZERS = {
    'adam': AdamConf,
    'sgd': SGDConf,
    # ...
}


@dataclass
class OptimSettings:
    optimizer: Any = MISSING
    scheduler: Optional[Any] = MISSING
    lr: float = 0.0001
    weight_decay = 0
    beta1 = 0.9
    beta2 = 0.999
