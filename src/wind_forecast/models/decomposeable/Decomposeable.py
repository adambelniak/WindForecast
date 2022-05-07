from typing import List

import PyEMD
import numpy
import torch
from pytorch_lightning import LightningModule


class Decomposeable(LightningModule):
    def __init__(self) -> None:
        super().__init__()

    def emd_decompose(self, x: torch.Tensor):
        raise NotImplementedError()

    # (batch, timestep, 1)
    def decompose(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    # (batch, component, timestep, 1)
    def compose(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class EMDDecomposeable(Decomposeable):
    def __init__(self, trials: int) -> None:
        super().__init__()
        self.trials = trials

    def emd_decompose(self, x: torch.Tensor):
        emd = PyEMD.EEMD(self.trials)
        if len(x.size()) == 2:
            return numpy.expand_dims(emd.eemd(x.squeeze().cpu().numpy(), max_imf=4), -1)
        return emd.eemd(x.cpu().numpy(), max_imf=4)

    # (batch, timestep, 1)
    def decompose(self, batch: torch.Tensor) -> List[torch.Tensor]:
        return [
            torch.Tensor(self.emd_decompose(x_i)).cuda() for x_i in torch.unbind(batch, dim=0)
        ]

    # (batch, component, timestep, 1)
    def compose(self, decomposed_batch: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack([torch.sum(component, 0) for component in decomposed_batch], 0)

