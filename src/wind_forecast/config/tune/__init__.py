from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TuneSettings:
    params: Any = field(default_factory=lambda: {})

    trials: int = 40

    # Set between 0 and 1 to prune after (max_epochs * prune_after_warmup_steps) epochs
    prune_after_warmup_steps: float = 0.5

    # Wait for metric to improve for (max_epochs * pruning_patience_factor) steps, then prune if run is below median results
    pruning_patience_factor: float = 0.25

    # do not prune if in (max_epochs * pruning_patience_factor) steps metric improved by at least patient_pruning_min_delta
    patient_pruning_min_delta: float = 0.05