from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TuneSettings:
    params: Any = field(default_factory=lambda: {})