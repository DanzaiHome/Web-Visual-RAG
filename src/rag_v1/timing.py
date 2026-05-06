from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TimingStats:
    durations: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def add(self, key: str, elapsed: float) -> None:
        self.durations[key] = self.durations.get(key, 0.0) + elapsed
        self.counts[key] = self.counts.get(key, 0) + 1

    def get_duration(self, key: str) -> float:
        return self.durations.get(key, 0.0)

    def get_count(self, key: str) -> int:
        return self.counts.get(key, 0)


_ACTIVE_TIMING: Optional[TimingStats] = None


def get_active_timing() -> Optional[TimingStats]:
    return _ACTIVE_TIMING


def set_active_timing(timing: Optional[TimingStats]) -> None:
    global _ACTIVE_TIMING
    _ACTIVE_TIMING = timing
