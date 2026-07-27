from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True)
class TimeManager:
    """统一管理一次搜索的软、硬截止时间。"""

    started_at: float
    time_limit_seconds: float | None
    soft_ratio: float = 0.88
    hard_deadline: float | None = None
    soft_deadline: float | None = None

    def __post_init__(self) -> None:
        if self.time_limit_seconds is not None:
            if self.time_limit_seconds <= 0:
                raise ValueError("time_limit_seconds 必须大于 0 或为 None。")
            if not 0.5 <= self.soft_ratio < 1.0:
                raise ValueError("soft_ratio 必须在 0.5～1.0 之间。")
            self.soft_deadline = (
                self.started_at
                + self.time_limit_seconds * self.soft_ratio
            )
            self.hard_deadline = (
                self.started_at
                + self.time_limit_seconds
            )

    @classmethod
    def start(
        cls,
        time_limit_seconds: float | None,
        *,
        soft_ratio: float = 0.88,
    ) -> "TimeManager":
        return cls(
            started_at=time.perf_counter(),
            time_limit_seconds=time_limit_seconds,
            soft_ratio=soft_ratio,
        )

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    @property
    def remaining_seconds(self) -> float | None:
        if self.hard_deadline is None:
            return None
        return max(0.0, self.hard_deadline - time.perf_counter())

    def soft_expired(self) -> bool:
        return (
            self.soft_deadline is not None
            and time.perf_counter() >= self.soft_deadline
        )

    def hard_expired(self) -> bool:
        return (
            self.hard_deadline is not None
            and time.perf_counter() >= self.hard_deadline
        )

    def sub_deadline(
        self,
        fraction: float,
        *,
        minimum_seconds: float = 0.0,
        maximum_seconds: float | None = None,
    ) -> float | None:
        """返回不超过硬截止时间的子任务截止时刻。"""
        if not 0 < fraction <= 1:
            raise ValueError("fraction 必须在 0～1 之间。")

        if self.time_limit_seconds is None:
            return None

        budget = max(
            minimum_seconds,
            self.time_limit_seconds * fraction,
        )
        if maximum_seconds is not None:
            budget = min(budget, maximum_seconds)

        deadline = self.started_at + budget
        if self.hard_deadline is not None:
            deadline = min(deadline, self.hard_deadline)
        return deadline
