from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_ARENA_SETTINGS_PATH = Path("arena_settings.json")
VALID_ENGINES = ("random", "tactical", "scoring", "search")


@dataclass(frozen=True, slots=True)
class AISelection:
    """一方参赛 AI 的类型与 SearchAI 专属参数。"""

    engine_name: str = "search"
    max_depth: int = 3
    time_limit_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.engine_name not in VALID_ENGINES:
            raise ValueError(
                "engine_name 必须是 random、tactical、scoring 或 search。"
            )
        if not 1 <= self.max_depth <= 8:
            raise ValueError("max_depth 必须在 1～8 之间。")
        if not 0.1 <= self.time_limit_seconds <= 60.0:
            raise ValueError(
                "time_limit_seconds 必须在 0.1～60.0 秒之间。"
            )

    @property
    def uses_search(self) -> bool:
        return self.engine_name == "search"

    def with_engine(self, engine_name: str) -> "AISelection":
        """切换引擎时保留该方上一次 SearchAI 参数。"""
        return AISelection(
            engine_name=engine_name,
            max_depth=self.max_depth,
            time_limit_seconds=self.time_limit_seconds,
        )


@dataclass(frozen=True, slots=True)
class ArenaSettings:
    """AI 对战台的可持久化设置。"""

    black: AISelection = AISelection(
        engine_name="search",
        max_depth=3,
        time_limit_seconds=2.0,
    )
    white: AISelection = AISelection(
        engine_name="scoring",
        max_depth=3,
        time_limit_seconds=2.0,
    )
    watch: bool = True
    show_evaluation: bool = False
    delay_seconds: float = 0.0
    save_record: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.delay_seconds <= 10.0:
            raise ValueError("delay_seconds 必须在 0～10 秒之间。")


def _selection_from_payload(
    payload: dict[str, Any],
    fallback: AISelection,
) -> AISelection:
    return AISelection(
        engine_name=str(payload.get("engine_name", fallback.engine_name)),
        max_depth=int(payload.get("max_depth", fallback.max_depth)),
        time_limit_seconds=float(
            payload.get(
                "time_limit_seconds",
                fallback.time_limit_seconds,
            )
        ),
    )


def load_arena_settings(
    path: str | Path = DEFAULT_ARENA_SETTINGS_PATH,
) -> ArenaSettings:
    """读取对战台设置；缺失或损坏时恢复默认值。"""
    settings_path = Path(path)
    defaults = ArenaSettings()

    if not settings_path.exists():
        return defaults

    try:
        payload = json.loads(
            settings_path.read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict):
            raise TypeError

        black_payload = payload.get("black", {})
        white_payload = payload.get("white", {})
        if not isinstance(black_payload, dict):
            raise TypeError
        if not isinstance(white_payload, dict):
            raise TypeError

        return ArenaSettings(
            black=_selection_from_payload(
                black_payload,
                defaults.black,
            ),
            white=_selection_from_payload(
                white_payload,
                defaults.white,
            ),
            watch=bool(payload.get("watch", defaults.watch)),
            show_evaluation=bool(
                payload.get(
                    "show_evaluation",
                    defaults.show_evaluation,
                )
            ),
            delay_seconds=float(
                payload.get(
                    "delay_seconds",
                    defaults.delay_seconds,
                )
            ),
            save_record=bool(
                payload.get("save_record", defaults.save_record)
            ),
        )
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return defaults


def save_arena_settings(
    settings: ArenaSettings,
    path: str | Path = DEFAULT_ARENA_SETTINGS_PATH,
) -> Path:
    """以临时文件替换方式保存对战台设置。"""
    settings_path = Path(path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = settings_path.with_suffix(
        settings_path.suffix + ".tmp"
    )
    temporary_path.write_text(
        json.dumps(
            asdict(settings),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(settings_path)
    return settings_path
