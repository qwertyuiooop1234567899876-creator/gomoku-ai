from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_SETTINGS_PATH = Path("search_settings.json")


@dataclass(frozen=True, slots=True)
class SearchSettings:
    """玩家可持久化的搜索参数。"""

    max_depth: int = 3
    time_limit_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_depth <= 8:
            raise ValueError("max_depth 必须在 1～8 之间。")
        if not 0.1 <= self.time_limit_seconds <= 60.0:
            raise ValueError(
                "time_limit_seconds 必须在 0.1～60.0 秒之间。"
            )


def load_search_settings(
    path: str | Path = DEFAULT_SETTINGS_PATH,
) -> SearchSettings:
    """读取本地搜索设置；文件缺失或损坏时使用默认值。"""
    settings_path = Path(path)

    if not settings_path.exists():
        return SearchSettings()

    try:
        payload = json.loads(
            settings_path.read_text(encoding="utf-8")
        )
        return SearchSettings(
            max_depth=int(payload["max_depth"]),
            time_limit_seconds=float(
                payload["time_limit_seconds"]
            ),
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return SearchSettings()


def save_search_settings(
    settings: SearchSettings,
    path: str | Path = DEFAULT_SETTINGS_PATH,
) -> Path:
    """以原子替换方式保存搜索设置。"""
    settings_path = Path(path)
    settings_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
