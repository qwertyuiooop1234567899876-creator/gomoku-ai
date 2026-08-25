from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.arena import GameResult, play_game
from engine.arena_settings import AISelection
from tools.telegram_notify import send_message


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_MODULE = "tools.cvc_analysis"


def _notify(text: str) -> None:
    """Send optional progress notifications without making workflows brittle."""
    if not os.environ.get("TELEGRAM_BOT_TOKEN", "").strip():
        return
    try:
        send_message(text)
    except Exception as error:  # noqa: BLE001 - notification is best effort
        print(f"Telegram notification skipped: {error}", file=sys.stderr)


@dataclass(frozen=True, slots=True)
class WorkflowStage:
    label: str
    black: AISelection
    white: AISelection


def default_stages() -> tuple[WorkflowStage, ...]:
    search = AISelection("search", max_depth=8, time_limit_seconds=60.0)
    yixin = AISelection("yixin", max_depth=3, time_limit_seconds=10.0)
    return (
        WorkflowStage("SearchAI vs SearchAI", search, search),
        WorkflowStage("SearchAI vs YiXin", search, yixin),
    )


def _saved_json(result: GameResult, *, label: str) -> Path:
    if result.record_paths is None:
        raise RuntimeError(f"{label} did not save a record.")
    record_path = result.record_paths.json.resolve()
    if not record_path.is_file():
        raise RuntimeError(f"{label} record was not found: {record_path}")
    return record_path


def analyze_record(record_path: Path) -> None:
    """Analyze the exact record returned by the preceding game."""
    subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            ANALYSIS_MODULE,
            str(record_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def run_workflow() -> tuple[Path, ...]:
    records: list[Path] = []
    stages = default_stages()
    for index, stage in enumerate(stages, start=1):
        print()
        print("=" * 64)
        print(f"Stage {index}/{len(stages)}: {stage.label}")
        print("=" * 64)
        result = play_game(
            black=stage.black,
            white=stage.white,
            watch=True,
            # Every new record is analyzed immediately in the next step.
            # Avoid launching an additional YiXin evaluator after every move;
            # it duplicates that work and, in the YiXin match, runs a second
            # external engine alongside the actual player.
            show_evaluation=False,
            save_record=True,
        )
        record_path = _saved_json(result, label=stage.label)
        records.append(record_path)
        _notify(f"Gomoku AI：{stage.label} 对局完成\n{record_path.name}")

        print()
        print(f"Analyzing the new record: {record_path}")
        analyze_record(record_path)
        _notify(f"Gomoku AI：{stage.label} 复盘完成\n{record_path.name}")

    return tuple(records)


def main() -> int:
    try:
        records = run_workflow()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Workflow failed: {error}", file=sys.stderr)
        _notify(f"Gomoku AI：工作流失败\n{error}")
        return 1

    print()
    print("All games and analyses completed successfully.")
    for record in records:
        print(f"  {record}")
    _notify("Gomoku AI：完整对局与复盘流程已完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
