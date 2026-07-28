from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from engine.ai import RandomAI, ScoringAI, TacticalAI
from engine.arena_settings import (
    AISelection,
    ArenaSettings,
    VALID_ENGINES,
    load_arena_settings,
    save_arena_settings,
)
from engine.board import BLACK, WHITE, Board
from engine.evaluator import evaluate_board, render_evaluation_bar
from engine.game import format_move, other_player, player_name
from engine.records import GameRecorder, RecordPaths
from engine.search import SearchAI

ENGINE_VERSION = "0.8.5"


class GomokuAI(Protocol):
    def choose_move(self, board: Board) -> tuple[int, int]: ...


@dataclass(slots=True)
class GameResult:
    winner: int | None
    move_count: int
    duration_seconds: float
    record_paths: RecordPaths | None


ENGINE_MENU = {
    "1": "random",
    "2": "tactical",
    "3": "scoring",
    "4": "search",
}

ENGINE_LABELS = {
    "random": "RandomAI（随机基准）",
    "tactical": "TacticalAI（胜负与封堵）",
    "scoring": "ScoringAI（V0.6.2 评分）",
    "search": "SearchAI（V0.8.5 防守分支 VCT 探针与前沿引导 PVS）",
}


def result_text(winner: int | None) -> str:
    if winner == BLACK:
        return "黑棋 X 获胜"
    if winner == WHITE:
        return "白棋 O 获胜"
    return "平局"


def create_ai(selection: AISelection, player: int) -> GomokuAI:
    """按选择创建任意阶段的 AI。"""
    if selection.engine_name == "random":
        return RandomAI()

    if selection.engine_name == "tactical":
        return TacticalAI(player=player)

    if selection.engine_name == "scoring":
        return ScoringAI(
            player=player,
            diagnostics=True,
            top_n=5,
        )

    if selection.engine_name == "search":
        return SearchAI(
            player=player,
            max_depth=selection.max_depth,
            time_limit_seconds=selection.time_limit_seconds,
            diagnostics=True,
            top_n=5,
        )

    raise ValueError(f"未知引擎：{selection.engine_name}")


def engine_display_name(selection: AISelection) -> str:
    """生成棋谱和终端中使用的参赛名称。"""
    if selection.engine_name == "search":
        return (
            "SearchAI"
            f"(d={selection.max_depth},"
            f"t={selection.time_limit_seconds:g}s)"
        )

    return {
        "random": "RandomAI",
        "tactical": "TacticalAI",
        "scoring": "ScoringAI",
    }[selection.engine_name]


def engine_file_token(selection: AISelection) -> str:
    if selection.engine_name == "search":
        time_text = f"{selection.time_limit_seconds:g}".replace(".", "p")
        return f"search-d{selection.max_depth}-t{time_text}"
    return selection.engine_name


def _analysis_to_dict(ai: GomokuAI) -> dict[str, object] | None:
    analysis = getattr(ai, "last_analysis", None)
    if analysis is None:
        return None
    return analysis.to_dict()


def play_game(
    *,
    black: AISelection,
    white: AISelection,
    watch: bool = False,
    show_evaluation: bool = False,
    delay_seconds: float = 0.0,
    save_record: bool = True,
) -> GameResult:
    """让任意两个阶段的 AI 自动完成一局并保存诊断。"""
    if not 0.0 <= delay_seconds <= 10.0:
        raise ValueError("delay_seconds 必须在 0～10 秒之间。")

    board = Board()
    black_name = engine_display_name(black)
    white_name = engine_display_name(white)
    players: dict[int, GomokuAI] = {
        BLACK: create_ai(black, BLACK),
        WHITE: create_ai(white, WHITE),
    }
    recorder = GameRecorder(
        mode="CVC",
        black_name=black_name,
        white_name=white_name,
    )

    current_player = BLACK
    winner: int | None = None
    game_started = time.perf_counter()

    print()
    print(f"Gomoku AI Arena V{ENGINE_VERSION}")
    print(f"黑棋：{black_name}")
    print(f"白棋：{white_name}")
    print("=" * 50)

    while not board.is_full():
        ai = players[current_player]
        evaluation_before = evaluate_board(board, WHITE)

        think_started = time.perf_counter()
        row, column = ai.choose_move(board)
        think_seconds = time.perf_counter() - think_started

        if not board.is_inside(row, column):
            raise RuntimeError(
                f"{player_name(current_player)} 返回越界坐标："
                f"({row}, {column})"
            )
        if not board.is_empty(row, column):
            raise RuntimeError(
                f"{player_name(current_player)} 返回已占用坐标："
                f"{format_move(row, column)}"
            )

        board.place(row, column, current_player)
        evaluation_after = evaluate_board(board, WHITE)
        analysis = _analysis_to_dict(ai)
        actor = engine_display_name(
            black if current_player == BLACK else white
        )

        move_record = recorder.record_move(
            player=current_player,
            row=row,
            column=column,
            actor=actor,
            think_seconds=think_seconds,
            evaluation_before=evaluation_before,
            evaluation_after=evaluation_after,
            analysis=analysis,
        )

        reason = analysis["reason"] if analysis else "基础策略"
        search_text = ""
        if analysis and (
            int(analysis.get("search_depth", 0)) > 0
            or int(analysis.get("nodes", 0)) > 0
        ):
            search_text = (
                f" 深度={analysis.get('search_depth', 0)}/"
                f"{analysis.get('requested_depth', 0)}"
                f" 节点={analysis.get('nodes', 0)}"
                f" NPS={analysis.get('nps', 0)}"
                f" 剪枝={analysis.get('cutoffs', 0)}"
                f" TT={analysis.get('transposition_hits', 0)}"
            )

        print(
            f"{move_record.number:3}. "
            f"{player_name(current_player)} "
            f"{format_move(row, column)} "
            f"思考 {think_seconds:.3f}s "
            f"原因：{reason}{search_text}"
        )

        if board.check_win(row, column):
            winner = current_player
            break

        next_player = other_player(current_player)

        if watch:
            print()
            print(board)
            print()
            print(recorder.render_score_sheet(last_rounds=8))
            print()

            if show_evaluation:
                print(
                    render_evaluation_bar(
                        board,
                        current_player=next_player,
                    )
                )
                print()

            if delay_seconds > 0:
                time.sleep(delay_seconds)

        current_player = next_player

    duration_seconds = time.perf_counter() - game_started
    result = result_text(winner)

    print()
    print("=" * 50)
    print(f"{black_name}（黑） vs {white_name}（白）")
    print(result)
    print(f"总手数：{len(recorder.moves)}")
    print(f"总耗时：{duration_seconds:.3f}s")
    print()
    print(board)
    print()
    print(recorder.render_score_sheet(full=True))

    record_paths: RecordPaths | None = None
    if save_record:
        prefix = (
            f"{engine_file_token(black)}-vs-"
            f"{engine_file_token(white)}-v080"
        )
        record_paths = recorder.save(
            board=board,
            result=result,
            duration_seconds=duration_seconds,
            prefix=prefix,
        )
        print()
        print(f"TXT 棋谱：{record_paths.txt}")
        print(f"JSON 诊断：{record_paths.json}")

    return GameResult(
        winner=winner,
        move_count=len(recorder.moves),
        duration_seconds=duration_seconds,
        record_paths=record_paths,
    )


def _prompt_engine(
    side_text: str,
    current: AISelection,
) -> AISelection:
    print()
    print(f"请选择{side_text}引擎：")
    print("  1  RandomAI   随机落子基准")
    print("  2  TacticalAI 立即胜负、封堵、邻近落子")
    print("  3  ScoringAI  V0.6.2 棋型评分与复合威胁")
    print("  4  SearchAI   V0.8.5 防守分支 VCT 探针、前沿引导 PVS、VCF 与独立 100k TT")

    default_number = {
        "random": "1",
        "tactical": "2",
        "scoring": "3",
        "search": "4",
    }[current.engine_name]

    while True:
        raw = input(
            f"{side_text}引擎 [{default_number}]："
        ).strip().lower()
        if raw == "":
            engine_name = current.engine_name
            break
        if raw in ENGINE_MENU:
            engine_name = ENGINE_MENU[raw]
            break
        if raw in VALID_ENGINES:
            engine_name = raw
            break
        print("输入无效：请输入 1～4，或引擎英文名。")

    selected = current.with_engine(engine_name)
    if not selected.uses_search:
        return selected

    while True:
        raw_depth = input(
            f"{side_text} SearchAI depth "
            f"[{selected.max_depth}]（1～8）："
        ).strip()
        try:
            depth = (
                selected.max_depth
                if raw_depth == ""
                else int(raw_depth)
            )
            if not 1 <= depth <= 8:
                raise ValueError
            break
        except ValueError:
            print("输入无效：depth 必须是 1～8 的整数。")

    while True:
        raw_time = input(
            f"{side_text} SearchAI time-limit "
            f"[{selected.time_limit_seconds:g}] 秒（0.1～60）："
        ).strip()
        try:
            time_limit = (
                selected.time_limit_seconds
                if raw_time == ""
                else float(raw_time)
            )
            if not 0.1 <= time_limit <= 60.0:
                raise ValueError
            break
        except ValueError:
            print("输入无效：time-limit 必须在 0.1～60 秒之间。")

    return AISelection(
        engine_name="search",
        max_depth=depth,
        time_limit_seconds=time_limit,
    )


def _prompt_bool(label: str, current: bool) -> bool:
    default_text = "Y" if current else "N"
    while True:
        raw = input(f"{label} [{default_text}]：").strip().lower()
        if raw == "":
            return current
        if raw in {"y", "yes", "1", "是"}:
            return True
        if raw in {"n", "no", "0", "否"}:
            return False
        print("输入无效：请输入 Y 或 N。")


def choose_interactive_settings(
    settings_path: str | Path = "arena_settings.json",
) -> ArenaSettings:
    """交互选择双方 AI，并记忆为下次默认。"""
    current = load_arena_settings(settings_path)

    print(f"Gomoku AI Arena V{ENGINE_VERSION}")
    print("任意阶段 AI 对战；直接回车沿用上次选择。")

    black = _prompt_engine("黑棋", current.black)
    white = _prompt_engine("白棋", current.white)
    watch = _prompt_bool("每手显示棋盘", current.watch)
    show_evaluation = _prompt_bool(
        "同时显示评价条",
        current.show_evaluation,
    )

    while True:
        raw_delay = input(
            "每手额外暂停秒 "
            f"[{current.delay_seconds:g}]（0～10）："
        ).strip()
        try:
            delay_seconds = (
                current.delay_seconds
                if raw_delay == ""
                else float(raw_delay)
            )
            if not 0.0 <= delay_seconds <= 10.0:
                raise ValueError
            break
        except ValueError:
            print("输入无效：暂停时间必须在 0～10 秒之间。")

    save_record = _prompt_bool(
        "保存 TXT 与 JSON 棋谱",
        current.save_record,
    )

    selected = ArenaSettings(
        black=black,
        white=white,
        watch=watch,
        show_evaluation=show_evaluation,
        delay_seconds=delay_seconds,
        save_record=save_record,
    )
    save_arena_settings(selected, settings_path)

    print()
    print("本局配置已保存为下次默认：")
    print(f"  黑棋：{engine_display_name(selected.black)}")
    print(f"  白棋：{engine_display_name(selected.white)}")
    print(
        f"  观看={selected.watch}，"
        f"评价条={selected.show_evaluation}，"
        f"延迟={selected.delay_seconds:g}s"
    )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V0.8.5 AI 对战台：Random/Tactical/Scoring/Search "
            "可任意组合。无参数运行时进入交互菜单。"
        ),
    )
    parser.add_argument(
        "--black",
        choices=VALID_ENGINES,
        default="search",
        help="黑棋引擎。",
    )
    parser.add_argument(
        "--white",
        choices=VALID_ENGINES,
        default="scoring",
        help="白棋引擎。",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="兼容参数：同时设置双方 SearchAI depth。",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="兼容参数：同时设置双方 SearchAI time-limit。",
    )
    parser.add_argument("--black-depth", type=int, default=None)
    parser.add_argument("--black-time-limit", type=float, default=None)
    parser.add_argument("--white-depth", type=int, default=None)
    parser.add_argument("--white-time-limit", type=float, default=None)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="每手显示棋盘与最近着法。",
    )
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="观看模式下显示评价条。",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="每手额外暂停秒数。",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存棋谱。",
    )
    return parser.parse_args()


def _selection_from_cli(
    engine_name: str,
    side_depth: int | None,
    side_time_limit: float | None,
    shared_depth: int | None,
    shared_time_limit: float | None,
) -> AISelection:
    return AISelection(
        engine_name=engine_name,
        max_depth=(
            side_depth
            if side_depth is not None
            else shared_depth if shared_depth is not None else 3
        ),
        time_limit_seconds=(
            side_time_limit
            if side_time_limit is not None
            else (
                shared_time_limit
                if shared_time_limit is not None
                else 2.0
            )
        ),
    )


def main() -> None:
    if len(sys.argv) == 1:
        settings = choose_interactive_settings()
        play_game(
            black=settings.black,
            white=settings.white,
            watch=settings.watch,
            show_evaluation=settings.show_evaluation,
            delay_seconds=settings.delay_seconds,
            save_record=settings.save_record,
        )
        return

    args = parse_args()
    black = _selection_from_cli(
        args.black,
        args.black_depth,
        args.black_time_limit,
        args.depth,
        args.time_limit,
    )
    white = _selection_from_cli(
        args.white,
        args.white_depth,
        args.white_time_limit,
        args.depth,
        args.time_limit,
    )
    play_game(
        black=black,
        white=white,
        watch=args.watch,
        show_evaluation=args.evaluation,
        delay_seconds=args.delay,
        save_record=not args.no_save,
    )


if __name__ == "__main__":
    main()
