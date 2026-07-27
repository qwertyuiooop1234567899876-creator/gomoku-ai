from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from engine.ai import ScoringAI
from engine.board import BLACK, WHITE, Board
from engine.evaluator import evaluate_board, render_evaluation_bar
from engine.game import format_move, other_player, player_name
from engine.records import GameRecorder, RecordPaths
from engine.search import SearchAI

AIType = ScoringAI | SearchAI


@dataclass(slots=True)
class GameResult:
    winner: int | None
    move_count: int
    duration_seconds: float
    record_paths: RecordPaths | None


def result_text(winner: int | None) -> str:
    if winner == BLACK:
        return "黑棋 X 获胜"
    if winner == WHITE:
        return "白棋 O 获胜"
    return "平局"


def create_ai(
    engine_name: str,
    player: int,
    *,
    max_depth: int,
    time_limit_seconds: float | None,
) -> AIType:
    if engine_name == "scoring":
        return ScoringAI(
            player=player,
            diagnostics=True,
            top_n=5,
        )

    if engine_name == "search":
        return SearchAI(
            player=player,
            max_depth=max_depth,
            time_limit_seconds=time_limit_seconds,
            diagnostics=True,
            top_n=5,
        )

    raise ValueError(f"未知引擎：{engine_name}")


def engine_display_name(engine_name: str) -> str:
    return "SearchAI" if engine_name == "search" else "ScoringAI"


def play_game(
    *,
    black_engine: str = "search",
    white_engine: str = "scoring",
    max_depth: int = 3,
    time_limit_seconds: float | None = 2.0,
    watch: bool = False,
    show_evaluation: bool = False,
    delay_seconds: float = 0.0,
    save_record: bool = True,
) -> GameResult:
    """让 SearchAI/ScoringAI 任意组合自动完成一局并保存诊断。"""
    if delay_seconds < 0:
        raise ValueError("delay_seconds 不能小于 0。")

    board = Board()
    black_name = engine_display_name(black_engine)
    white_name = engine_display_name(white_engine)
    players = {
        BLACK: create_ai(
            black_engine,
            BLACK,
            max_depth=max_depth,
            time_limit_seconds=time_limit_seconds,
        ),
        WHITE: create_ai(
            white_engine,
            WHITE,
            max_depth=max_depth,
            time_limit_seconds=time_limit_seconds,
        ),
    }
    recorder = GameRecorder(
        mode="CVC",
        black_name=black_name,
        white_name=white_name,
    )

    current_player = BLACK
    winner: int | None = None
    game_started = time.perf_counter()

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
        analysis = (
            ai.last_analysis.to_dict()
            if ai.last_analysis is not None
            else None
        )

        actor = type(ai).__name__
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

        reason = analysis["reason"] if analysis else "未记录"
        search_text = ""
        if analysis and analysis.get("search_depth", 0) > 0:
            search_text = (
                f" 深度={analysis['search_depth']}"
                f" 节点={analysis['nodes']}"
                f" 剪枝={analysis['cutoffs']}"
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
        prefix = f"{black_engine}-vs-{white_engine}-v07"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V0.7 AI 对战台：SearchAI 与 ScoringAI 可自由组合。",
    )
    parser.add_argument(
        "--black",
        choices=("search", "scoring"),
        default="search",
        help="黑棋引擎，默认 search。",
    )
    parser.add_argument(
        "--white",
        choices=("search", "scoring"),
        default="scoring",
        help="白棋引擎，默认 scoring，用于新旧版本对比。",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="SearchAI 最大迭代深度，默认 3。",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=2.0,
        help="SearchAI 每手时间上限（秒），默认 2。",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="每手棋后显示完整棋盘和最近着法。",
    )
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="观看模式下同时显示实时评价条。",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="每手棋后的暂停秒数，例如 0.3。",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存棋谱与 JSON 诊断。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    play_game(
        black_engine=args.black,
        white_engine=args.white,
        max_depth=args.depth,
        time_limit_seconds=args.time_limit,
        watch=args.watch,
        show_evaluation=args.evaluation,
        delay_seconds=args.delay,
        save_record=not args.no_save,
    )


if __name__ == "__main__":
    main()
