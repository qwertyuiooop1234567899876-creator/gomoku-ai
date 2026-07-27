from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from engine.ai import ScoringAI
from engine.board import BLACK, WHITE, Board
from engine.evaluator import evaluate_board, render_evaluation_bar
from engine.game import format_move, other_player, player_name
from engine.records import GameRecorder, RecordPaths


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


def play_game(
    *,
    watch: bool = False,
    show_evaluation: bool = False,
    delay_seconds: float = 0.0,
    save_record: bool = True,
) -> GameResult:
    """让黑白两个 ScoringAI 自动完成一局并保存完整诊断。"""
    if delay_seconds < 0:
        raise ValueError("delay_seconds 不能小于 0。")

    board = Board()
    players = {
        BLACK: ScoringAI(player=BLACK, diagnostics=True, top_n=5),
        WHITE: ScoringAI(player=WHITE, diagnostics=True, top_n=5),
    }
    recorder = GameRecorder(
        mode="CVC",
        black_name="ScoringAI",
        white_name="ScoringAI",
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

        move_record = recorder.record_move(
            player=current_player,
            row=row,
            column=column,
            actor="ScoringAI",
            think_seconds=think_seconds,
            evaluation_before=evaluation_before,
            evaluation_after=evaluation_after,
            analysis=analysis,
        )

        reason = analysis["reason"] if analysis else "未记录"
        print(
            f"{move_record.number:3}. "
            f"{player_name(current_player)} "
            f"{format_move(row, column)} "
            f"思考 {think_seconds:.3f}s "
            f"原因：{reason}"
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
    print(result)
    print(f"总手数：{len(recorder.moves)}")
    print(f"总耗时：{duration_seconds:.3f}s")
    print()
    print(board)
    print()
    print(recorder.render_score_sheet(full=True))

    record_paths: RecordPaths | None = None

    if save_record:
        record_paths = recorder.save(
            board=board,
            result=result,
            duration_seconds=duration_seconds,
            prefix="scoring-vs-scoring",
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
        description="让 ScoringAI 执黑与 ScoringAI 执白自动对战。",
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
        watch=args.watch,
        show_evaluation=args.evaluation,
        delay_seconds=args.delay,
        save_record=not args.no_save,
    )


if __name__ == "__main__":
    main()
