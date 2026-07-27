from __future__ import annotations

import time

from engine.ai import ScoringAI
from engine.board import BLACK, WHITE, Board
from engine.evaluator import evaluate_board, render_evaluation_bar
from engine.game import (
    format_move,
    other_player,
    parse_move,
    player_name,
)
from engine.records import GameRecorder, RecordPaths


def create_recorder() -> GameRecorder:
    return GameRecorder(
        mode="PVC",
        black_name="Human",
        white_name="ScoringAI",
    )


def save_and_report(
    recorder: GameRecorder,
    board: Board,
    result: str,
    game_started: float,
) -> RecordPaths | None:
    if not recorder.moves:
        return None

    paths = recorder.save(
        board=board,
        result=result,
        duration_seconds=time.perf_counter() - game_started,
        prefix="pvc",
    )
    print(f"TXT 棋谱：{paths.txt}")
    print(f"JSON 诊断：{paths.json}")
    return paths


def main() -> None:
    board = Board()
    computer = ScoringAI(
        player=WHITE,
        diagnostics=True,
        top_n=5,
    )
    recorder = create_recorder()
    current_player = BLACK
    game_started = time.perf_counter()

    print("Gomoku Engine V0.6.2")
    print("玩家执黑棋 X，电脑执白棋 O。")
    print("已加入实时着法表、自动棋谱和 AI 决策诊断。")
    print("输入 H 查看指令，U 悔棋，R 重开，M 棋谱，Q 退出。")

    while True:
        print()
        print(board)
        print()
        print(render_evaluation_bar(board, current_player))
        print()
        print(recorder.render_score_sheet(last_rounds=8))
        print()

        if current_player == BLACK:
            input_started = time.perf_counter()
            raw_move = input(
                f"{player_name(current_player)}，请输入落子位置："
            )
            human_think_seconds = time.perf_counter() - input_started
            command = raw_move.strip().upper()

            if command == "Q":
                recorder.add_event("quit", "玩家主动退出")
                print("游戏已退出。")
                save_and_report(
                    recorder,
                    board,
                    "玩家退出",
                    game_started,
                )
                break

            if command == "H":
                print("可用指令：")
                print("  H8  在 H8 落子")
                print("  U   悔棋（撤销玩家和电脑各一手）")
                print("  R   保存当前棋谱并重新开局")
                print("  M   显示完整着法记录")
                print("  Q   保存棋谱并退出游戏")
                continue

            if command == "M":
                print()
                print(recorder.render_score_sheet(full=True))
                continue

            if command == "R":
                recorder.add_event("restart", "玩家要求重新开局")
                paths = save_and_report(
                    recorder,
                    board,
                    "重新开局",
                    game_started,
                )
                if paths is not None:
                    print("当前对局已保存。")

                board = Board()
                computer = ScoringAI(
                    player=WHITE,
                    diagnostics=True,
                    top_n=5,
                )
                recorder = create_recorder()
                current_player = BLACK
                game_started = time.perf_counter()
                print("棋盘已清空，重新开局。")
                continue

            if command == "U":
                if len(board.move_history) < 2:
                    print("当前没有完整的一轮棋可以撤销。")
                    continue

                computer_move = board.undo()
                player_move = board.undo()

                if computer_move is None or player_move is None:
                    print("悔棋失败：落子历史不完整。")
                    continue

                recorder.undo_last_moves(2)

                computer_row, computer_column, _ = computer_move
                player_row, player_column, _ = player_move

                print(
                    "已悔棋：撤销玩家 "
                    f"{format_move(player_row, player_column)}，"
                    "以及电脑 "
                    f"{format_move(computer_row, computer_column)}。"
                )
                continue

            try:
                row, column = parse_move(
                    command,
                    board.size,
                )

                evaluation_before = evaluate_board(board, WHITE)
                board.place(
                    row,
                    column,
                    current_player,
                )
                evaluation_after = evaluate_board(board, WHITE)

                recorder.record_move(
                    player=current_player,
                    row=row,
                    column=column,
                    actor="Human",
                    think_seconds=human_think_seconds,
                    evaluation_before=evaluation_before,
                    evaluation_after=evaluation_after,
                )

            except ValueError as error:
                print(f"输入无效：{error}")
                continue

        else:
            evaluation_before = evaluate_board(board, WHITE)
            think_started = time.perf_counter()
            row, column = computer.choose_move(board)
            think_seconds = time.perf_counter() - think_started

            board.place(
                row,
                column,
                WHITE,
            )
            evaluation_after = evaluate_board(board, WHITE)
            analysis = (
                computer.last_analysis.to_dict()
                if computer.last_analysis is not None
                else None
            )

            recorder.record_move(
                player=WHITE,
                row=row,
                column=column,
                actor="ScoringAI",
                think_seconds=think_seconds,
                evaluation_before=evaluation_before,
                evaluation_after=evaluation_after,
                analysis=analysis,
            )

            print(
                f"电脑白棋 O 落子：{format_move(row, column)} "
                f"（{think_seconds:.3f}s）"
            )
            if analysis:
                print(
                    "电脑决策："
                    f"{analysis['reason']}；"
                    f"候选点 {analysis['candidate_count']} 个"
                )

        if board.check_win(row, column):
            print()
            print(board)
            print()
            print(render_evaluation_bar(board, current_player))
            print()
            result = f"{player_name(current_player)}获胜"
            print(f"{result}！")
            save_and_report(
                recorder,
                board,
                result,
                game_started,
            )
            break

        if board.is_full():
            print()
            print(board)
            print()
            print(render_evaluation_bar(board, current_player))
            print()
            print("棋盘已满，本局平局。")
            save_and_report(
                recorder,
                board,
                "平局",
                game_started,
            )
            break

        current_player = other_player(current_player)


if __name__ == "__main__":
    main()
