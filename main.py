from engine.ai import ScoringAI
from engine.board import BLACK, WHITE, Board
from engine.evaluator import render_evaluation_bar
from engine.game import (
    format_move,
    other_player,
    parse_move,
    player_name,
)


def main() -> None:
    board = Board()
    computer = ScoringAI(player=WHITE)
    current_player = BLACK

    print("Gomoku Engine V0.6.1")
    print("玩家执黑棋 X，电脑执白棋 O。")
    print("电脑使用复合威胁识别与棋型评分选择落点。")
    print("输入 H 查看指令，U 悔棋，R 重开，Q 退出。")

    while True:
        print()
        print(board)
        print()
        print(render_evaluation_bar(board, current_player))
        print()

        if current_player == BLACK:
            raw_move = input(
                f"{player_name(current_player)}，请输入落子位置："
            )

            command = raw_move.strip().upper()

            if command == "Q":
                print("游戏已退出。")
                break

            if command == "H":
                print("可用指令：")
                print("  H8  在 H8 落子")
                print("  U   悔棋")
                print("  R   重新开局")
                print("  Q   退出游戏")
                continue

            if command == "R":
                board = Board()
                current_player = BLACK
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

                board.place(
                    row,
                    column,
                    current_player,
                )

            except ValueError as error:
                print(f"输入无效：{error}")
                continue

        else:
            row, column = computer.choose_move(board)

            board.place(
                row,
                column,
                WHITE,
            )

            print(
                f"电脑白棋 O 落子："
                f"{format_move(row, column)}"
            )

        if board.check_win(row, column):
            print()
            print(board)
            print()
            print(render_evaluation_bar(board, current_player))
            print()
            print(f"{player_name(current_player)}获胜！")
            break

        if board.is_full():
            print()
            print(board)
            print()
            print(render_evaluation_bar(board, current_player))
            print()
            print("棋盘已满，本局平局。")
            break

        current_player = other_player(current_player)


if __name__ == "__main__":
    main()