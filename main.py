from engine.board import BLACK, Board
from engine.game import other_player, parse_move, player_name


def main() -> None:
    board = Board()
    current_player = BLACK

    print("Gomoku Engine V0.3")
    print("输入棋盘坐标落子，例如 H8。")
    print("输入 Q 可以退出游戏。")

    while True:
        print()
        print(board)
        print()

        raw_move = input(
            f"{player_name(current_player)}，请输入落子位置："
        )

        if raw_move.strip().upper() == "Q":
            print("游戏已退出。")
            break

        try:
            row, column = parse_move(
                raw_move,
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

        if board.check_win(row, column):
            print()
            print(board)
            print()
            print(f"{player_name(current_player)}获胜！")
            break

        if board.is_full():
            print()
            print(board)
            print()
            print("棋盘已满，本局平局。")
            break

        current_player = other_player(current_player)


if __name__ == "__main__":
    main()