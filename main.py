from engine.board import BLACK, WHITE, Board


def main() -> None:
    board = Board()

    board.place(7, 5, BLACK)
    board.place(7, 6, BLACK)
    board.place(7, 7, BLACK)
    board.place(7, 8, BLACK)
    board.place(7, 9, BLACK)

    board.place(8, 7, WHITE)

    print("Gomoku Engine V0.2")
    print()
    print(board)
    print()

    if board.check_win(7, 9):
        print("检测结果：黑棋获胜。")
    else:
        print("检测结果：尚未分出胜负。")


if __name__ == "__main__":
    main()