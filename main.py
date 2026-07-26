from engine.board import BLACK, WHITE, Board


def main() -> None:
    board = Board()

    board.place(7, 7, BLACK)
    board.place(7, 8, WHITE)
    board.place(8, 7, BLACK)

    print("Gomoku Engine V0.1")
    print()
    print(board)


if __name__ == "__main__":
    main()