import unittest

from engine.ai import RandomAI
from engine.board import BLACK, Board


class TestRandomAI(unittest.TestCase):
    def setUp(self) -> None:
        self.board = Board()
        self.ai = RandomAI()

    def test_ai_returns_legal_move(self) -> None:
        """电脑应选择一个棋盘内的空位。"""
        row, column = self.ai.choose_move(self.board)

        self.assertTrue(
            self.board.is_inside(row, column)
        )
        self.assertTrue(
            self.board.is_empty(row, column)
        )

    def test_ai_avoids_occupied_position(self) -> None:
        """电脑不能选择已有棋子的位置。"""
        self.board.place(7, 7, BLACK)

        for _ in range(100):
            row, column = self.ai.choose_move(self.board)

            self.assertNotEqual(
                (7, 7),
                (row, column),
            )

    def test_ai_raises_error_on_full_board(self) -> None:
        """棋盘填满后，电脑应报告无法落子。"""
        small_board = Board(size=5)

        for row in range(small_board.size):
            for column in range(small_board.size):
                small_board.place(
                    row,
                    column,
                    BLACK,
                )

        with self.assertRaises(ValueError):
            self.ai.choose_move(small_board)


if __name__ == "__main__":
    unittest.main()