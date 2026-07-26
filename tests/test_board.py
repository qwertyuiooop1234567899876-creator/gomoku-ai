import unittest

from engine.board import BLACK, EMPTY, WHITE, Board


class TestBoard(unittest.TestCase):
    def setUp(self) -> None:
        """每个测试开始前创建一个全新的棋盘。"""
        self.board = Board()

    def test_horizontal_win(self) -> None:
        """测试横向五连。"""
        for column in range(3, 8):
            self.board.place(7, column, BLACK)

        self.assertTrue(self.board.check_win(7, 7))

    def test_vertical_win(self) -> None:
        """测试纵向五连。"""
        for row in range(3, 8):
            self.board.place(row, 7, WHITE)

        self.assertTrue(self.board.check_win(7, 7))

    def test_downward_diagonal_win(self) -> None:
        """测试左上到右下的五连。"""
        for offset in range(5):
            self.board.place(
                3 + offset,
                4 + offset,
                BLACK,
            )

        self.assertTrue(self.board.check_win(7, 8))

    def test_upward_diagonal_win(self) -> None:
        """测试右上到左下的五连。"""
        for offset in range(5):
            self.board.place(
                3 + offset,
                10 - offset,
                WHITE,
            )

        self.assertTrue(self.board.check_win(7, 6))

    def test_four_stones_are_not_a_win(self) -> None:
        """连续四颗不能判定胜利。"""
        for column in range(4):
            self.board.place(7, column, BLACK)

        self.assertFalse(self.board.check_win(7, 3))

    def test_empty_position_is_not_a_win(self) -> None:
        """空位置不能判定胜利。"""
        self.assertFalse(self.board.check_win(7, 7))

    def test_duplicate_move_is_rejected(self) -> None:
        """同一位置不能重复落子。"""
        self.board.place(7, 7, BLACK)

        with self.assertRaises(ValueError):
            self.board.place(7, 7, WHITE)

    def test_undo_restores_empty_position(self) -> None:
        """悔棋后应恢复为空位。"""
        self.board.place(7, 7, BLACK)

        undone_move = self.board.undo()

        self.assertEqual((7, 7, BLACK), undone_move)
        self.assertEqual(EMPTY, self.board.grid[7][7])
        self.assertTrue(self.board.is_empty(7, 7))

    def test_full_board_detection(self) -> None:
        """棋盘全部填满后应返回 True。"""
        small_board = Board(size=5)

        self.assertFalse(small_board.is_full())

        for row in range(small_board.size):
            for column in range(small_board.size):
                small_board.place(
                    row,
                    column,
                    BLACK,
                )

        self.assertTrue(small_board.is_full())


if __name__ == "__main__":
    unittest.main()