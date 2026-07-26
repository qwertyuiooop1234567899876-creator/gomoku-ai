import unittest

from engine.ai import (
    RandomAI,
    ScoringAI,
    TacticalAI,
)
from engine.board import BLACK, WHITE, Board


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

class TestTacticalAI(unittest.TestCase):
    def setUp(self) -> None:
        self.board = Board()
        self.ai = TacticalAI(player=WHITE)

    def test_ai_plays_center_on_empty_board(self) -> None:
        """空棋盘时，电脑应选择天元。"""
        self.assertEqual(
            (7, 7),
            self.ai.choose_move(self.board),
        )

    def test_ai_takes_immediate_winning_move(self) -> None:
        """电脑自己能形成五连时，应立即获胜。"""
        self.board.place(7, 4, BLACK)

        for column in range(5, 9):
            self.board.place(
                7,
                column,
                WHITE,
            )

        self.assertEqual(
            (7, 9),
            self.ai.choose_move(self.board),
        )

    def test_ai_blocks_opponent_winning_move(self) -> None:
        """玩家下一步能五连时，电脑应立即封堵。"""
        self.board.place(6, 4, WHITE)

        for column in range(5, 9):
            self.board.place(
                6,
                column,
                BLACK,
            )

        self.assertEqual(
            (6, 9),
            self.ai.choose_move(self.board),
        )

    def test_ai_analysis_does_not_modify_board(self) -> None:
        """AI 思考过程中不应改变真实棋盘。"""
        self.board.place(7, 7, BLACK)

        grid_before = [
            row.copy()
            for row in self.board.grid
        ]
        history_before = self.board.move_history.copy()

        self.ai.choose_move(self.board)

        self.assertEqual(
            grid_before,
            self.board.grid,
        )
        self.assertEqual(
            history_before,
            self.board.move_history,
        )

class TestScoringAI(unittest.TestCase):
    def setUp(self) -> None:
        self.board = Board()
        self.ai = ScoringAI(player=WHITE)

    def test_ai_extends_open_three_in_all_directions(self) -> None:
        """电脑应在横、竖及两种斜线方向发展自己的活三。"""
        directions = {
            "horizontal": (0, 1),
            "vertical": (1, 0),
            "downward_diagonal": (1, 1),
            "upward_diagonal": (1, -1),
        }

        for direction_name, (
            row_step,
            column_step,
        ) in directions.items():
            with self.subTest(direction=direction_name):
                board = Board()

                center_row = 7
                center_column = 7

                for offset in (-1, 0, 1):
                    board.place(
                        center_row + offset * row_step,
                        center_column + offset * column_step,
                        WHITE,
                    )

                move = self.ai.choose_move(board)

                expected_moves = {
                    (
                        center_row - 2 * row_step,
                        center_column - 2 * column_step,
                    ),
                    (
                        center_row + 2 * row_step,
                        center_column + 2 * column_step,
                    ),
                }

                self.assertIn(
                    move,
                    expected_moves,
                )

    def test_ai_blocks_open_three_in_all_directions(self) -> None:
        """电脑应在横、竖及两种斜线方向限制玩家活三。"""
        directions = {
            "horizontal": (0, 1),
            "vertical": (1, 0),
            "downward_diagonal": (1, 1),
            "upward_diagonal": (1, -1),
        }

        for direction_name, (
            row_step,
            column_step,
        ) in directions.items():
            with self.subTest(direction=direction_name):
                board = Board()

                center_row = 7
                center_column = 7

                for offset in (-1, 0, 1):
                    board.place(
                        center_row + offset * row_step,
                        center_column + offset * column_step,
                        BLACK,
                    )

                move = self.ai.choose_move(board)

                expected_moves = {
                    (
                        center_row - 2 * row_step,
                        center_column - 2 * column_step,
                    ),
                    (
                        center_row + 2 * row_step,
                        center_column + 2 * column_step,
                    ),
                }

                self.assertIn(
                    move,
                    expected_moves,
                )

if __name__ == "__main__":
    unittest.main()