import unittest

from engine.board import BLACK, WHITE, Board
from engine.evaluator import (
    evaluate_board,
    evaluate_move,
    evaluate_player,
    score_to_percentage,
)


class TestEvaluator(unittest.TestCase):
    def test_empty_board_is_balanced(self) -> None:
        """空棋盘的黑白静态评分应当相同。"""
        board = Board()

        self.assertEqual(
            0,
            evaluate_board(board, WHITE),
        )
        self.assertEqual(
            50.0,
            score_to_percentage(0),
        )

    def test_five_scores_more_than_open_four(self) -> None:
        """五连的评分必须高于活四。"""
        five_board = Board()
        open_four_board = Board()

        for column in range(5, 10):
            five_board.place(
                7,
                column,
                WHITE,
            )

        for column in range(5, 9):
            open_four_board.place(
                7,
                column,
                WHITE,
            )

        five_score = evaluate_player(
            five_board,
            WHITE,
        )
        open_four_score = evaluate_player(
            open_four_board,
            WHITE,
        )

        self.assertGreater(
            five_score,
            open_four_score,
        )

    def test_white_advantage_is_positive(self) -> None:
        """白棋形成活三时，白棋视角评分应为正数。"""
        board = Board()

        board.place(7, 6, WHITE)
        board.place(7, 7, WHITE)
        board.place(7, 8, WHITE)

        self.assertGreater(
            evaluate_board(board, WHITE),
            0,
        )

    def test_move_evaluation_restores_board(self) -> None:
        """评价候选位置后，真实棋盘不能被改变。"""
        board = Board()
        board.place(7, 7, BLACK)

        grid_before = [
            row.copy()
            for row in board.grid
        ]
        history_before = board.move_history.copy()

        evaluate_move(
            board,
            7,
            8,
            WHITE,
        )

        self.assertEqual(
            grid_before,
            board.grid,
        )
        self.assertEqual(
            history_before,
            board.move_history,
        )

    def test_percentage_stays_in_range(self) -> None:
        """评分映射结果必须保持在 0～100。"""
        self.assertGreaterEqual(
            score_to_percentage(-1_000_000_000),
            0.0,
        )
        self.assertLessEqual(
            score_to_percentage(1_000_000_000),
            100.0,
        )
    
    def test_pattern_priority_in_all_directions(self) -> None:
        """横、竖及两种斜线都应识别活三、活四和五连。"""
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
                open_three_board = Board()
                open_four_board = Board()
                five_board = Board()

                center_row = 7
                center_column = 7

                for offset in (-1, 0, 1):
                    open_three_board.place(
                        center_row + offset * row_step,
                        center_column + offset * column_step,
                        WHITE,
                    )

                for offset in (-1, 0, 1, 2):
                    open_four_board.place(
                        center_row + offset * row_step,
                        center_column + offset * column_step,
                        WHITE,
                    )

                for offset in (-2, -1, 0, 1, 2):
                    five_board.place(
                        center_row + offset * row_step,
                        center_column + offset * column_step,
                        WHITE,
                    )

                open_three_score = evaluate_player(
                    open_three_board,
                    WHITE,
                )
                open_four_score = evaluate_player(
                    open_four_board,
                    WHITE,
                )
                five_score = evaluate_player(
                    five_board,
                    WHITE,
                )

                self.assertGreater(
                    open_three_score,
                    0,
                )
                self.assertGreater(
                    open_four_score,
                    open_three_score,
                )
                self.assertGreater(
                    five_score,
                    open_four_score,
                )



if __name__ == "__main__":
    unittest.main()