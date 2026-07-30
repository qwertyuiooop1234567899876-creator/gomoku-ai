import unittest

from engine.board import BLACK, WHITE, Board
from engine.evaluator import (
    PositionEvaluation,
    analyze_move_threats,
    evaluate_board,
    evaluate_move,
    evaluate_player,
    find_winning_moves,
    render_evaluation_bar,
    score_to_percentage,
    yixin_score_to_percentage,
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

    def test_yixin_display_mapping_has_decisive_bounds(self) -> None:
        """YiXin 的 ±10000 必须显示为已证明的两端结果。"""
        self.assertEqual(0.0, yixin_score_to_percentage(-10_000))
        self.assertEqual(50.0, yixin_score_to_percentage(0))
        self.assertEqual(100.0, yixin_score_to_percentage(10_000))
        self.assertLess(yixin_score_to_percentage(-115), 50.0)
        self.assertGreater(yixin_score_to_percentage(115), 50.0)

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

    def test_double_three_profile_counts_two_directions(self) -> None:
        """同一落点同时形成横向和斜向活三时，应识别为双活三。"""
        board = Board()

        for move in (
            (7, 6),
            (7, 8),
            (6, 6),
            (8, 8),
        ):
            board.place(*move, BLACK)

        profile = analyze_move_threats(
            board,
            7,
            7,
            BLACK,
        )

        self.assertTrue(profile.double_three)
        self.assertEqual(2, profile.open_three_directions)
        self.assertEqual("双活三", profile.label)

    def test_open_four_returns_both_winning_points(self) -> None:
        """两端开放的四连必须返回两个一步胜点。"""
        board = Board()

        for column in range(4, 8):
            board.place(7, column, BLACK)

        self.assertEqual(
            [(7, 3), (7, 8)],
            find_winning_moves(board, BLACK),
        )

    def test_threat_analysis_restores_board(self) -> None:
        """复合威胁分析不得污染真实棋盘和落子历史。"""
        board = Board()
        board.place(7, 6, BLACK)
        board.place(6, 6, BLACK)

        grid_before = [row.copy() for row in board.grid]
        history_before = board.move_history.copy()

        analyze_move_threats(board, 7, 7, BLACK)

        self.assertEqual(grid_before, board.grid)
        self.assertEqual(history_before, board.move_history)

    def test_evaluation_bar_respects_side_to_move(self) -> None:
        """当前方存在立即胜点时，评分条应优先显示一步取胜。"""
        board = Board()

        for column in range(4, 8):
            board.place(7, column, BLACK)

        bar = render_evaluation_bar(
            board,
            current_player=BLACK,
        )

        self.assertIn("黑棋一步取胜", bar)
        self.assertIn("2 个胜点", bar)

    def test_evaluation_bar_uses_yixin_score_and_metadata(self) -> None:
        """外部评价存在时，不再使用静态棋型分生成评分条。"""
        board = Board()
        board.place(7, 7, BLACK)

        bar = render_evaluation_bar(
            board,
            current_player=WHITE,
            position_evaluation=PositionEvaluation(
                source="YiXin",
                score_white=-115,
                raw_score=115,
                depth=13,
                selective_depth=27,
                elapsed_seconds=2.0,
            ),
        )

        self.assertIn("YiXin：黑棋 +115", bar)
        self.assertIn("深度 13-27", bar)
        self.assertIn("2.000s", bar)
        self.assertIn("形势条非胜率", bar)

    def test_unavailable_yixin_does_not_fall_back_to_static_score(
        self,
    ) -> None:
        board = Board()
        board.place(7, 7, BLACK)

        bar = render_evaluation_bar(
            board,
            current_player=WHITE,
            position_evaluation=PositionEvaluation(
                source="YiXin",
                score_white=None,
                error="核心未启动",
            ),
        )

        self.assertIn("YiXin评价不可用：核心未启动", bar)
        self.assertIn("黑 X   -- ", bar)
        self.assertIn("  --  白 O", bar)

    def test_mirrored_broken_patterns_score_equally(self) -> None:
        """断点棋型及其镜像必须获得相同静态分数。"""
        left_gap = Board()
        right_gap = Board()

        for column in (5, 6, 8):
            left_gap.place(7, column, WHITE)

        for column in (6, 8, 9):
            right_gap.place(7, column, WHITE)

        self.assertEqual(
            evaluate_player(left_gap, WHITE),
            evaluate_player(right_gap, WHITE),
        )



    def test_single_winning_point_uses_post_block_evaluation(self) -> None:
        """只有一个胜点时，应显示唯一应手并评价封堵后的局面。"""
        board = Board()

        # 左端已被白棋封住，黑棋只剩右端一个立即胜点。
        board.place(7, 3, WHITE)
        for column in range(4, 8):
            board.place(7, column, BLACK)

        grid_before = [row.copy() for row in board.grid]
        history_before = board.move_history.copy()

        result = render_evaluation_bar(
            board,
            current_player=WHITE,
        )

        self.assertIn("白棋唯一应手：I8", result)
        self.assertIn("以下为封堵后评价", result)
        self.assertNotIn("黑 X 100.0%", result)

        # 评分条的临时试算不能改变真实棋盘。
        self.assertEqual(grid_before, board.grid)
        self.assertEqual(history_before, board.move_history)

if __name__ == "__main__":
    unittest.main()
