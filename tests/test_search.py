import unittest

from engine.board import BLACK, WHITE, Board
from engine.search import SearchAI, SearchConfig


class TestSearchConfig(unittest.TestCase):
    def test_invalid_depth_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SearchConfig(max_depth=0)

    def test_invalid_time_limit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SearchConfig(time_limit_seconds=0)


class TestSearchAI(unittest.TestCase):
    def test_empty_board_plays_center(self) -> None:
        board = Board()
        ai = SearchAI(
            player=BLACK,
            max_depth=2,
            time_limit_seconds=None,
            diagnostics=True,
        )

        self.assertEqual((7, 7), ai.choose_move(board))
        self.assertIsNotNone(ai.last_analysis)
        assert ai.last_analysis is not None
        self.assertEqual("空棋盘选择天元", ai.last_analysis.reason)

    def test_search_completes_requested_depth_and_records_pv(self) -> None:
        board = Board()
        board.place(7, 7, BLACK)
        ai = SearchAI(
            player=WHITE,
            max_depth=2,
            time_limit_seconds=None,
            root_candidate_limit=6,
            branch_candidate_limit=4,
            threat_extension_depth=1,
            diagnostics=True,
            top_n=3,
        )

        move = ai.choose_move(board)

        self.assertTrue(board.is_empty(*move))
        self.assertIsNotNone(ai.last_analysis)
        assert ai.last_analysis is not None
        self.assertEqual(2, ai.last_analysis.search_depth)
        self.assertTrue(ai.last_analysis.search_completed)
        self.assertGreater(ai.last_analysis.nodes, 0)
        self.assertGreater(ai.last_analysis.cutoffs, 0)
        self.assertEqual(move, ai.last_analysis.principal_variation[0])
        self.assertLessEqual(len(ai.last_analysis.top_candidates), 3)

    def test_search_does_not_modify_board(self) -> None:
        board = Board()
        board.place(7, 7, BLACK)
        board.place(6, 7, WHITE)

        grid_before = [row.copy() for row in board.grid]
        history_before = board.move_history.copy()

        ai = SearchAI(
            player=BLACK,
            max_depth=2,
            time_limit_seconds=None,
            root_candidate_limit=6,
            branch_candidate_limit=4,
            threat_extension_depth=1,
        )
        ai.choose_move(board)

        self.assertEqual(grid_before, board.grid)
        self.assertEqual(history_before, board.move_history)

    def test_search_takes_immediate_win(self) -> None:
        board = Board()
        board.place(7, 4, BLACK)
        for column in range(5, 9):
            board.place(7, column, WHITE)

        ai = SearchAI(player=WHITE, diagnostics=True)

        self.assertEqual((7, 9), ai.choose_move(board))
        assert ai.last_analysis is not None
        self.assertEqual("立即五连", ai.last_analysis.reason)
        self.assertEqual(0, ai.last_analysis.search_depth)

    def test_search_blocks_unique_winning_point(self) -> None:
        board = Board()
        board.place(6, 4, WHITE)
        for column in range(5, 9):
            board.place(6, column, BLACK)

        ai = SearchAI(player=WHITE, diagnostics=True)

        self.assertEqual((6, 9), ai.choose_move(board))
        assert ai.last_analysis is not None
        self.assertEqual("封堵唯一胜点", ai.last_analysis.reason)

    def test_search_finds_counterattack_from_v062_loss(self) -> None:
        """搜索应在 V0.6.2 败局中找到 J9 的主动反击。"""
        board = Board()
        moves = (
            (7, 7, BLACK),   # H8
            (6, 7, WHITE),   # H7
            (6, 8, BLACK),   # I7
            (8, 6, WHITE),   # G9
            (7, 9, BLACK),   # J8
            (7, 8, WHITE),   # I8
            (8, 10, BLACK),  # K9
            (5, 7, WHITE),   # H6
            (9, 9, BLACK),   # J10
        )

        for row, column, player in moves:
            board.place(row, column, player)

        ai = SearchAI(
            player=WHITE,
            max_depth=2,
            time_limit_seconds=None,
            root_candidate_limit=12,
            branch_candidate_limit=8,
            threat_extension_depth=2,
            diagnostics=True,
        )

        self.assertEqual((8, 9), ai.choose_move(board))  # J9
        assert ai.last_analysis is not None
        self.assertGreaterEqual(ai.last_analysis.search_depth, 1)
        self.assertEqual(
            (8, 9),
            ai.last_analysis.principal_variation[0],
        )
        self.assertGreaterEqual(
            len(ai.last_analysis.principal_variation),
            2,
        )


if __name__ == "__main__":
    unittest.main()
