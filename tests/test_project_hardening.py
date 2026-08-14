from __future__ import annotations

import math
import unittest

from engine.board import BLACK, Board
from engine.evaluator import analyze_move_threats, find_winning_moves
from engine.native_core import native_core
from engine.search import SearchAI
from engine.time_manager import TimeManager


class TestFiniteTimeBudgets(unittest.TestCase):
    def test_search_rejects_nan_time_limit(self) -> None:
        with self.assertRaises(ValueError):
            SearchAI(BLACK, time_limit_seconds=math.nan)

    def test_time_manager_rejects_nonfinite_sub_budget(self) -> None:
        manager = TimeManager.start(1.0)

        with self.assertRaises(ValueError):
            manager.sub_deadline(math.nan)
        with self.assertRaises(ValueError):
            manager.sub_deadline(0.5, minimum_seconds=math.inf)
        with self.assertRaises(ValueError):
            manager.sub_deadline(0.5, maximum_seconds=-1.0)


class TestNativeFallbackBoundaries(unittest.TestCase):
    def test_large_python_board_bypasses_fixed_size_native_abi(self) -> None:
        board = Board(size=26)
        for column in range(4):
            board.place(12, column, BLACK)

        self.assertIsNone(native_core.find_winning_moves(board, BLACK))
        self.assertEqual([(12, 4)], find_winning_moves(board, BLACK))
        profile = analyze_move_threats(board, 12, 4, BLACK)
        self.assertTrue(profile.immediate_win)


if __name__ == "__main__":
    unittest.main()
