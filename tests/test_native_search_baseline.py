from __future__ import annotations

import unittest

from engine.game import parse_move
from tools import native_search_baseline


class TestNativeSearchBaseline(unittest.TestCase):
    def test_fixture_preserves_ordered_move_13_position(self) -> None:
        case = native_search_baseline.load_case()
        board = native_search_baseline.build_board(case)

        self.assertEqual(12, len(board.move_history))
        self.assertEqual(case.expected_hash, board.zobrist_hash)
        self.assertEqual(
            (7, 7, 1),
            board.move_history[0],
        )
        self.assertEqual(
            (9, 8, 2),
            board.move_history[-1],
        )
        self.assertTrue(board.is_empty(*parse_move("F7", board.size)))
        self.assertTrue(board.is_empty(*parse_move("J11", board.size)))

    def test_depth_one_full_window_is_reproducible(self) -> None:
        case = native_search_baseline.load_case()
        first = native_search_baseline.run_full_window_candidate(
            case,
            "F7",
            1,
        )
        second = native_search_baseline.run_full_window_candidate(
            case,
            "F7",
            1,
        )

        self.assertEqual("F7", first.selected_move)
        self.assertEqual(first.score, second.score)
        self.assertEqual(first.nodes, second.nodes)
        self.assertEqual(first.tt_digest, second.tt_digest)

    def test_iterative_pair_obeys_fixed_node_limit(self) -> None:
        case = native_search_baseline.load_case()
        run = native_search_baseline.run_iterative_pair(
            case,
            3,
            node_limit=1,
        )

        self.assertFalse(run.completed)
        self.assertEqual("node_limit", run.stop_reason)
        self.assertEqual(1, run.nodes)


if __name__ == "__main__":
    unittest.main()
