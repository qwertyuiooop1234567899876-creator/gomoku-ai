from __future__ import annotations

import unittest
from unittest.mock import patch

from engine.board import BLACK, WHITE, Board
from engine.search import SearchAI
from engine.search_types import RootResult, SearchConfig


class TestMainSearchNodeLimit(unittest.TestCase):
    def test_config_rejects_non_positive_limit(self) -> None:
        with self.assertRaises(ValueError):
            SearchConfig(main_search_node_limit=0)

    def test_iterative_search_preserves_last_completed_depth(self) -> None:
        board = Board()
        board.place(7, 7, WHITE)
        candidate = (7, 8)
        ai = SearchAI(
            BLACK,
            max_depth=3,
            time_limit_seconds=None,
            node_limit=1,
        )
        ai._begin_move_search()
        result = RootResult(
            move=candidate,
            score=100,
            principal_variation=(candidate,),
            ranked_moves=((candidate, 100),),
        )

        def bounded_root(*_args, **_kwargs) -> RootResult:
            ai._check_node_limit()
            ai._counters.nodes += 1
            return result

        with (
            patch.object(ai, "_search_root", side_effect=bounded_root),
            patch.object(ai, "_filter_proven_losing_candidates", side_effect=list),
            patch.object(ai, "_filter_root_vcf_candidates", side_effect=list),
            patch.object(ai, "_maybe_run_post_filter_defense_probe", side_effect=lambda _board, moves, **_kwargs: (moves, None)),
            patch.object(ai, "_quarantine_unproven_root_scores", side_effect=lambda _board, value, **_kwargs: value),
        ):
            outcome = ai._run_iterative_root_search(
                board,
                [candidate],
                fallback_move=candidate,
                preserve_frontier_order=False,
                allow_near_loss_expansion=False,
                defense_probe=None,
            )

        self.assertEqual(1, outcome.completed_depth)
        self.assertFalse(outcome.search_completed)
        self.assertEqual("node_limit", outcome.stop_reason)
        self.assertEqual(1, ai._counters.nodes)
        self.assertIsNone(ai._main_search_node_limit)


if __name__ == "__main__":
    unittest.main()
