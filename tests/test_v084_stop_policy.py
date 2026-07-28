import unittest
from dataclasses import replace
from unittest.mock import patch

from engine.board import BLACK, WHITE, Board
from engine.search import MATE_SCORE, RootResult, SearchAI


class TestV084StrictStopPolicy(unittest.TestCase):
    def test_mate_like_pvs_score_does_not_stop_iterative_deepening(self) -> None:
        """受限 PVS 的近将杀分不能代替严格必杀证明。"""
        board = Board()
        board.place(7, 7, BLACK)
        candidate = (6, 7)

        ai = SearchAI(
            player=WHITE,
            max_depth=4,
            time_limit_seconds=None,
            diagnostics=True,
        )
        ai.config = replace(ai.config, use_aspiration=False)
        result = RootResult(
            move=candidate,
            score=MATE_SCORE - 3,
            principal_variation=(candidate,),
            ranked_moves=((candidate, MATE_SCORE - 3),),
        )

        with (
            patch.object(ai, "_timed_winning_moves", return_value=[]),
            patch.object(ai, "_root_profile_pool", return_value=[candidate]),
            patch.object(ai, "_profile_moves_timed", return_value={}),
            patch.object(ai, "_multi_threat_frontiers", return_value={}),
            patch.object(ai, "_ordered_moves", return_value=[candidate]),
            patch.object(ai, "_search_root", return_value=result) as search_root,
        ):
            selected = ai.choose_move(board)

        self.assertEqual(candidate, selected)
        self.assertEqual(4, search_root.call_count)
        self.assertIsNotNone(ai.last_analysis)
        assert ai.last_analysis is not None
        self.assertEqual(4, ai.last_analysis.search_depth)
        self.assertTrue(ai.last_analysis.search_completed)
        self.assertEqual(
            "requested_depth_completed",
            ai.last_analysis.stop_reason,
        )

    def test_explicit_immediate_win_keeps_fast_proven_shortcut(self) -> None:
        """真实一步五连仍应立即返回，并记录明确停止原因。"""
        board = Board()
        for column in range(3, 7):
            board.place(7, column, WHITE)

        ai = SearchAI(
            player=WHITE,
            max_depth=8,
            time_limit_seconds=60.0,
            diagnostics=True,
        )
        selected = ai.choose_move(board)

        board.place(*selected, WHITE)
        try:
            self.assertTrue(board.check_win(*selected))
        finally:
            board.undo()

        self.assertIsNotNone(ai.last_analysis)
        assert ai.last_analysis is not None
        self.assertEqual("immediate_win", ai.last_analysis.stop_reason)
        self.assertEqual(0, ai.last_analysis.search_depth)
        self.assertTrue(ai.last_analysis.search_completed)
        self.assertIsNotNone(ai.last_analysis.time_used_ratio)
        self.assertLess(ai.last_analysis.time_used_ratio or 1.0, 0.1)
        payload = ai.last_analysis.to_dict()
        self.assertEqual("immediate_win", payload["stop_reason"])
        self.assertIn("time_used_ratio", payload)


if __name__ == "__main__":
    unittest.main()
