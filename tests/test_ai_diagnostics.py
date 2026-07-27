import unittest

from engine.ai import ScoringAI
from engine.board import BLACK, WHITE, Board


class TestScoringAIDiagnostics(unittest.TestCase):
    def test_empty_board_records_opening_reason(self) -> None:
        board = Board()
        ai = ScoringAI(player=BLACK, diagnostics=True)

        move = ai.choose_move(board)

        self.assertEqual((7, 7), move)
        self.assertIsNotNone(ai.last_analysis)
        assert ai.last_analysis is not None
        self.assertEqual("空棋盘选择天元", ai.last_analysis.reason)
        self.assertEqual(225, ai.last_analysis.candidate_count)

    def test_normal_choice_keeps_ranked_candidates(self) -> None:
        board = Board()
        board.place(7, 7, BLACK)
        ai = ScoringAI(player=WHITE, diagnostics=True, top_n=3)

        ai.choose_move(board)

        self.assertIsNotNone(ai.last_analysis)
        assert ai.last_analysis is not None
        self.assertGreater(len(ai.last_analysis.top_candidates), 0)
        self.assertLessEqual(len(ai.last_analysis.top_candidates), 3)


if __name__ == "__main__":
    unittest.main()
