import unittest

from engine.board import BLACK, WHITE, Board
from engine.game import format_move, parse_move
from engine.search import DefenseProof, SearchAI


class TestV085DefenseVCTProbe(unittest.TestCase):
    @staticmethod
    def _build_position() -> Board:
        board = Board()
        player = BLACK
        for coordinate in (
            "H8", "I7",
            "I6", "H7",
            "G7", "F6",
            "J7", "K8",
            "J8", "J9",
            "I8", "I10",
        ):
            board.place(*parse_move(coordinate, board.size), player)
            player = WHITE if player == BLACK else BLACK
        return board

    def test_h11_is_preferred_over_l7_by_defense_vct_probe(self) -> None:
        """两个封堵点表面接近时，窄分支威胁探针应选择 H11。"""
        board = self._build_position()
        ai = SearchAI(
            player=BLACK,
            max_depth=3,
            time_limit_seconds=None,
            diagnostics=True,
        )

        selected = ai.choose_move(board)

        self.assertEqual("H11", format_move(*selected))
        self.assertIsNotNone(ai.last_analysis)
        assert ai.last_analysis is not None
        self.assertTrue(ai.last_analysis.defense_vct_checked)
        self.assertEqual(
            "H11",
            format_move(*ai.last_analysis.defense_vct_best_move),
        )
        self.assertIn("防守分支 VCT", ai.last_analysis.reason)
        candidates = {
            format_move(*candidate.move): candidate
            for candidate in ai.last_analysis.defense_vct_candidates
        }
        self.assertIn("L7", candidates)
        self.assertGreater(candidates["H11"].score, candidates["L7"].score)
        self.assertEqual(
            DefenseProof.SURVIVES_PROBE.value,
            candidates["H11"].status,
        )

    def test_defense_probe_restores_board_hash_and_history(self) -> None:
        """防守探针的递归模拟不能污染真实棋盘。"""
        board = self._build_position()
        before_grid = tuple(tuple(row) for row in board.grid)
        before_history = tuple(board.move_history)
        before_hash = board.zobrist_hash
        ai = SearchAI(
            player=BLACK,
            max_depth=3,
            time_limit_seconds=None,
            diagnostics=True,
        )

        ai.choose_move(board)

        self.assertEqual(before_grid, tuple(tuple(row) for row in board.grid))
        self.assertEqual(before_history, tuple(board.move_history))
        self.assertEqual(before_hash, board.zobrist_hash)

    def test_defense_probe_diagnostics_are_serializable(self) -> None:
        board = self._build_position()
        ai = SearchAI(
            player=BLACK,
            max_depth=3,
            time_limit_seconds=None,
            diagnostics=True,
        )
        ai.choose_move(board)
        assert ai.last_analysis is not None

        payload = ai.last_analysis.to_dict()

        self.assertTrue(payload["defense_vct_checked"])
        self.assertEqual("H11", payload["defense_vct_best_coordinate"])
        self.assertGreater(payload["defense_vct_nodes"], 0)
        self.assertEqual(
            "H11",
            payload["defense_vct_candidates"][0]["coordinate"],
        )


if __name__ == "__main__":
    unittest.main()
