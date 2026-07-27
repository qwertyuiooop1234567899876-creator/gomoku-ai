import json
import unittest
from pathlib import Path

from engine.board import BLACK, WHITE, Board
from engine.search import SearchAI
from engine.time_manager import TimeManager
from engine.zobrist import compute_grid_hash


class TestZobristBoard(unittest.TestCase):
    def test_incremental_hash_matches_full_rebuild(self) -> None:
        board = Board()
        sequence = (
            (7, 7, BLACK),
            (6, 7, WHITE),
            (8, 8, BLACK),
            (5, 6, WHITE),
        )
        for move in sequence:
            board.place(*move)
            self.assertEqual(
                compute_grid_hash(board.grid),
                board.zobrist_hash,
            )

    def test_undo_restores_hash_and_empty_count(self) -> None:
        board = Board()
        original_hash = board.zobrist_hash
        original_empty = board.empty_count
        board.place(7, 7, BLACK)
        board.place(6, 7, WHITE)
        board.undo()
        board.undo()
        self.assertEqual(original_hash, board.zobrist_hash)
        self.assertEqual(original_empty, board.empty_count)


class TestTimeManager(unittest.TestCase):
    def test_unlimited_manager_has_no_deadlines(self) -> None:
        manager = TimeManager.start(None)
        self.assertIsNone(manager.soft_deadline)
        self.assertIsNone(manager.hard_deadline)
        self.assertFalse(manager.soft_expired())
        self.assertFalse(manager.hard_expired())

    def test_sub_deadline_never_exceeds_hard_deadline(self) -> None:
        manager = TimeManager.start(1.0)
        deadline = manager.sub_deadline(0.9, maximum_seconds=5.0)
        self.assertIsNotNone(deadline)
        assert deadline is not None
        assert manager.hard_deadline is not None
        self.assertLessEqual(deadline, manager.hard_deadline)


class TestV080SearchDiagnostics(unittest.TestCase):
    def test_search_reports_v080_metrics(self) -> None:
        board = Board()
        board.place(7, 7, BLACK)
        ai = SearchAI(
            player=WHITE,
            max_depth=2,
            time_limit_seconds=None,
            root_candidate_limit=6,
            branch_candidate_limit=4,
            diagnostics=True,
        )
        ai.choose_move(board)
        analysis = ai.last_analysis
        self.assertIsNotNone(analysis)
        assert analysis is not None
        self.assertEqual(2, analysis.requested_depth)
        self.assertEqual(2, analysis.search_depth)
        self.assertGreater(analysis.nodes, 0)
        self.assertGreater(analysis.nps, 0)
        self.assertGreaterEqual(analysis.transposition_size, 0)

    def test_vcf_shortcut_is_reported(self) -> None:
        board = Board()
        for column in (5, 6, 7):
            board.place(7, column, WHITE)
        board.place(0, 0, BLACK)

        ai = SearchAI(
            player=WHITE,
            max_depth=3,
            time_limit_seconds=None,
            diagnostics=True,
        )
        move = ai.choose_move(board)
        self.assertIn(move, {(7, 4), (7, 8)})
        assert ai.last_analysis is not None
        self.assertTrue(ai.last_analysis.vcf_found)
        self.assertIn("VCF", ai.last_analysis.reason)


class TestRegressionPositions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = (
            Path(__file__).parent
            / "positions"
            / "regression_v080.json"
        )
        cls.positions = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _parse(coordinate: str) -> tuple[int, int]:
        return int(coordinate[1:]) - 1, ord(coordinate[0]) - ord("A")

    def _build_board(self, position: dict[str, object]) -> Board:
        board = Board()
        moves = position.get("moves")
        if isinstance(moves, list):
            for index, coordinate in enumerate(moves):
                row, column = self._parse(str(coordinate))
                board.place(
                    row,
                    column,
                    BLACK if index % 2 == 0 else WHITE,
                )
            return board

        stones = position.get("stones", {})
        assert isinstance(stones, dict)
        for player_name, player in (("BLACK", BLACK), ("WHITE", WHITE)):
            coordinates = stones.get(player_name, [])
            assert isinstance(coordinates, list)
            for coordinate in coordinates:
                row, column = self._parse(str(coordinate))
                board.place(row, column, player)
        return board

    def test_regression_catalog(self) -> None:
        for position in self.positions:
            with self.subTest(position=position["name"]):
                board = self._build_board(position)
                player = BLACK if position["player"] == "BLACK" else WHITE
                ai = SearchAI(
                    player=player,
                    max_depth=int(position.get("max_depth", 2)),
                    time_limit_seconds=None,
                    diagnostics=True,
                )
                move = ai.choose_move(board)
                expected = {
                    self._parse(item)
                    for item in position["expected_moves"]
                }
                self.assertIn(move, expected)
                expected_reason = position.get("expected_reason_contains")
                if expected_reason:
                    assert ai.last_analysis is not None
                    self.assertIn(
                        str(expected_reason),
                        ai.last_analysis.reason,
                    )


if __name__ == "__main__":
    unittest.main()
