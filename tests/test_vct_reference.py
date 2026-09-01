from __future__ import annotations

from pathlib import Path
import unittest

from engine.board import BLACK, WHITE, Board
from engine.game import parse_move
from engine.proof_search import ProofState
from tools.vct_reference import (
    VCTReferenceCase,
    build_board,
    load_case,
    run_reference,
)


POSITIONS = Path(__file__).resolve().parent / "positions"
FIXTURES = (
    POSITIONS / "v0175_reverse_move10_vct.json",
    POSITIONS / "v0175_selfplay_move24_vct.json",
    POSITIONS / "v0175_yixin_move21_vct.json",
)


class TestVCTReference(unittest.TestCase):
    def test_v0175_fixtures_preserve_ordered_history(self) -> None:
        expected = (
            (WHITE, ("G10", "K6")),
            (WHITE, ("J10", "J6")),
            (BLACK, ("I8", "H11")),
        )
        for path, (player, candidates) in zip(FIXTURES, expected):
            with self.subTest(path=path.name):
                case = load_case(path)
                board = build_board(case)
                self.assertEqual(player, case.player)
                self.assertEqual(candidates, case.candidates)
                self.assertEqual(len(case.history), len(board.move_history))
                self.assertEqual(case.expected_hash, board.zobrist_hash)

    def test_zero_node_budget_remains_unknown_for_every_fixture(self) -> None:
        for path in FIXTURES:
            with self.subTest(path=path.name):
                case = load_case(path)
                run = run_reference(
                    case,
                    seconds_per_candidate=0.1,
                    max_nodes=0,
                )
                self.assertTrue(run.candidates)
                for candidate in run.candidates:
                    self.assertEqual(
                        ProofState.UNKNOWN.value,
                        candidate.attacker_state,
                    )
                    self.assertFalse(candidate.completed)
                    self.assertEqual("node_limit", candidate.cutoff_reason)

    def test_open_four_witness_is_a_completed_attacker_win(self) -> None:
        history = ("H8", "A15", "I8", "B15", "J8", "C15", "K8")
        board = Board(15)
        for index, coordinate in enumerate(history):
            board.place(
                *parse_move(coordinate, board.size),
                BLACK if index % 2 == 0 else WHITE,
            )
        case = VCTReferenceCase(
            name="open_four_reference",
            board_size=15,
            player=WHITE,
            history=history,
            candidates=("A1",),
            expected_hash=board.zobrist_hash,
        )

        run = run_reference(
            case,
            seconds_per_candidate=1.0,
            max_nodes=1_000,
            max_quiet_frontiers=0,
            max_quiet_attacker_moves=0,
        )

        result = run.candidates[0]
        self.assertEqual(ProofState.PROVEN_WIN.value, result.attacker_state)
        self.assertTrue(result.completed)
        self.assertIsNone(result.cutoff_reason)
        self.assertIn(result.best_coordinate, {"G8", "L8"})

    def test_candidate_filter_cannot_invent_a_fixture_move(self) -> None:
        case = load_case(FIXTURES[0])
        with self.assertRaisesRegex(ValueError, "必须来自夹具"):
            run_reference(
                case,
                coordinates=("A1",),
                seconds_per_candidate=0.1,
                max_nodes=0,
            )


if __name__ == "__main__":
    unittest.main()
