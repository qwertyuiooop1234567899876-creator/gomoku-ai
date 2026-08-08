from __future__ import annotations

import random
import time
import unittest

from engine.board import BLACK, WHITE, Board
from engine.evaluator import (
    _analyze_hypothetical_move_python,
    _find_winning_moves_python,
)
from engine.game import parse_move
from engine.native_core import ABI_VERSION, native_core, native_core_status
from engine.proof_search import ProofBudget, ProofSearch, ProofState, ProofTable
from engine.threats import ThreatAnalyzer
from engine.vcf import validate_vcf_certificate


BLACK_FIFTEEN_PREFIX = """
H8 H9 G7 I9 G9 I7 I8 G8 I10 J8 H6 K7 H10 K9
""".split()

BLACK_FIFTY_ONE_PREFIX = """
H8 I7 G7 I9 I8 J8 H6 H10 G11 I5 H7 H5 J9 G6 F7 G5 F5 F6
E7 D7 K10 L11 F8 E8 F9 E9 F10 F11 G8 I6 K7 K8 J7 E6 E10 D11
G10 G9 K6 H9 K5 D10 D8 D12 D13 I4 I3 J4 D6 K4
""".split()


def build_board(coordinates: list[str]) -> Board:
    board = Board()
    for index, coordinate in enumerate(coordinates):
        board.place(
            *parse_move(coordinate, board.size),
            BLACK if index % 2 == 0 else WHITE,
        )
    return board


def board_state(board: Board) -> tuple[object, ...]:
    return (
        tuple(tuple(row) for row in board.grid),
        tuple(board.move_history),
        board.zobrist_hash,
        board.empty_count,
    )


@unittest.skipUnless(native_core.available, "NativeCore尚未编译")
class TestNativeKernelEquivalence(unittest.TestCase):
    def test_native_abi_is_loaded(self) -> None:
        status = native_core_status()
        self.assertTrue(status["available"])
        self.assertEqual(ABI_VERSION, status["abi_version"])

    def test_randomized_profiles_and_wins_match_python_reference(self) -> None:
        generator = random.Random(20260801)
        for _ in range(80):
            board = Board()
            legal = board.get_legal_moves()
            generator.shuffle(legal)
            for ply, move in enumerate(legal[: generator.randrange(0, 70)]):
                board.place(*move, BLACK if ply % 2 == 0 else WHITE)

            for player in (BLACK, WHITE):
                self.assertEqual(
                    _find_winning_moves_python(board, player),
                    native_core.find_winning_moves(board, player),
                )
                probes = board.get_legal_moves()
                generator.shuffle(probes)
                probes = probes[: min(8, len(probes))]
                native_profiles = native_core.analyze_moves(
                    board,
                    probes,
                    player,
                )
                assert native_profiles is not None
                for move, native in zip(probes, native_profiles, strict=True):
                    reference = _analyze_hypothetical_move_python(
                        board,
                        *move,
                        player,
                    )
                    self.assertEqual(
                        (
                            reference.immediate_win,
                            reference.open_four_directions,
                            reference.four_directions,
                            reference.open_three_directions,
                            reference.winning_moves,
                        ),
                        (
                            native.immediate_win,
                            native.open_four_directions,
                            native.four_directions,
                            native.open_three_directions,
                            native.winning_moves,
                        ),
                    )
                native_support = native_core.counter_support_mask(
                    board,
                    probes,
                    player,
                    minimum=3,
                )
                assert native_support is not None
                self.assertEqual(
                    tuple(
                        ThreatAnalyzer._could_create_immediate_counter(
                            board,
                            move,
                            player,
                        )
                        for move in probes
                    ),
                    native_support,
                )

    def test_native_vcf_is_only_accepted_after_python_replay(self) -> None:
        board = Board()
        for column in (5, 6, 8):
            board.place(7, column, WHITE)
        before = board_state(board)
        result = native_core.find_vcf(
            board,
            WHITE,
            5,
            max_nodes=1_000,
            timeout_seconds=1.0,
            candidate_limit=16,
        )
        assert result is not None
        self.assertTrue(result.found)
        self.assertTrue(validate_vcf_certificate(board, WHITE, result.line))
        self.assertFalse(
            validate_vcf_certificate(board, WHITE, ((0, 0),))
        )
        self.assertEqual(before, board_state(board))


@unittest.skipUnless(native_core.available, "NativeCore尚未编译")
class TestV0140NativeProofRegressions(unittest.TestCase):
    def _prove_after_black_move(
        self,
        prefix: list[str],
        coordinate: str,
    ):
        board = build_board(prefix)
        before = board_state(board)
        proof = ProofSearch(
            budget=ProofBudget.from_now(
                8.0,
                max_nodes=20_000,
                max_attacker_moves=10,
                max_quiet_frontiers=16,
                max_quiet_attacker_moves=1,
                use_vcf_oracle=True,
                clock=time.perf_counter,
            ),
            analyzer=ThreatAnalyzer(
                candidate_limit=16,
                frontier_scan_limit=24,
            ),
            table=ProofTable(),
            clock=time.perf_counter,
        ).search_after_move(
            board,
            move=parse_move(coordinate, board.size),
            mover=BLACK,
            attacker=WHITE,
            side_to_move=WHITE,
        )
        self.assertIs(ProofState.PROVEN_WIN, proof.state)
        self.assertTrue(proof.completed)
        self.assertLess(proof.elapsed_seconds, 8.0)
        self.assertEqual(before, board_state(board))
        return proof

    def test_black_fifteen_l10_finishes_inside_final_audit_budget(self) -> None:
        proof = self._prove_after_black_move(BLACK_FIFTEEN_PREFIX, "L10")
        expected_prefix = tuple(
            parse_move(move, 15)
            for move in ("J7", "H7", "J9", "L9", "J10")
        )
        self.assertEqual(expected_prefix, proof.principal_variation[:5])

    def test_black_fifty_one_h4_finishes_inside_final_audit_budget(self) -> None:
        proof = self._prove_after_black_move(BLACK_FIFTY_ONE_PREFIX, "H4")
        expected_prefix = tuple(
            parse_move(move, 15)
            for move in ("J3", "K2", "J5", "J1", "L5", "I2")
        )
        self.assertEqual(expected_prefix, proof.principal_variation[:6])


if __name__ == "__main__":
    unittest.main()
