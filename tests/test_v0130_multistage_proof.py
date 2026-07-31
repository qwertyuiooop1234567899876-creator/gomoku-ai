from __future__ import annotations

import time
import unittest
from dataclasses import replace

from engine.ai import ProofCandidateAnalysis, RootVCFCandidateAnalysis
from engine.board import BLACK, WHITE, Board
from engine.evaluator import find_winning_moves
from engine.game import parse_move
from engine.proof_search import (
    ProofBudget,
    ProofSearch,
    ProofState,
    ProofTable,
)
from engine.root_safety import RootCandidateSafety
from engine.search import SearchAI
from engine.search_types import RootResult, RootVCFScanResult
from engine.threats import ThreatAnalyzer


BLACK_EIGHTY_FIVE_PREFIX = """
H8 I7 G7 I9 I8 J8 H10 H6 K9 E8 H9 H11 G5 G10 F9 I12
G6 G8 G4 G3 H7 J9 I10 E4 J10 J7 I11 H12 K10 L10 K7 K8
J12 G9 L8 M7 I5 J6 J5 H5 K13 L14 K12 K11 I6 G12 G11 F12
E12 J13 K14 M9 M8 L9 L13 J11 M12 N11 M10 L11 M11 N12 L7 K6
N10 L12 I4 H4 M13 M14 I3 I2 N13 O13 K4 L3 K5 N8 O7 H2 H3 G2
L5 M5
""".split()

BLACK_TWENTY_THREE_PREFIX = """
H8 H9 G7 I9 G9 I7 I8 G8 F7 J8 E7 D7 H10 F8 K9 D8 D6 E8 C8 D11
F9 E11
""".split()

WHITE_TWENTY_SIX_PREFIX = """
H8 I7 G7 I9 I8 G8 F6 J8 H10 H9 K9 I10 F7 F8 E7 D7 G5 D8 F9 E8
C8 J11 K12 D9 D6
""".split()

BLACK_EIGHTY_SEVEN_PREFIX = [
    *BLACK_EIGHTY_FIVE_PREFIX,
    "L6",
    "L4",
]


def build_black_eighty_five_board() -> Board:
    return build_board(BLACK_EIGHTY_FIVE_PREFIX)


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


class TestV0130CertificateAwareProof(unittest.TestCase):
    def test_black_eighty_five_losing_exchange_is_strictly_proven(self) -> None:
        board = build_black_eighty_five_board()
        losing_move = parse_move("L6", board.size)
        forced_block = parse_move("L4", board.size)
        before = board_state(board)

        board.place(*losing_move, BLACK)
        try:
            self.assertEqual(
                (forced_block,),
                tuple(find_winning_moves(board, BLACK)),
            )
        finally:
            board.undo()

        proof = ProofSearch(
            budget=ProofBudget(
                max_nodes=2_000,
                max_attacker_moves=10,
                max_quiet_frontiers=16,
                max_quiet_attacker_moves=1,
                use_vcf_oracle=True,
            ),
            analyzer=ThreatAnalyzer(
                candidate_limit=16,
                frontier_scan_limit=24,
            ),
            table=ProofTable(),
            clock=time.perf_counter,
        ).search_after_move(
            board,
            move=losing_move,
            mover=BLACK,
            attacker=WHITE,
            side_to_move=WHITE,
        )

        self.assertIs(ProofState.PROVEN_WIN, proof.state)
        self.assertTrue(proof.completed)
        self.assertLess(proof.nodes, 1_000)
        self.assertEqual(forced_block, proof.principal_variation[0])
        self.assertEqual(before, board_state(board))

    def test_black_twenty_three_two_stage_loss_is_proven(self) -> None:
        board = build_board(BLACK_TWENTY_THREE_PREFIX)
        before = board_state(board)
        proof = ProofSearch(
            budget=ProofBudget(
                max_nodes=4_000,
                max_attacker_moves=10,
                max_quiet_frontiers=16,
                max_quiet_attacker_moves=1,
                use_vcf_oracle=True,
            ),
            analyzer=ThreatAnalyzer(
                candidate_limit=16,
                frontier_scan_limit=24,
            ),
            table=ProofTable(),
            clock=time.perf_counter,
        ).search_after_move(
            board,
            move=parse_move("I11", board.size),
            mover=BLACK,
            attacker=WHITE,
            side_to_move=WHITE,
        )

        self.assertIs(ProofState.PROVEN_WIN, proof.state)
        self.assertTrue(proof.completed)
        self.assertGreaterEqual(len(proof.principal_variation), 5)
        self.assertLess(proof.nodes, 4_000)
        self.assertEqual(before, board_state(board))

    def test_white_twenty_six_multistage_attack_is_proven(self) -> None:
        board = build_board(WHITE_TWENTY_SIX_PREFIX)
        before = board_state(board)
        proof = ProofSearch(
            budget=ProofBudget(
                max_nodes=2_000,
                max_attacker_moves=10,
                max_quiet_frontiers=16,
                max_quiet_attacker_moves=1,
                use_vcf_oracle=True,
            ),
            analyzer=ThreatAnalyzer(
                candidate_limit=16,
                frontier_scan_limit=24,
            ),
            table=ProofTable(),
            clock=time.perf_counter,
        ).search_after_move(
            board,
            move=parse_move("E6", board.size),
            mover=WHITE,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertIs(ProofState.PROVEN_WIN, proof.state)
        self.assertTrue(proof.completed)
        self.assertGreaterEqual(len(proof.principal_variation), 5)
        self.assertLess(proof.nodes, 2_000)
        self.assertEqual(before, board_state(board))

    def test_black_eighty_seven_discovers_every_direct_vcf_rescue(self) -> None:
        board = build_board(BLACK_EIGHTY_SEVEN_PREFIX)
        before = board_state(board)
        ai = SearchAI(
            BLACK,
            max_depth=1,
            time_limit_seconds=None,
            diagnostics=True,
        )
        ai._begin_move_search()
        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )

        scan = ai._run_root_opponent_vcf_scan(board, plan.moves)

        self.assertIsNotNone(scan)
        assert scan is not None
        ai._root_vcf_scan = scan
        survivors = {
            candidate.move
            for candidate in scan.analyses
            if candidate.status
            == RootCandidateSafety.SURVIVES_VCF_SCAN.value
        }
        expected = {
            parse_move(coordinate, board.size)
            for coordinate in ("J2", "C8", "O9", "O10", "C11")
        }
        self.assertTrue(scan.exhaustive_rescue_scanned)
        self.assertEqual(expected, survivors)
        self.assertEqual(
            expected,
            set(ai._filter_root_vcf_candidates(list(scan.candidates))),
        )
        self.assertEqual(before, board_state(board))

    def test_final_audit_rejects_proved_loss_without_fixed_replacement(self) -> None:
        board = build_black_eighty_five_board()
        losing_move = parse_move("L6", board.size)
        forced_block = parse_move("L4", board.size)
        ordinary_alternative = parse_move("M6", board.size)
        before = board_state(board)

        ai = SearchAI(
            BLACK,
            max_depth=1,
            time_limit_seconds=60.0,
            diagnostics=True,
        )
        ai.config = replace(
            ai.config,
            proof_final_time_fraction=0.02,
            proof_final_max_seconds=1.0,
            proof_final_min_seconds=0.05,
        )
        ai._begin_move_search()
        ai._root_vcf_scan = RootVCFScanResult(
            original_candidates=(losing_move, forced_block),
            candidates=(losing_move, forced_block),
            baseline_line=(),
            analyses=(
                RootVCFCandidateAnalysis(
                    move=forced_block,
                    status=RootCandidateSafety.PROVEN_LOSS.value,
                    completed=True,
                    nodes=1,
                    elapsed_seconds=0.0,
                    principal_variation=(forced_block,),
                ),
            ),
            nodes=1,
            elapsed_seconds=0.0,
        )
        # The proof engine itself is exercised without a wall-clock deadline
        # by test_black_eighty_five_losing_exchange_is_strictly_proven.  This
        # test isolates the final-selection policy: seed that strict result so
        # a slower host cannot turn a one-second test deadline into UNKNOWN.
        ai._proof_candidates = (
            ProofCandidateAnalysis(
                move=losing_move,
                state=ProofState.PROVEN_WIN.value,
                completed=True,
                nodes=160,
                elapsed_seconds=0.0,
                cutoff_reason=None,
                principal_variation=(forced_block,),
                threat_risk=None,
            ),
        )
        root = RootResult(
            move=losing_move,
            score=899_300,
            principal_variation=(losing_move,),
            ranked_moves=(
                (losing_move, 899_300),
                (ordinary_alternative, 5_300),
            ),
            ranked_variations=(
                (losing_move, 899_300, (losing_move,)),
                (
                    ordinary_alternative,
                    5_300,
                    (ordinary_alternative,),
                ),
            ),
        )

        revised = ai._run_final_proof_audit(board, root)

        states = {
            candidate.move: candidate.state
            for candidate in ai._proof_candidates
        }
        self.assertEqual(
            ProofState.PROVEN_WIN.value,
            states[losing_move],
        )
        self.assertNotEqual(losing_move, revised.move)
        self.assertIn(losing_move, ai._final_proof_rejected)
        self.assertEqual(before, board_state(board))


if __name__ == "__main__":
    unittest.main()
