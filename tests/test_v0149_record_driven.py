from __future__ import annotations

import json
from pathlib import Path
import unittest

from engine import root_candidates, root_policy, root_review, root_safety
from engine.ai import (
    ProofCandidateAnalysis,
    RootSafetyCandidateAnalysis,
    RootVCFCandidateAnalysis,
)
from engine.board import BLACK, WHITE, Board
from engine.game import format_move, parse_move
from engine.proof_search import ProofState
from engine.search import SearchAI
from engine.search_types import (
    HEURISTIC_SCORE_LIMIT,
    RootResult,
    RootSafetyProbeResult,
    SearchConfig,
)


FIXTURE = (
    Path(__file__).parent
    / "positions"
    / "v0149_record_20260803.json"
)


def load_fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def board_before(move_number: int, *, rotate: bool = False) -> Board:
    payload = load_fixture()
    board = Board(int(payload["board_size"]))
    moves = payload["moves"]
    assert isinstance(moves, list)
    for index, coordinate in enumerate(moves[: move_number - 1]):
        move = parse_move(str(coordinate), board.size)
        if rotate:
            move = (board.size - 1 - move[0], board.size - 1 - move[1])
        board.place(*move, BLACK if index % 2 == 0 else WHITE)
    return board


def rotated(coordinate: str, size: int = 15) -> str:
    row, column = parse_move(coordinate, size)
    return format_move(size - 1 - row, size - 1 - column)


class TestV0149RecordDriven(unittest.TestCase):
    def test_fixture_rebuilds_move_17_root_exactly(self) -> None:
        board = board_before(17)

        self.assertEqual(16, len(board.move_history))
        self.assertEqual(5797165559781197225, board.zobrist_hash)
        self.assertTrue(board.is_empty(*parse_move("J8")))
        self.assertTrue(board.is_empty(*parse_move("F12")))

    def test_root_membership_ignores_cross_move_history_noise(self) -> None:
        board = board_before(13)
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
        target = parse_move("E7")
        for move in board.get_legal_moves():
            if move != target:
                ai._history_scores[(BLACK, move[0], move[1])] = 1_000_000

        ai._begin_move_search()
        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )

        self.assertIn(target, plan.moves)
        self.assertLessEqual(len(plan.moves), ai.config.root_candidate_limit)

    def test_unscanned_root_expansion_remains_unknown_eligible(self) -> None:
        survivor = (1, 1)
        expanded = (2, 2)
        analyses = (
            RootVCFCandidateAnalysis(
                survivor,
                root_safety.RootCandidateSafety.SURVIVES_VCF_SCAN.value,
                True,
                10,
                0.01,
            ),
        )

        self.assertEqual(
            [survivor, expanded],
            root_safety.apply_vcf_scan([survivor, expanded], analyses),
        )

    def test_dual_frontier_bridge_keeps_e7_at_move_15_and_rotation(self) -> None:
        for rotate, expected in (
            (False, "E7"),
            (True, rotated("E7")),
        ):
            with self.subTest(expected=expected):
                board = board_before(15, rotate=rotate)
                ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
                ai._begin_move_search()
                plan = ai._prepare_root_candidate_plan(
                    board,
                    board.get_legal_moves(),
                )
                move = parse_move(expected)

                self.assertIn(move, plan.moves)
                self.assertIn(
                    root_candidates.CandidateSource.DUAL_FRONTIER_BRIDGE,
                    ai._root_candidate_sources[move],
                )
                self.assertLessEqual(
                    len(plan.moves),
                    ai.config.root_candidate_limit,
                )

    def test_source_aware_finalists_reserve_offensive_move_9(self) -> None:
        names = ("F8", "F5", "F9", "F4", "G9", "I7", "G6")
        moves = {name: parse_move(name) for name in names}
        ranked = tuple(
            (moves[name], 1000 - index)
            for index, name in enumerate(names)
        )
        result = RootResult(moves["F8"], 1000, (moves["F8"],), ranked)
        config = SearchConfig(root_dynamic_review_finalist_limit=4)

        finalists = root_review.finalists(
            config,
            result,
            [move for move, _score in ranked],
            {move: index for index, (move, _score) in enumerate(ranked)},
            preferred_groups=((moves["F5"],), (moves["G6"],)),
        )

        self.assertEqual(moves["F8"], finalists[0])
        self.assertIn(moves["G6"], finalists)
        self.assertLessEqual(
            len(finalists),
            config.root_dynamic_review_finalist_limit,
        )

    def test_mixed_boundary_unknown_pair_can_use_frontier_shape(self) -> None:
        leader = parse_move("J8")
        preferred = parse_move("F12")
        result = RootResult(
            leader,
            100,
            (leader,),
            ((leader, 100), (preferred, 100)),
        )
        probe = RootSafetyProbeResult(
            trigger="dynamic_remaining_review",
            pvs_gap=0,
            main_rank_stable=True,
            completed_depth=5,
            nodes=100,
            candidates=(
                RootSafetyCandidateAnalysis(leader, 90_200),
                RootSafetyCandidateAnalysis(
                    preferred,
                    -HEURISTIC_SCORE_LIMIT,
                ),
            ),
            leader_history=(leader, leader, leader),
        )

        move, basis = root_review.approve_move(
            SearchConfig(),
            result,
            probe,
            {leader: 5_681, preferred: 5_677},
            structure_keys={
                leader: (11, 6, 45, 21),
                preferred: (12, 6, 44, 17),
            },
            unknown_moves={leader, preferred},
        )

        self.assertEqual(preferred, move)
        self.assertEqual("frontier_shape", basis)

    def test_same_saturated_boundary_keeps_pvs_fallback(self) -> None:
        leader = parse_move("K8")
        structural = parse_move("H9")
        result = RootResult(
            leader,
            -20_900,
            (leader,),
            ((leader, -20_900), (structural, -19_900)),
        )
        probe = RootSafetyProbeResult(
            trigger="dynamic_remaining_review",
            pvs_gap=1_000,
            main_rank_stable=True,
            completed_depth=3,
            nodes=60,
            candidates=(
                RootSafetyCandidateAnalysis(
                    leader,
                    -HEURISTIC_SCORE_LIMIT,
                ),
                RootSafetyCandidateAnalysis(
                    structural,
                    -HEURISTIC_SCORE_LIMIT,
                ),
            ),
            leader_history=(leader, leader),
        )

        move, basis = root_review.approve_move(
            SearchConfig(),
            result,
            probe,
            {leader: -3_087, structural: -1_641},
            structure_keys={
                leader: (1, 0, 1, 0),
                structural: (2, 0, 1, 0),
            },
            unknown_moves={leader, structural},
        )

        self.assertEqual(leader, move)
        self.assertEqual("pvs_fallback", basis)

    def test_opposite_saturated_bounds_do_not_use_frontier_shape(self) -> None:
        leader = parse_move("F10")
        structural = parse_move("H3")
        result = RootResult(
            leader,
            8_600,
            (leader,),
            ((leader, 8_600), (structural, 7_800)),
        )
        probe = RootSafetyProbeResult(
            trigger="dynamic_remaining_review",
            pvs_gap=800,
            main_rank_stable=True,
            completed_depth=3,
            nodes=142,
            candidates=(
                RootSafetyCandidateAnalysis(
                    leader,
                    HEURISTIC_SCORE_LIMIT,
                ),
                RootSafetyCandidateAnalysis(
                    structural,
                    -HEURISTIC_SCORE_LIMIT,
                ),
            ),
            leader_history=(leader, leader),
        )

        move, basis = root_review.approve_move(
            SearchConfig(),
            result,
            probe,
            {leader: 100, structural: 200},
            structure_keys={
                leader: (1, 0, 1, 0),
                structural: (2, 0, 1, 0),
            },
            unknown_moves={leader, structural},
        )

        self.assertEqual(leader, move)
        self.assertEqual("pvs_fallback", basis)

    def test_proof_like_scores_are_not_frontier_shape_boundaries(self) -> None:
        leader = parse_move("E7")
        structural = parse_move("F8")
        result = RootResult(
            leader,
            100,
            (leader,),
            ((leader, 100), (structural, 100)),
        )
        probe = RootSafetyProbeResult(
            trigger="dynamic_remaining_review",
            pvs_gap=0,
            main_rank_stable=True,
            completed_depth=2,
            nodes=20,
            candidates=(
                RootSafetyCandidateAnalysis(leader, -999_999_996),
                RootSafetyCandidateAnalysis(structural, -999_999_997),
            ),
            leader_history=(leader,),
        )

        move, basis = root_review.approve_move(
            SearchConfig(),
            result,
            probe,
            {leader: 0, structural: 100},
            structure_keys={leader: (0,), structural: (1,)},
            unknown_moves={leader, structural},
        )

        self.assertEqual(leader, move)
        self.assertEqual("pvs_fallback", basis)

    def test_stable_positive_boundary_can_select_without_becoming_proof(self) -> None:
        original = parse_move("G7")
        preferred = parse_move("E12")
        result = RootResult(
            original,
            73_000,
            (original,),
            ((original, 73_000), (preferred, 82_000)),
        )
        probe = RootSafetyProbeResult(
            trigger="dynamic_remaining_review",
            pvs_gap=9_000,
            main_rank_stable=True,
            completed_depth=5,
            nodes=100,
            candidates=(
                RootSafetyCandidateAnalysis(
                    preferred,
                    HEURISTIC_SCORE_LIMIT,
                ),
                RootSafetyCandidateAnalysis(original, 30_000),
            ),
            leader_history=(preferred, preferred, preferred),
        )

        move, basis = root_review.approve_move(
            SearchConfig(),
            result,
            probe,
            {original: 5_081, preferred: 3_483},
            unknown_moves={original, preferred},
        )

        self.assertEqual(preferred, move)
        self.assertEqual("equal_window", basis)

    def test_final_proof_divisor_uses_real_queue_not_empty_slots(self) -> None:
        self.assertEqual(
            2,
            root_policy.pending_proof_checks(
                candidate_limit=4,
                checks_completed=1,
                queued_unseen=1,
            ),
        )
        self.assertEqual(
            1,
            root_policy.pending_proof_checks(
                candidate_limit=4,
                checks_completed=2,
                queued_unseen=0,
            ),
        )

    def test_unknown_source_is_not_upgraded_to_proof(self) -> None:
        move = parse_move("E7")
        analysis = ProofCandidateAnalysis(
            move,
            ProofState.UNKNOWN.value,
            False,
            10,
            0.01,
        )

        self.assertEqual(ProofState.UNKNOWN.value, analysis.state)
        self.assertFalse(analysis.completed)


if __name__ == "__main__":
    unittest.main()
