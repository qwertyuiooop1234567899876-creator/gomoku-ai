from __future__ import annotations

import json
from pathlib import Path
import unittest

from engine import root_policy, root_review
from engine.ai import RootSafetyCandidateAnalysis
from engine.board import BLACK, WHITE, Board
from engine.game import parse_move
from engine.search_types import (
    HEURISTIC_SCORE_LIMIT,
    RootResult,
    RootSafetyProbeResult,
    SearchConfig,
)


FIXTURE = Path(__file__).parent / "positions" / "v0150_record_20260808.json"


class TestV0150PairedReview(unittest.TestCase):
    def test_record_roots_rebuild_exactly(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            with self.subTest(case=case["name"]):
                board = Board(payload["board_size"])
                for index, coordinate in enumerate(case["history"]):
                    board.place(
                        *parse_move(coordinate, board.size),
                        BLACK if index % 2 == 0 else WHITE,
                    )

                self.assertEqual(case["zobrist_hash"], board.zobrist_hash)
                self.assertTrue(board.is_empty(*parse_move(case["played"])))
                self.assertTrue(board.is_empty(*parse_move(case["review"])))

    def test_strongest_pvs_challenger_is_reserved_before_source_slots(self) -> None:
        leader = parse_move("G7")
        strongest = parse_move("E12")
        low_risk = parse_move("E8")
        bridge = parse_move("G6")
        offensive = parse_move("F11")
        result = RootResult(
            leader,
            73_000,
            (leader,),
            (
                (leader, 73_000),
                (parse_move("E9"), 73_000),
                (strongest, 82_000),
                (low_risk, 43_900),
                (offensive, -14_900),
                (bridge, -26_000),
            ),
        )
        pool = [move for move, _score in result.ranked_moves]

        finalists = root_review.finalists(
            SearchConfig(),
            result,
            pool,
            {move: 0 for move in pool},
            preferred_groups=((low_risk,), (bridge,), (offensive,)),
        )

        self.assertEqual(leader, finalists[0])
        self.assertEqual(strongest, finalists[1])
        self.assertLessEqual(len(finalists), 4)

    def test_distant_pvs_move_cannot_be_overridden_by_structure_only(self) -> None:
        leader = parse_move("E8")
        structural = parse_move("F11")
        result = RootResult(
            leader,
            43_900,
            (leader,),
            ((leader, 43_900), (structural, -14_900)),
        )
        probe = RootSafetyProbeResult(
            trigger="dynamic_remaining_review",
            pvs_gap=58_800,
            main_rank_stable=True,
            completed_depth=4,
            nodes=100,
            candidates=(
                RootSafetyCandidateAnalysis(leader, -HEURISTIC_SCORE_LIMIT),
                RootSafetyCandidateAnalysis(
                    structural,
                    -HEURISTIC_SCORE_LIMIT,
                ),
            ),
            leader_history=(leader, leader, leader),
        )

        move, basis = root_review.approve_move(
            SearchConfig(),
            result,
            probe,
            {leader: 8_471, structural: 12_305},
            unknown_moves={leader, structural},
        )

        self.assertEqual(leader, move)
        self.assertEqual("pvs_fallback", basis)

    def test_near_tied_boundary_can_use_material_frontier_balance(self) -> None:
        original = parse_move("K6")
        preferred = parse_move("J11")
        result = RootResult(
            original,
            -1_100,
            (original,),
            ((original, -1_100), (preferred, -1_200)),
        )
        probe = RootSafetyProbeResult(
            trigger="dynamic_remaining_review",
            pvs_gap=100,
            main_rank_stable=True,
            completed_depth=3,
            nodes=100,
            candidates=(
                RootSafetyCandidateAnalysis(
                    original,
                    -HEURISTIC_SCORE_LIMIT,
                ),
                RootSafetyCandidateAnalysis(
                    preferred,
                    -HEURISTIC_SCORE_LIMIT,
                ),
            ),
            leader_history=(original, original),
        )

        move, basis = root_review.approve_move(
            SearchConfig(),
            result,
            probe,
            {original: -4_298, preferred: -1_939},
            unknown_moves={original, preferred},
        )

        self.assertEqual(preferred, move)
        self.assertEqual("frontier_balance", basis)

    def test_serial_review_and_proof_reserves_are_added(self) -> None:
        self.assertEqual(
            14.0,
            root_policy.serial_verification_reserve(
                final_proof_seconds=8.0,
                root_review_seconds=6.0,
            ),
        )


if __name__ == "__main__":
    unittest.main()
