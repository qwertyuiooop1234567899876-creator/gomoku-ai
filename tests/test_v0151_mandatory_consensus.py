from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from engine import root_candidates, root_review, root_safety
from engine.ai import RootSafetyCandidateAnalysis
from engine.board import BLACK, WHITE, Board
from engine.game import parse_move
from engine.search import SearchAI
from engine.search_types import (
    HEURISTIC_SCORE_LIMIT,
    RootResult,
    RootSafetyProbeResult,
    SearchConfig,
)


FIXTURE = Path(__file__).parent / "positions" / "v0151_record_20260809.json"


class TestV0151MandatoryConsensus(unittest.TestCase):
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

    def test_credible_layer_abstains_on_tie_boundary_and_cliff(self) -> None:
        first = parse_move("F6")
        second = parse_move("J10")

        def leader(first_score: int, second_score: int):
            return root_review.credible_layer_leader(
                (
                    RootSafetyCandidateAnalysis(first, first_score),
                    RootSafetyCandidateAnalysis(second, second_score),
                ),
                score_margin=20_000,
            )

        self.assertEqual(first, leader(-8_100, -9_900))
        self.assertIsNone(leader(-9_800, -9_800))
        self.assertIsNone(leader(-9_000, -HEURISTIC_SCORE_LIMIT))
        self.assertIsNone(leader(-78_100, -998_100))

    def test_unique_boundary_escape_can_override_an_unstable_pvs_tie(self) -> None:
        original = parse_move("J10")
        escape = parse_move("F6")
        result = RootResult(
            original,
            -9_800,
            (original,),
            ((original, -9_800), (escape, -9_800)),
        )
        probe = RootSafetyProbeResult(
            trigger="dynamic_remaining_review",
            pvs_gap=0,
            main_rank_stable=True,
            completed_depth=5,
            nodes=7_292,
            candidates=(
                RootSafetyCandidateAnalysis(escape, -9_000),
                RootSafetyCandidateAnalysis(
                    original,
                    -HEURISTIC_SCORE_LIMIT,
                ),
            ),
            leader_history=(original, escape),
        )

        approved, basis = root_review.approve_move(
            SearchConfig(),
            result,
            probe,
            {original: 1_988, escape: 1_388},
            mandatory_defense_consensus=True,
        )
        approved_probe = replace(
            probe,
            approved_move=approved,
            selection_basis=basis,
        )

        self.assertEqual(escape, approved)
        self.assertEqual("mandatory_boundary_escape", basis)
        self.assertEqual(
            escape,
            root_safety.apply_probe(
                SearchConfig(),
                result,
                approved_probe,
            ).move,
        )

    def test_two_credible_depths_approve_mandatory_defense_leader(self) -> None:
        original = parse_move("I8")
        preferred = parse_move("I4")
        result = RootResult(
            original,
            -17_000,
            (original,),
            ((original, -17_000), (preferred, -16_100)),
        )
        probe = RootSafetyProbeResult(
            trigger="dynamic_remaining_review",
            pvs_gap=900,
            main_rank_stable=True,
            completed_depth=7,
            nodes=6_939,
            candidates=(
                RootSafetyCandidateAnalysis(preferred, 990_000),
                RootSafetyCandidateAnalysis(original, 980_800),
            ),
            leader_history=(preferred, preferred),
        )

        approved, basis = root_review.approve_move(
            SearchConfig(),
            result,
            probe,
            {original: 11_605, preferred: 11_056},
            mandatory_defense_consensus=True,
        )

        self.assertEqual(preferred, approved)
        self.assertEqual("mandatory_depth_consensus", basis)

    def test_mandatory_pair_uses_low_extension_consensus_probe(self) -> None:
        original = parse_move("I8")
        challenger = parse_move("I4")
        ai = SearchAI(player=BLACK, max_depth=8, time_limit_seconds=60)
        source = root_candidates.CandidateSource.MANDATORY_DEFENSE
        ai._root_candidate_sources = {
            original: frozenset({source}),
            challenger: frozenset({source}),
        }
        result = RootResult(
            original,
            -17_000,
            (original,),
            ((original, -17_000), (challenger, -16_100)),
        )

        with (
            patch.object(ai, "_frontier_balance_after_move", return_value=0),
            patch.object(ai, "_frontier_shape_after_move", return_value=(0, 0, 0, 0)),
            patch.object(ai, "_run_root_safety_probe", return_value=None) as run,
        ):
            self.assertIsNone(
                ai._run_dynamic_pair_review(
                    Board(),
                    result,
                    challenger,
                    completed_depth=8,
                    budget_seconds=20.0,
                )
            )

        kwargs = run.call_args.kwargs
        self.assertFalse(kwargs["quiet_frontier_extension"])
        self.assertEqual(1, kwargs["start_depth"])
        self.assertEqual(2, kwargs["stable_leader_count"])
        self.assertEqual(2, kwargs["extension_depth_override"])
        self.assertEqual(12, kwargs["branch_candidate_limit_override"])
        self.assertFalse(kwargs["recalibrate_mate_like"])
        self.assertEqual(20_000, kwargs["credible_score_margin"])


if __name__ == "__main__":
    unittest.main()
