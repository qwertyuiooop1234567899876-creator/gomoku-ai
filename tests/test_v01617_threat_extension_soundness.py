from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from engine.ai import DecisionAnalysis, RootSafetyCandidateAnalysis
from engine.board import BLACK, Board
from engine.evaluator import ThreatProfile
from engine.game import parse_move
from engine.root_review import approve_move
from engine.search import SearchAI
from engine.search_types import (
    HEURISTIC_SCORE_LIMIT,
    MATE_SCORE,
    RootResult,
    RootSafetyProbeResult,
    SearchConfig,
)
from tools import native_search_baseline


FIXTURE = (
    Path(__file__).resolve().parent
    / "positions"
    / "v01617_selfplay_move14.json"
)


class TestV01617ThreatExtensionSoundness(unittest.TestCase):
    def test_mixed_extension_batch_keeps_per_move_vcf_classification(self) -> None:
        strict = (7, 7)
        selective = (7, 8)
        board = Board()
        ai = SearchAI(BLACK, time_limit_seconds=None)
        ai._begin_move_search()

        with (
            patch.object(
                ai,
                "_raw_candidates",
                return_value=[selective, strict],
            ),
            patch.object(ai, "_quick_order_score", return_value=0),
            patch.object(
                ai._proof_analyzer,
                "analyze_profiles",
                return_value={
                    strict: ThreatProfile(four_directions=1),
                    selective: ThreatProfile(open_three_directions=1),
                },
            ),
        ):
            options = ai._forcing_attack_options(
                board,
                BLACK,
                vcf_only=False,
                limit=4,
            )

        self.assertEqual(
            {strict: True, selective: False},
            dict(options),
        )

    def test_boundary_secondary_recovers_h10_after_selective_extension(
        self,
    ) -> None:
        case = native_search_baseline.load_case(FIXTURE)
        board = native_search_baseline.build_board(case)
        played = parse_move("I6", board.size)
        defense = parse_move("H10", board.size)
        ai = SearchAI(
            case.player,
            max_depth=8,
            time_limit_seconds=None,
        )
        ai._begin_move_search()

        probe = ai._run_root_safety_probe(
            board,
            [played, defense],
            trigger="v01617_selective_extension_regression",
            pvs_gap=0,
            main_rank_stable=True,
            completed_depth=6,
            budget_seconds=60.0,
            target_depth_override=6,
            minimum_stable_depth=3,
            stable_leader_count=3,
            start_depth=1,
            branch_candidate_limit_override=4,
            recalibrate_mate_like=False,
            reject_mate_like=True,
        )

        self.assertIsNotNone(probe)
        assert probe is not None
        self.assertEqual(6, probe.completed_depth)
        self.assertTrue(probe.final_dimension_recovered)
        self.assertTrue(probe.rank_stable)
        self.assertEqual(
            defense,
            ai._boundary_secondary_approved_move(probe),
        )
        self.assertTrue(
            all(
                abs(candidate.score) < HEURISTIC_SCORE_LIMIT
                for candidate in probe.candidates
            )
        )

    def test_clean_terminal_win_stays_in_mate_band(self) -> None:
        case = native_search_baseline.load_case(FIXTURE)
        board = native_search_baseline.build_board(case)
        ai = SearchAI(case.player, time_limit_seconds=None)
        ai._begin_move_search()

        for column in range(4):
            board.place(14, column, case.player)
        score, _pv = ai._threat_extension(
            board,
            case.player,
            -MATE_SCORE * 2,
            MATE_SCORE * 2,
            ply=0,
            extension_depth=2,
        )

        self.assertGreaterEqual(score, MATE_SCORE - 10_000)

    def test_unstable_frontier_shape_keeps_pvs_move(self) -> None:
        leader = (5, 8)
        shaped = (9, 7)
        result = RootResult(
            leader,
            100,
            (leader,),
            ((leader, 100), (shaped, 100)),
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
                    shaped,
                    -HEURISTIC_SCORE_LIMIT,
                ),
            ),
            leader_history=(leader, shaped),
        )

        move, basis = approve_move(
            SearchConfig(),
            result,
            probe,
            {leader: 5_681, shaped: 5_677},
            structure_keys={
                leader: (11, 6, 45, 21),
                shaped: (12, 6, 44, 17),
            },
            unknown_moves={leader, shaped},
        )

        self.assertEqual(leader, move)
        self.assertEqual("pvs_fallback", basis)

    def test_shallow_frontier_shape_keeps_pvs_move(self) -> None:
        leader = (5, 8)
        shaped = (9, 7)
        result = RootResult(
            leader,
            100,
            (leader,),
            ((leader, 100), (shaped, 100)),
        )
        probe = RootSafetyProbeResult(
            trigger="dynamic_remaining_review",
            pvs_gap=0,
            main_rank_stable=True,
            completed_depth=4,
            nodes=100,
            candidates=(
                RootSafetyCandidateAnalysis(leader, 90_200),
                RootSafetyCandidateAnalysis(
                    shaped,
                    -HEURISTIC_SCORE_LIMIT,
                ),
            ),
            leader_history=(shaped, shaped),
        )

        move, basis = approve_move(
            SearchConfig(),
            result,
            probe,
            {leader: 5_681, shaped: 5_677},
            structure_keys={
                leader: (11, 6, 45, 21),
                shaped: (12, 6, 44, 17),
            },
            unknown_moves={leader, shaped},
        )

        self.assertEqual(leader, move)
        self.assertEqual("pvs_fallback", basis)

    def test_non_vcf_extension_counter_is_serialized(self) -> None:
        payload = DecisionAnalysis(
            selected_move=(7, 7),
            reason="test",
            candidate_count=1,
            selective_non_vcf_extensions=3,
        ).to_dict()

        self.assertEqual(3, payload["selective_non_vcf_extensions"])


if __name__ == "__main__":
    unittest.main()
