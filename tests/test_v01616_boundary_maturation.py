from __future__ import annotations

import unittest
from unittest.mock import patch

from engine.ai import RootSafetyCandidateAnalysis
from engine.board import BLACK, Board
from engine.search import SearchAI
from engine.search_diagnostics import build_search_analysis
from engine.search_types import (
    HEURISTIC_SCORE_LIMIT,
    MATE_SCORE,
    RootResult,
    RootSafetyProbeResult,
)


H6 = (5, 7)
G10 = (9, 6)


def fake_negamax_with_scores(scores_by_depth):
    def fake_negamax(
        _search,
        board,
        _player,
        depth,
        _alpha,
        _beta,
        **_kwargs,
    ):
        root_depth = depth + 1
        move = board.move_history[-1][:2]
        return -scores_by_depth(root_depth, move), ()

    return fake_negamax


def run_secondary_probe(score_at_depth):
    ai = SearchAI(
        BLACK,
        max_depth=8,
        time_limit_seconds=60.0,
        diagnostics=True,
    )
    ai._begin_move_search()
    with patch.object(
        SearchAI,
        "_negamax",
        new=fake_negamax_with_scores(score_at_depth),
    ):
        probe = ai._run_root_safety_probe(
            Board(),
            [H6, G10],
            trigger="boundary_secondary_regression",
            pvs_gap=799_400,
            main_rank_stable=True,
            completed_depth=6,
            budget_seconds=1.6,
            quiet_frontier_extension=False,
            target_depth_override=8,
            minimum_stable_depth=3,
            stable_leader_count=3,
            start_depth=1,
            branch_candidate_limit_override=(
                ai.config.root_boundary_review_branch_limit
            ),
            recalibrate_mate_like=False,
            reject_mate_like=True,
        )
    return ai, probe


class TestV01616BoundaryMaturation(unittest.TestCase):
    def test_transient_mate_like_layer_recovers_and_approves(self) -> None:
        def score_at_depth(depth, move):
            if depth <= 3:
                return -MATE_SCORE + 4 if move == H6 else -99_900
            return -901_600 if move == H6 else -92_000

        ai, probe = run_secondary_probe(score_at_depth)

        self.assertIsNotNone(probe)
        assert probe is not None
        self.assertEqual(4, probe.completed_depth)
        self.assertEqual((1, 2, 3), probe.mate_like_hit_depths)
        self.assertTrue(probe.final_dimension_recovered)
        self.assertEqual((G10, G10, G10, G10), probe.leader_history)
        self.assertEqual(G10, ai._boundary_secondary_approved_move(probe))

    def test_persistent_mate_like_layers_remain_unresolved(self) -> None:
        def score_at_depth(_depth, move):
            return -MATE_SCORE + (4 if move == H6 else 5)

        ai, probe = run_secondary_probe(score_at_depth)

        self.assertIsNotNone(probe)
        assert probe is not None
        self.assertEqual(tuple(range(1, 9)), probe.mate_like_hit_depths)
        self.assertFalse(probe.final_dimension_recovered)
        self.assertIsNone(ai._boundary_secondary_approved_move(probe))

        initial = RootSafetyProbeResult(
            trigger="dynamic_remaining_review",
            pvs_gap=0,
            main_rank_stable=True,
            completed_depth=2,
            nodes=10,
            candidates=(
                RootSafetyCandidateAnalysis(H6, -HEURISTIC_SCORE_LIMIT),
                RootSafetyCandidateAnalysis(G10, -HEURISTIC_SCORE_LIMIT),
            ),
            leader_history=(H6,),
            approved_move=H6,
            selection_basis="pvs_fallback",
            requested_budget_seconds=1.0,
        )
        tactical = RootSafetyProbeResult(
            trigger=initial.trigger,
            pvs_gap=initial.pvs_gap,
            main_rank_stable=True,
            completed_depth=3,
            nodes=20,
            candidates=initial.candidates,
            leader_history=(H6,),
            approved_move=H6,
            selection_basis="pvs_fallback",
            requested_budget_seconds=2.4,
        )
        result = RootResult(
            move=H6,
            score=0,
            principal_variation=(H6,),
            ranked_moves=((H6, 0), (G10, 0)),
        )
        with (
            patch.object(
                ai,
                "_boundary_tie_escalation_budget_seconds",
                return_value=4.0,
            ),
            patch.object(
                ai,
                "_run_dynamic_pair_review",
                side_effect=(tactical, probe),
            ),
        ):
            fallback = ai._escalate_boundary_tie_review(
                Board(),
                result,
                G10,
                initial_probe=initial,
                completed_depth=6,
                fallback_move=H6,
            )

        self.assertEqual(H6, fallback.approved_move)
        self.assertEqual(
            "boundary_tie_pvs_fallback",
            fallback.selection_basis,
        )

    def test_unstable_recovered_leader_is_not_approved(self) -> None:
        def score_at_depth(depth, move):
            if depth == 1:
                return -MATE_SCORE + 4 if move == H6 else -99_900
            leader = G10 if depth % 2 else H6
            return 1_000 if move == leader else 900

        ai, probe = run_secondary_probe(score_at_depth)

        self.assertIsNotNone(probe)
        assert probe is not None
        self.assertTrue(probe.final_dimension_recovered)
        self.assertIsNone(ai._boundary_secondary_approved_move(probe))

    def test_secondary_maturation_is_serialized(self) -> None:
        def score_at_depth(depth, move):
            if depth <= 3:
                return -MATE_SCORE + 4 if move == H6 else -99_900
            return -901_600 if move == H6 else -92_000

        ai, probe = run_secondary_probe(score_at_depth)
        assert probe is not None
        ai._root_safety_probe = probe
        ai._root_review_trace.append(("boundary_secondary", probe))
        ai._root_boundary_secondary_attempted = True
        ai._root_boundary_secondary_mate_like_hit_depths = (
            probe.mate_like_hit_depths
        )
        ai._root_boundary_secondary_final_dimension_recovered = (
            probe.final_dimension_recovered
        )
        result = RootResult(
            move=G10,
            score=-92_000,
            principal_variation=(G10,),
            ranked_moves=((G10, -92_000), (H6, -901_600)),
        )

        payload = build_search_analysis(
            ai,
            selected_move=result.move,
            reason="test",
            candidate_count=2,
            ranked_moves=list(result.ranked_moves),
            completed_depth=4,
            principal_variation=result.principal_variation,
            search_completed=True,
            stop_reason="test",
        ).to_dict()

        self.assertTrue(payload["boundary_secondary_attempted"])
        self.assertEqual(
            [1, 2, 3],
            payload["boundary_secondary_mate_like_hit_depths"],
        )
        self.assertTrue(
            payload["boundary_secondary_final_dimension_recovered"]
        )
        pair = payload["root_review_pairs"][0]
        self.assertEqual([1, 2, 3], pair["mate_like_hit_depths"])
        self.assertTrue(pair["final_dimension_recovered"])


if __name__ == "__main__":
    unittest.main()
