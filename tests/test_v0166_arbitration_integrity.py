from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from engine import root_review
from engine.ai import DecisionAnalysis, RootSafetyCandidateAnalysis
from engine.board import BLACK, WHITE, Board
from engine.game import parse_move
from engine.search import SearchAI
from engine.search_diagnostics import review_arbitration_state
from engine.search_types import (
    HEURISTIC_SCORE_LIMIT,
    RootResult,
    RootSafetyProbeResult,
    SearchConfig,
)


SELFPLAY_MOVE_17_PREFIX = (
    "H8 G7 I7 G9 G8 H9 I8 J8 F8 E8 F7 F9 I9 I10 I6 I5"
)


def replay_prefix(text: str) -> Board:
    board = Board()
    for index, coordinate in enumerate(text.split()):
        board.place(
            *parse_move(coordinate, board.size),
            BLACK if index % 2 == 0 else WHITE,
        )
    return board


def probe(
    first: tuple[int, int],
    second: tuple[int, int],
    *,
    depth: int,
    score: int = -HEURISTIC_SCORE_LIMIT,
    leader_history: tuple[tuple[int, int], ...] = (),
    approved_move: tuple[int, int] | None = None,
    selection_basis: str = "pvs_fallback",
    requested_budget_seconds: float = 0.0,
) -> RootSafetyProbeResult:
    return RootSafetyProbeResult(
        trigger="dynamic_remaining_review",
        pvs_gap=799_400,
        main_rank_stable=True,
        completed_depth=depth,
        nodes=38,
        candidates=(
            RootSafetyCandidateAnalysis(first, score),
            RootSafetyCandidateAnalysis(second, score),
        ),
        leader_history=leader_history,
        approved_move=approved_move,
        selection_basis=selection_basis,
        requested_budget_seconds=requested_budget_seconds,
    )


class TestV0166ArbitrationIntegrity(unittest.TestCase):
    def test_move17_boundary_tie_escalates_before_explicit_fallback(
        self,
    ) -> None:
        board = replay_prefix(SELFPLAY_MOVE_17_PREFIX)
        h6 = parse_move("H6", board.size)
        g10 = parse_move("G10", board.size)
        result = RootResult(
            move=h6,
            score=-101_500,
            principal_variation=(h6,),
            ranked_moves=((h6, -101_500), (g10, -900_900)),
            ranked_variations=(
                (h6, -101_500, (h6,)),
                (g10, -900_900, (g10,)),
            ),
        )
        initial = probe(
            h6,
            g10,
            depth=2,
            leader_history=(h6,),
            approved_move=h6,
            requested_budget_seconds=3.0,
        )
        still_unresolved = probe(
            h6,
            g10,
            depth=3,
            leader_history=(h6,),
            approved_move=h6,
            requested_budget_seconds=2.0,
        )
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=60.0)
        ai._begin_move_search()

        with (
            patch.object(ai, "_dynamic_review_budget_seconds", return_value=3.0),
            patch.object(
                ai,
                "_boundary_tie_escalation_budget_seconds",
                return_value=2.0,
            ),
            patch.object(
                ai,
                "_critical_root_review_groups",
                return_value=(),
            ),
            patch.object(
                ai,
                "_frontier_balance_after_move",
                return_value=0,
            ),
            patch.object(
                ai,
                "_run_dynamic_pair_review",
                side_effect=(initial, still_unresolved),
            ) as run_pair,
            patch.object(root_review, "review_pool", return_value=[h6, g10]),
            patch.object(root_review, "finalists", return_value=[h6, g10]),
        ):
            actual = ai._maybe_run_dynamic_root_review(
                board,
                result,
                [h6, g10],
                completed_depth=6,
            )

        self.assertIsNotNone(actual)
        assert actual is not None
        self.assertEqual(2, run_pair.call_count)
        escalation_call = run_pair.call_args_list[1]
        self.assertEqual(8, escalation_call.kwargs["target_depth_override"])
        self.assertEqual(3, escalation_call.kwargs["start_depth"])
        self.assertEqual(
            ai.config.root_boundary_review_branch_limit,
            escalation_call.kwargs["branch_candidate_limit_override"],
        )
        self.assertEqual("boundary_tie_pvs_fallback", actual.selection_basis)
        self.assertEqual(h6, actual.approved_move)
        self.assertTrue(actual.boundary_tie_detected)
        self.assertEqual(5.0, actual.requested_budget_seconds)
        self.assertEqual(2.0, actual.escalation_budget_seconds)

    def test_boundary_escalation_borrows_only_final_proof_reserve(self) -> None:
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=60.0)
        ai.config = replace(
            ai.config,
            root_boundary_review_shared_fraction=0.5,
            root_boundary_review_max_seconds=6.0,
        )
        ai._begin_move_search()

        with patch.object(
            ai,
            "_final_proof_reserve_seconds",
            return_value=8.0,
        ):
            budget = ai._boundary_tie_escalation_budget_seconds()

        self.assertEqual(4.0, budget)
        self.assertLessEqual(
            budget,
            8.0 * ai.config.root_boundary_review_shared_fraction,
        )

    def test_completed_escalation_can_replace_the_pvs_fallback(self) -> None:
        board = replay_prefix(SELFPLAY_MOVE_17_PREFIX)
        h6 = parse_move("H6", board.size)
        g10 = parse_move("G10", board.size)
        initial = probe(
            h6,
            g10,
            depth=2,
            approved_move=h6,
            requested_budget_seconds=3.0,
        )
        resolved = replace(
            probe(
                g10,
                h6,
                depth=5,
                score=-80_000,
                leader_history=(g10, g10),
                approved_move=g10,
                selection_basis="equal_window",
                requested_budget_seconds=2.0,
            ),
            candidates=(
                RootSafetyCandidateAnalysis(g10, -80_000),
                RootSafetyCandidateAnalysis(h6, -90_000),
            ),
        )
        result = RootResult(
            move=h6,
            score=-101_500,
            principal_variation=(h6,),
            ranked_moves=((h6, -101_500), (g10, -900_900)),
        )
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=60.0)

        with (
            patch.object(
                ai,
                "_boundary_tie_escalation_budget_seconds",
                return_value=2.0,
            ),
            patch.object(
                ai,
                "_run_dynamic_pair_review",
                return_value=resolved,
            ),
        ):
            actual = ai._escalate_boundary_tie_review(
                board,
                result,
                g10,
                initial_probe=initial,
                completed_depth=6,
                fallback_move=h6,
            )

        self.assertEqual(g10, actual.approved_move)
        self.assertEqual("equal_window", actual.selection_basis)
        self.assertTrue(actual.boundary_tie_detected)
        self.assertEqual(5.0, actual.requested_budget_seconds)

    def test_proof_like_mate_scores_are_not_clamp_ties(self) -> None:
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=60.0)
        near_mate = probe(
            (5, 7),
            (9, 6),
            depth=2,
            score=-999_999_990,
        )
        opposite_boundaries = replace(
            near_mate,
            candidates=(
                RootSafetyCandidateAnalysis(
                    (5, 7),
                    -HEURISTIC_SCORE_LIMIT,
                ),
                RootSafetyCandidateAnalysis(
                    (9, 6),
                    HEURISTIC_SCORE_LIMIT,
                ),
            ),
        )

        self.assertFalse(ai._needs_boundary_tie_escalation(near_mate))
        self.assertFalse(
            ai._needs_boundary_tie_escalation(opposite_boundaries)
        )

    def test_review_states_distinguish_unresolved_and_unstable(self) -> None:
        first = (5, 7)
        second = (9, 6)
        config = SearchConfig(max_depth=8, time_limit_seconds=60.0)
        boundary = replace(
            probe(first, second, depth=3),
            selection_basis="boundary_tie_pvs_fallback",
            boundary_tie_detected=True,
        )
        unstable = probe(
            first,
            second,
            depth=5,
            score=-97_800,
            leader_history=(second,),
        )
        completed = replace(
            unstable,
            leader_history=(second, second),
            approved_move=second,
            selection_basis="equal_window",
        )

        self.assertEqual(
            "boundary_tie_unresolved",
            review_arbitration_state(boundary, config),
        )
        self.assertEqual(
            "insufficient_depth",
            review_arbitration_state(unstable, config),
        )
        self.assertEqual(
            "completed",
            review_arbitration_state(completed, config),
        )

    def test_review_provenance_is_serialized(self) -> None:
        payload = DecisionAnalysis(
            selected_move=(5, 7),
            reason="test",
            candidate_count=2,
            review_arbitration_state="boundary_tie_unresolved",
            review_completed_depth=3,
            review_rank_stable=False,
            review_boundary_tie_detected=True,
            review_budget_seconds=5.0,
            review_escalation_budget_seconds=2.0,
        ).to_dict()

        self.assertEqual(
            "boundary_tie_unresolved",
            payload["review_arbitration_state"],
        )
        self.assertEqual(3, payload["review_completed_depth"])
        self.assertTrue(payload["review_boundary_tie_detected"])
        self.assertEqual(5.0, payload["review_budget_seconds"])
        self.assertEqual(2.0, payload["review_escalation_budget_seconds"])

    def test_invalid_boundary_review_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SearchConfig(root_boundary_review_shared_fraction=1.0)
        with self.assertRaises(ValueError):
            SearchConfig(root_boundary_review_branch_limit=1)


if __name__ == "__main__":
    unittest.main()
