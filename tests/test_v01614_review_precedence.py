from __future__ import annotations

import unittest
from dataclasses import replace

from engine.ai import RootSafetyCandidateAnalysis
from engine.board import BLACK, Board
from engine.search import SearchAI
from engine.search_diagnostics import build_search_analysis
from engine.search_types import (
    HEURISTIC_SCORE_LIMIT,
    RootResult,
    RootSafetyProbeResult,
)


def root_result(
    leader: tuple[int, int],
    challenger: tuple[int, int],
) -> RootResult:
    return RootResult(
        leader,
        100,
        (leader,),
        ((leader, 100), (challenger, 90)),
    )


def probe(
    leader: tuple[int, int],
    challenger: tuple[int, int],
    *,
    approved: tuple[int, int],
    basis: str,
    depth: int = 4,
    rank_stable: bool = True,
    scores: tuple[int, int] = (20_200, 10_200),
) -> RootSafetyProbeResult:
    return RootSafetyProbeResult(
        trigger="dynamic_remaining_review",
        pvs_gap=10,
        main_rank_stable=True,
        completed_depth=depth,
        nodes=10,
        candidates=(
            RootSafetyCandidateAnalysis(leader, scores[0]),
            RootSafetyCandidateAnalysis(challenger, scores[1]),
        ),
        leader_history=(
            (leader, leader)
            if rank_stable
            else (challenger, leader)
        ),
        approved_move=approved,
        selection_basis=basis,
        requested_budget_seconds=1.0,
    )


class TestV01614ReviewPrecedence(unittest.TestCase):
    def test_probe_records_exact_depth_for_each_leader(self) -> None:
        ai = SearchAI(
            BLACK,
            max_depth=2,
            time_limit_seconds=2.0,
            branch_candidate_limit=2,
        )
        ai._begin_move_search()
        result = ai._run_root_safety_probe(
            Board(),
            [(7, 7), (7, 8)],
            trigger="test",
            pvs_gap=0,
            main_rank_stable=True,
            completed_depth=1,
            budget_seconds=1.0,
            target_depth_override=2,
            minimum_stable_depth=2,
            stable_leader_count=2,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual((1, 2), tuple(
            depth for depth, _leader in result.leader_depth_history
        ))
        self.assertEqual(
            result.leader_history,
            tuple(
                leader for _depth, leader in result.leader_depth_history
            ),
        )

    def test_parity_shadow_records_consensus_without_changing_move(self) -> None:
        leader = (7, 8)
        challenger = (8, 7)
        ai = SearchAI(BLACK, time_limit_seconds=60.0)
        ai._begin_move_search()
        parity_probe = replace(
            probe(
                leader,
                challenger,
                approved=leader,
                basis="equal_window",
                depth=6,
            ),
            leader_history=(leader, leader, leader, leader),
            leader_depth_history=(
                (3, leader),
                (4, leader),
                (5, leader),
                (6, leader),
            ),
        )

        revised = ai._apply_and_record_root_safety(
            root_result(leader, challenger),
            parity_probe,
        )

        self.assertEqual(leader, revised.move)
        self.assertEqual(leader, ai._root_review_confirmed_move)
        self.assertEqual(1, len(ai._root_review_parity_shadow_events))
        event = ai._root_review_parity_shadow_events[0]
        self.assertEqual("parity_consistent", event.parity_state)
        self.assertEqual(leader, event.parity_leader)
        self.assertFalse(event.would_veto)

    def test_parity_shadow_disagreement_has_no_decision_authority(self) -> None:
        leader = (7, 8)
        challenger = (8, 7)
        ai = SearchAI(BLACK, time_limit_seconds=60.0)
        ai._begin_move_search()
        parity_probe = replace(
            probe(
                leader,
                challenger,
                approved=leader,
                basis="equal_window",
                depth=6,
            ),
            leader_history=(leader, challenger, leader, leader),
            leader_depth_history=(
                (3, leader),
                (4, challenger),
                (5, leader),
                (6, leader),
            ),
        )

        revised = ai._apply_and_record_root_safety(
            root_result(leader, challenger),
            parity_probe,
        )

        self.assertEqual(leader, revised.move)
        self.assertEqual(leader, ai._root_review_confirmed_move)
        event = ai._root_review_parity_shadow_events[0]
        self.assertEqual("parity_disagreement", event.parity_state)
        self.assertIsNone(event.parity_leader)
        self.assertTrue(event.would_veto)

    def test_parity_shadow_keeps_insufficient_history_separate(self) -> None:
        leader = (7, 8)
        challenger = (8, 7)
        ai = SearchAI(BLACK, time_limit_seconds=60.0)
        ai._begin_move_search()
        parity_probe = replace(
            probe(
                leader,
                challenger,
                approved=leader,
                basis="equal_window",
                depth=5,
            ),
            leader_history=(leader, leader, leader),
            leader_depth_history=(
                (3, leader),
                (4, leader),
                (5, leader),
            ),
        )

        revised = ai._apply_and_record_root_safety(
            root_result(leader, challenger),
            parity_probe,
        )

        self.assertEqual(leader, revised.move)
        event = ai._root_review_parity_shadow_events[0]
        self.assertEqual(
            "insufficient_parity_history",
            event.parity_state,
        )
        self.assertTrue(event.would_veto)

    def test_parity_shadow_event_survives_confirmation_clear(self) -> None:
        leader = (7, 8)
        challenger = (8, 7)
        ai = SearchAI(BLACK, time_limit_seconds=60.0)
        ai._begin_move_search()
        confirmed = ai._apply_and_record_root_safety(
            root_result(leader, challenger),
            probe(
                leader,
                challenger,
                approved=leader,
                basis="equal_window",
            ),
        )

        revised = ai._apply_and_record_root_safety(
            confirmed,
            probe(
                leader,
                challenger,
                approved=challenger,
                basis="frontier_balance",
                rank_stable=True,
            ),
        )

        self.assertEqual(challenger, revised.move)
        self.assertIsNone(ai._root_review_confirmed_move)
        self.assertEqual(1, len(ai._root_review_parity_shadow_events))

    def test_non_confirmation_does_not_create_parity_shadow_event(self) -> None:
        leader = (7, 8)
        challenger = (8, 7)
        ai = SearchAI(BLACK, time_limit_seconds=60.0)
        ai._begin_move_search()

        ai._apply_and_record_root_safety(
            root_result(leader, challenger),
            probe(
                leader,
                challenger,
                approved=challenger,
                basis="frontier_shape",
            ),
        )

        self.assertEqual([], ai._root_review_parity_shadow_events)

    def test_unstable_structure_cannot_replace_confirmed_leader(self) -> None:
        leader = (9, 6)
        challenger = (5, 10)
        for basis in ("frontier_balance", "frontier_shape"):
            with self.subTest(basis=basis):
                ai = SearchAI(BLACK, time_limit_seconds=60.0)
                ai._begin_move_search()
                result = root_result(leader, challenger)

                confirmed = ai._apply_and_record_root_safety(
                    result,
                    probe(
                        leader,
                        challenger,
                        approved=leader,
                        basis="equal_window",
                    ),
                )
                revised = ai._apply_and_record_root_safety(
                    confirmed,
                    probe(
                        leader,
                        challenger,
                        approved=challenger,
                        basis=basis,
                        rank_stable=False,
                        scores=(-999_100, -1_010_000),
                    ),
                )

                self.assertEqual(leader, revised.move)
                self.assertEqual(
                    "unstable_structure_after_confirmation",
                    ai._root_review_apply_reason,
                )
                self.assertEqual(
                    challenger,
                    ai._root_review_approved_move,
                )
                self.assertFalse(ai._root_review_result_changed)

    def test_structure_still_applies_without_prior_confirmation(self) -> None:
        leader = (9, 6)
        challenger = (5, 10)
        ai = SearchAI(BLACK, time_limit_seconds=60.0)
        ai._begin_move_search()

        revised = ai._apply_and_record_root_safety(
            root_result(leader, challenger),
            probe(
                leader,
                challenger,
                approved=challenger,
                basis="frontier_balance",
                rank_stable=False,
            ),
        )

        self.assertEqual(challenger, revised.move)
        self.assertEqual("applied", ai._root_review_apply_reason)

    def test_stable_structure_at_confirmed_depth_can_replace_leader(self) -> None:
        leader = (9, 6)
        challenger = (5, 10)
        ai = SearchAI(BLACK, time_limit_seconds=60.0)
        ai._begin_move_search()
        confirmed = ai._apply_and_record_root_safety(
            root_result(leader, challenger),
            probe(
                leader,
                challenger,
                approved=leader,
                basis="equal_window",
                depth=4,
            ),
        )

        revised = ai._apply_and_record_root_safety(
            confirmed,
            probe(
                leader,
                challenger,
                approved=challenger,
                basis="frontier_balance",
                depth=4,
                rank_stable=True,
            ),
        )

        self.assertEqual(challenger, revised.move)
        self.assertEqual("applied", ai._root_review_apply_reason)
        self.assertIsNone(ai._root_review_confirmed_move)

    def test_mandatory_boundary_escape_can_replace_confirmation(self) -> None:
        leader = (7, 7)
        escape = (7, 8)
        ai = SearchAI(BLACK, time_limit_seconds=60.0)
        ai._begin_move_search()
        result = root_result(leader, escape)
        confirmed = ai._apply_and_record_root_safety(
            result,
            probe(
                leader,
                escape,
                approved=leader,
                basis="equal_window",
                depth=3,
            ),
        )

        revised = ai._apply_and_record_root_safety(
            confirmed,
            probe(
                leader,
                escape,
                approved=escape,
                basis="mandatory_boundary_escape",
                depth=4,
                rank_stable=False,
                scores=(-HEURISTIC_SCORE_LIMIT, 12_000),
            ),
        )

        self.assertEqual(escape, revised.move)
        self.assertEqual("applied", ai._root_review_apply_reason)
        self.assertIsNone(ai._root_review_confirmed_move)

    def test_shallower_mandatory_escape_cannot_replace_confirmation(self) -> None:
        leader = (7, 7)
        escape = (7, 8)
        ai = SearchAI(BLACK, time_limit_seconds=60.0)
        ai._begin_move_search()
        confirmed = ai._apply_and_record_root_safety(
            root_result(leader, escape),
            probe(
                leader,
                escape,
                approved=leader,
                basis="equal_window",
                depth=5,
            ),
        )

        revised = ai._apply_and_record_root_safety(
            confirmed,
            probe(
                leader,
                escape,
                approved=escape,
                basis="mandatory_boundary_escape",
                depth=4,
                rank_stable=False,
                scores=(-HEURISTIC_SCORE_LIMIT, 12_000),
            ),
        )

        self.assertEqual(leader, revised.move)
        self.assertEqual(
            "shallower_than_confirmed_review",
            ai._root_review_apply_reason,
        )

    def test_confirmation_provenance_is_serialized(self) -> None:
        leader = (7, 8)
        challenger = (8, 7)
        ai = SearchAI(BLACK, time_limit_seconds=60.0, diagnostics=True)
        ai._begin_move_search()
        result = root_result(leader, challenger)
        revised = ai._apply_and_record_root_safety(
            result,
            replace(
                probe(
                    leader,
                    challenger,
                    approved=leader,
                    basis="equal_window",
                    depth=4,
                ),
                leader_depth_history=(
                    (1, leader),
                    (2, leader),
                    (3, leader),
                    (4, leader),
                ),
            ),
        )

        payload = build_search_analysis(
            ai,
            selected_move=revised.move,
            reason="test",
            candidate_count=2,
            ranked_moves=list(revised.ranked_moves),
            completed_depth=4,
            principal_variation=revised.principal_variation,
            search_completed=False,
            stop_reason="test",
        ).to_dict()

        self.assertEqual(4, payload["root_review_confirmed_depth"])
        self.assertEqual(
            "equal_window",
            payload["root_review_confirmed_basis"],
        )
        self.assertTrue(payload["root_review_confirmed_rank_stable"])
        self.assertFalse(payload["root_review_confirmed_boundary"])
        events = payload["root_review_parity_shadow_events"]
        self.assertEqual(1, len(events))
        self.assertEqual("parity_consistent", events[0]["parity_state"])
        self.assertEqual("I8", events[0]["parity_leader_coordinate"])
        self.assertFalse(events[0]["would_veto"])


if __name__ == "__main__":
    unittest.main()
