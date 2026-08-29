from __future__ import annotations

import unittest
from unittest.mock import patch

from engine.ai import RootSafetyCandidateAnalysis
from engine.board import BLACK, Board
from engine.records import GameRecorder
from engine.search import SearchAI
from engine.search_diagnostics import build_search_analysis
from engine.search_types import RootResult, RootSafetyProbeResult


def root_result(
    leader: tuple[int, int],
    challenger: tuple[int, int],
) -> RootResult:
    return RootResult(
        move=leader,
        score=100,
        principal_variation=(leader,),
        ranked_moves=((leader, 100), (challenger, 90)),
    )


def payload(
    ai: SearchAI,
    selected: tuple[int, int],
) -> dict[str, object]:
    return build_search_analysis(
        ai,
        selected_move=selected,
        reason="test",
        candidate_count=2,
        ranked_moves=[(selected, 100)],
        completed_depth=1,
        principal_variation=(selected,),
        search_completed=True,
    ).to_dict()


class TestRootReviewPairAttemptAudit(unittest.TestCase):
    def setUp(self) -> None:
        self.board = Board()
        self.leader = (7, 7)
        self.challenger = (7, 8)
        self.ai = SearchAI(
            BLACK,
            max_depth=8,
            time_limit_seconds=60.0,
            diagnostics=False,
        )
        self.ai._begin_move_search()

    def _run(self, probe, *, channel: str = "primary"):
        with (
            patch.object(
                self.ai,
                "_frontier_balance_after_move",
                return_value=0,
            ),
            patch.object(
                self.ai,
                "_frontier_shape_after_move",
                return_value=(0, 0, 0, 0),
            ),
            patch.object(
                self.ai,
                "_run_root_safety_probe",
                return_value=probe,
            ),
        ):
            return self.ai._run_dynamic_pair_review(
                self.board,
                root_result(self.leader, self.challenger),
                self.challenger,
                completed_depth=3,
                budget_seconds=0.75,
                audit_channel=channel,
            )

    def test_none_result_remains_visible_in_serialized_attempts(self) -> None:
        self.assertIsNone(self._run(None))

        self.assertEqual(1, len(self.ai._root_review_pair_attempts))
        attempt = self.ai._root_review_pair_attempts[0]
        self.assertEqual("primary", attempt.channel)
        self.assertEqual(self.leader, attempt.leader)
        self.assertEqual(self.challenger, attempt.challenger)
        self.assertEqual("no_completed_layer", attempt.status)
        self.assertEqual(0, attempt.completed_depth)
        self.assertGreaterEqual(attempt.elapsed_seconds, 0.0)

        serialized = payload(self.ai, self.leader)[
            "root_review_pair_attempts"
        ]
        assert isinstance(serialized, list)
        self.assertEqual(1, len(serialized))
        self.assertEqual("H8", serialized[0]["leader_coordinate"])
        self.assertEqual("I8", serialized[0]["challenger_coordinate"])
        self.assertEqual("no_completed_layer", serialized[0]["status"])

    def test_available_result_records_channel_depth_and_nodes(self) -> None:
        probe = RootSafetyProbeResult(
            trigger="dynamic_remaining_review",
            pvs_gap=10,
            main_rank_stable=True,
            completed_depth=4,
            nodes=17,
            candidates=(
                RootSafetyCandidateAnalysis(
                    move=self.leader,
                    score=100,
                    principal_variation=(self.leader,),
                ),
                RootSafetyCandidateAnalysis(
                    move=self.challenger,
                    score=90,
                    principal_variation=(self.challenger,),
                ),
            ),
            leader_history=(self.leader, self.leader),
        )

        result = self._run(probe, channel="boundary_tactical")

        self.assertIsNotNone(result)
        attempt = self.ai._root_review_pair_attempts[0]
        self.assertEqual("boundary_tactical", attempt.channel)
        self.assertEqual("result_available", attempt.status)
        self.assertEqual(4, attempt.completed_depth)
        self.assertEqual(17, attempt.nodes)
        self.assertEqual(0, len(self.ai._root_review_trace))

    def test_begin_move_search_clears_previous_attempts(self) -> None:
        self._run(None)
        self.assertEqual(1, len(self.ai._root_review_pair_attempts))

        self.ai._begin_move_search()

        self.assertEqual([], self.ai._root_review_pair_attempts)

    def test_text_audit_survives_without_root_safety_result(self) -> None:
        recorder = GameRecorder(
            mode="TEST",
            black_name="Black",
            white_name="White",
        )
        recorder.record_move(
            player=BLACK,
            row=7,
            column=7,
            actor="SearchAI",
            think_seconds=0.1,
            evaluation_before=0,
            evaluation_after=0,
            analysis={
                "root_safety_checked": False,
                "root_review_pair_attempts": [
                    {"status": "no_completed_layer"},
                    {"status": "result_available"},
                ],
            },
        )

        details = "\n".join(recorder._render_move_details())

        self.assertIn("root_review_attempts=2", details)
        self.assertIn("no_completed_layer:1", details)
        self.assertIn("result_available:1", details)


if __name__ == "__main__":
    unittest.main()
