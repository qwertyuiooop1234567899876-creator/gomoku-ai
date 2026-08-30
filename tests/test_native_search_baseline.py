from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path

from engine.game import parse_move
from engine.search import SearchAI
from tools import native_search_baseline


MOVE_21_REVIEW_FIXTURE = (
    Path(__file__).resolve().parent
    / "positions"
    / "v0172_yixin_move21_native_review.json"
)


class TestNativeSearchBaseline(unittest.TestCase):
    def test_fixture_preserves_ordered_move_13_position(self) -> None:
        case = native_search_baseline.load_case()
        board = native_search_baseline.build_board(case)

        self.assertEqual(12, len(board.move_history))
        self.assertEqual(case.expected_hash, board.zobrist_hash)
        self.assertEqual(
            (7, 7, 1),
            board.move_history[0],
        )
        self.assertEqual(
            (9, 8, 2),
            board.move_history[-1],
        )
        self.assertTrue(board.is_empty(*parse_move("F7", board.size)))
        self.assertTrue(board.is_empty(*parse_move("J11", board.size)))

    def test_depth_one_full_window_is_reproducible(self) -> None:
        case = native_search_baseline.load_case()
        first = native_search_baseline.run_full_window_candidate(
            case,
            "F7",
            1,
        )
        second = native_search_baseline.run_full_window_candidate(
            case,
            "F7",
            1,
        )

        self.assertEqual("F7", first.selected_move)
        self.assertEqual(first.score, second.score)
        self.assertEqual(first.nodes, second.nodes)
        self.assertEqual(first.tt_digest, second.tt_digest)

    def test_iterative_pair_obeys_fixed_node_limit(self) -> None:
        case = native_search_baseline.load_case()
        run = native_search_baseline.run_iterative_pair(
            case,
            3,
            node_limit=1,
        )

        self.assertFalse(run.completed)
        self.assertEqual("node_limit", run.stop_reason)
        self.assertEqual(1, run.nodes)

    def test_native_pair_runs_without_production_integration(self) -> None:
        case = native_search_baseline.load_case()
        run = native_search_baseline.run_native_pair(
            case,
            1,
            node_limit=None,
            threat_extension_depth=2,
            branch_candidate_limit=8,
        )

        self.assertTrue(run.completed)
        self.assertEqual(1, run.completed_depth)
        self.assertIn(run.selected_move, case.candidates)
        self.assertEqual(2, len(run.ranked_moves))

    def test_native_review_uses_independent_candidate_calls(self) -> None:
        case = native_search_baseline.load_case(MOVE_21_REVIEW_FIXTURE)
        run = native_search_baseline.run_native_full_window_review(
            case,
            (1, 2, 3),
            node_limit=None,
            threat_extension_depth=2,
            branch_candidate_limit=8,
        )

        self.assertEqual(("K7", "H7", "K8"), case.candidates)
        self.assertEqual(("H7", "K8", "H7"), run.leader_history)
        self.assertEqual(3, run.completed_depth)
        self.assertEqual("requested_depths_completed", run.stop_reason)
        self.assertTrue(all(layer.completed for layer in run.layers))
        self.assertTrue(
            all(
                len(candidate.principal_variation) >= 1
                for layer in run.layers
                for candidate in layer.candidates
            )
        )

    def test_native_review_rejects_an_incomplete_layer(self) -> None:
        case = native_search_baseline.load_case(MOVE_21_REVIEW_FIXTURE)
        run = native_search_baseline.run_native_full_window_review(
            case,
            (8, 9),
            node_limit=10,
            threat_extension_depth=2,
            branch_candidate_limit=8,
        )

        self.assertEqual((), run.leader_history)
        self.assertEqual(0, run.completed_depth)
        self.assertEqual("node_limit", run.stop_reason)
        self.assertEqual(1, len(run.layers))
        self.assertFalse(run.layers[0].completed)
        self.assertIsNone(run.layers[0].leader)
        self.assertTrue(
            any(
                candidate.status == "node_limit"
                for candidate in run.layers[0].candidates
            )
        )

    def test_full_window_exposes_parameters_and_bounded_trace(self) -> None:
        case = native_search_baseline.load_case()
        run = native_search_baseline.run_full_window_candidate(
            case,
            "F7",
            2,
            threat_extension_depth=0,
            branch_candidate_limit=12,
            candidate_trace_limit=1,
            candidate_sample_limit=2,
            leaf_trace_limit=1,
        )

        self.assertEqual(0, run.threat_extension_depth)
        self.assertEqual(12, run.branch_candidate_limit)
        self.assertGreaterEqual(run.extensions, 0)
        self.assertLessEqual(len(run.candidate_layers), 1)
        self.assertTrue(
            all(
                len(layer.sample_moves) <= 2
                for layer in run.candidate_layers
            )
        )
        self.assertLessEqual(len(run.leaf_trace), 1)

    def test_default_full_window_avoids_the_trace_subclass(self) -> None:
        case = native_search_baseline.load_case()
        with patch.object(
            native_search_baseline,
            "_TracingSearchAI",
        ) as tracer:
            run = native_search_baseline.run_full_window_candidate(
                case,
                "F7",
                1,
            )

        tracer.assert_not_called()
        self.assertEqual((), run.candidate_layers)
        self.assertEqual((), run.leaf_trace)

    def test_dynamic_pair_passes_the_explicit_branch_override(self) -> None:
        case = native_search_baseline.load_case()
        captured: dict[str, object] = {}

        def unavailable_review(
            _self: SearchAI,
            *_args: object,
            **kwargs: object,
        ) -> None:
            captured.update(kwargs)
            return None

        with patch.object(
            SearchAI,
            "_run_dynamic_pair_review",
            new=unavailable_review,
        ):
            run = native_search_baseline.run_dynamic_pair(
                case,
                2,
                review_budget_seconds=1.0,
                quiet_frontier_extension=False,
                threat_extension_depth=2,
                branch_candidate_limit=13,
            )

        self.assertEqual(13, captured["branch_candidate_limit_override"])
        self.assertEqual(13, run.branch_candidate_limit)
        self.assertEqual(4, run.threat_extension_depth)


if __name__ == "__main__":
    unittest.main()
