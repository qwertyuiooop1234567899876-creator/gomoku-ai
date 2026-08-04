from __future__ import annotations

import time
import unittest

from engine import root_review
from engine.ai import (
    DecisionAnalysis,
    RootSafetyCandidateAnalysis,
    SearchPhaseTiming,
)
from engine.board import BLACK, WHITE, Board
from engine.game import format_move, parse_move
from engine.search import SearchAI
from engine.search_types import RootResult, RootSafetyProbeResult, SearchConfig
from engine.time_manager import TimeManager


YIXIN_MOVES = "H8 H9 G9 I7 F10 G8 E11 D12 F7 I6 F9 F8 I8 H6".split()
SELFPLAY_MOVES = (
    "H8 I7 H6 H7 G7 F8 F6 I9 I8 G6 E5 D4 J8 K8 I5 J7 "
    "I6 K7 L7 K9 K6 K10 K11"
).split()


def build_board(coordinates: list[str]) -> Board:
    board = Board()
    for index, coordinate in enumerate(coordinates):
        board.place(
            *parse_move(coordinate, board.size),
            BLACK if index % 2 == 0 else WHITE,
        )
    return board


def rotate_coordinate_180(coordinate: str, size: int = 15) -> str:
    row, column = parse_move(coordinate, size)
    return format_move(size - 1 - row, size - 1 - column)


def rotated(coordinates: list[str]) -> list[str]:
    return [rotate_coordinate_180(item) for item in coordinates]


class TestV0148DynamicReview(unittest.TestCase):
    def test_dynamic_review_never_spends_final_proof_reserve(self) -> None:
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=60.0)
        ai._time = TimeManager(
            started_at=time.perf_counter() - 52.0,
            time_limit_seconds=60.0,
        )

        self.assertEqual(0.0, ai._dynamic_review_budget_seconds())

    def test_phase_timings_are_serialized_for_performance_audit(self) -> None:
        analysis = DecisionAnalysis(
            selected_move=(7, 7),
            reason="test",
            candidate_count=2,
            phase_timings=(SearchPhaseTiming("main_pvs", 1.25),),
        )

        self.assertEqual(
            [{"phase": "main_pvs", "elapsed_seconds": 1.25}],
            analysis.to_dict()["phase_timings"],
        )

    def test_offensive_bridge_adds_d10_and_rotated_equivalent(self) -> None:
        for coordinates, expected in (
            (YIXIN_MOVES[:12], "D10"),
            (
                rotated(YIXIN_MOVES[:12]),
                rotate_coordinate_180("D10"),
            ),
        ):
            with self.subTest(expected=expected):
                board = build_board(coordinates)
                ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
                ai._begin_move_search()
                plan = ai._prepare_root_candidate_plan(
                    board,
                    board.get_legal_moves(),
                )
                self.assertIn(parse_move(expected, board.size), plan.moves)
                self.assertLessEqual(
                    len(plan.moves),
                    ai.config.root_candidate_limit,
                )

    def test_frontier_balance_separates_recorded_horizon_pairs(self) -> None:
        for coordinates, player, preferred, rejected in (
            (YIXIN_MOVES[:10], BLACK, "G10", "F9"),
            (YIXIN_MOVES[:14], BLACK, "G6", "G7"),
            (
                rotated(YIXIN_MOVES[:10]),
                BLACK,
                rotate_coordinate_180("G10"),
                rotate_coordinate_180("F9"),
            ),
            (
                rotated(YIXIN_MOVES[:14]),
                BLACK,
                rotate_coordinate_180("G6"),
                rotate_coordinate_180("G7"),
            ),
        ):
            with self.subTest(preferred=preferred):
                board = build_board(coordinates)
                ai = SearchAI(player, max_depth=8, time_limit_seconds=None)
                ai._begin_move_search()
                preferred_score = ai._frontier_balance_after_move(
                    board,
                    parse_move(preferred, board.size),
                )
                rejected_score = ai._frontier_balance_after_move(
                    board,
                    parse_move(rejected, board.size),
                )
                self.assertGreaterEqual(
                    preferred_score - rejected_score,
                    ai.config.root_dynamic_review_structure_margin,
                )

    def test_equal_probe_tie_uses_frontier_balance_without_proof(self) -> None:
        config = SearchConfig()
        rejected = (1, 1)
        preferred = (2, 2)
        result = RootResult(
            rejected,
            100,
            (rejected,),
            ((rejected, 100), (preferred, 90)),
        )
        probe = RootSafetyProbeResult(
            trigger="dynamic_remaining_review",
            pvs_gap=10,
            main_rank_stable=True,
            completed_depth=5,
            nodes=10,
            candidates=(
                RootSafetyCandidateAnalysis(rejected, -100_000_000),
                RootSafetyCandidateAnalysis(preferred, -100_000_000),
            ),
            leader_history=(rejected, rejected, rejected),
        )

        move, basis = root_review.approve_move(
            config,
            result,
            probe,
            {rejected: 1_000, preferred: 5_000},
        )

        self.assertEqual(preferred, move)
        self.assertEqual("frontier_balance", basis)

    def test_stable_non_tied_probe_remains_authoritative(self) -> None:
        config = SearchConfig()
        rejected = (1, 1)
        preferred = (2, 2)
        result = RootResult(
            rejected,
            100,
            (rejected,),
            ((rejected, 100), (preferred, 90)),
        )
        probe = RootSafetyProbeResult(
            trigger="dynamic_remaining_review",
            pvs_gap=10,
            main_rank_stable=True,
            completed_depth=5,
            nodes=10,
            candidates=(
                RootSafetyCandidateAnalysis(preferred, 20_000),
                RootSafetyCandidateAnalysis(rejected, 0),
            ),
            leader_history=(preferred, preferred, preferred),
        )

        move, basis = root_review.approve_move(
            config,
            result,
            probe,
            {rejected: 2_000, preferred: 0},
        )

        self.assertEqual(preferred, move)
        self.assertEqual("equal_window", basis)

    def test_incomplete_depth_four_can_use_material_structure_gap(self) -> None:
        config = SearchConfig()
        rejected = (1, 1)
        preferred = (2, 2)
        result = RootResult(
            rejected,
            100,
            (rejected,),
            ((rejected, 100), (preferred, 90)),
        )
        probe = RootSafetyProbeResult(
            trigger="dynamic_remaining_review",
            pvs_gap=10,
            main_rank_stable=True,
            completed_depth=4,
            nodes=10,
            candidates=(
                RootSafetyCandidateAnalysis(rejected, 20_000),
                RootSafetyCandidateAnalysis(preferred, 10_000),
            ),
            leader_history=(preferred, rejected, rejected),
        )

        move, basis = root_review.approve_move(
            config,
            result,
            probe,
            {rejected: 1_000, preferred: 5_000},
        )

        self.assertEqual(preferred, move)
        self.assertEqual("frontier_balance", basis)

    def test_very_large_structure_gap_can_override_stable_horizon(self) -> None:
        config = SearchConfig()
        rejected = (1, 1)
        preferred = (2, 2)
        result = RootResult(
            rejected,
            100,
            (rejected,),
            ((rejected, 100), (preferred, 90)),
        )
        probe = RootSafetyProbeResult(
            trigger="dynamic_remaining_review",
            pvs_gap=10,
            main_rank_stable=True,
            completed_depth=6,
            nodes=10,
            candidates=(
                RootSafetyCandidateAnalysis(rejected, 83_100),
                RootSafetyCandidateAnalysis(preferred, -100_000_000),
            ),
            leader_history=(rejected, rejected, rejected),
        )

        move, basis = root_review.approve_move(
            config,
            result,
            probe,
            {rejected: 0, preferred: 11_000},
        )

        self.assertEqual(preferred, move)
        self.assertEqual("frontier_balance", basis)

    def test_selfplay_main_competitor_is_kept_in_finalists(self) -> None:
        board = build_board(SELFPLAY_MOVES)
        ai = SearchAI(WHITE, max_depth=8, time_limit_seconds=None)
        ai._begin_move_search()
        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )
        moves = {
            name: parse_move(name, board.size)
            for name in ("J9", "L9", "H9", "L8")
        }
        result = RootResult(
            moves["J9"],
            81_300,
            (moves["J9"],),
            (
                (moves["J9"], 81_300),
                (moves["L9"], 62_000),
                (moves["H9"], 51_100),
                (moves["L8"], -11_000),
            ),
        )
        pool = [move for move, _score in result.ranked_moves]
        finalists = root_review.finalists(
            ai.config,
            result,
            pool,
            {move: index for index, move in enumerate(pool)},
            preferred_moves=ai._root_quiet_prevention,
        )

        self.assertIn(moves["L9"], finalists)
        self.assertLessEqual(
            len(finalists),
            ai.config.root_dynamic_review_finalist_limit,
        )
        self.assertIn(moves["L9"], plan.moves)


if __name__ == "__main__":
    unittest.main()
