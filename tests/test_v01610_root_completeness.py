from __future__ import annotations

import unittest
from unittest.mock import patch

from engine import root_candidates
from engine.ai import RootSafetyCandidateAnalysis
from engine.board import BLACK, WHITE, Board
from engine.game import parse_move
from engine.search import SearchAI
from engine.search_types import RootResult, RootSafetyProbeResult
from engine.threats import ThreatFrontier, ThreatKind


YIXIN_MOVE_7_PREFIX = "H8 H9 G8 G7 F8 I8".split()


def replay(coordinates: list[str]) -> Board:
    board = Board()
    for index, coordinate in enumerate(coordinates):
        board.place(
            *parse_move(coordinate, board.size),
            BLACK if index % 2 == 0 else WHITE,
        )
    return board


def quiet_frontier(
    move: tuple[int, int],
    *,
    count: int,
    rank: int = 40,
) -> ThreatFrontier:
    return ThreatFrontier(
        gain_move=move,
        kind=ThreatKind.QUIET,
        continuations=tuple((move[0] + 1, index) for index in range(count)),
        continuation_kinds=(ThreatKind.OPEN_THREE,) * count,
        continuation_ranks=(rank,) * count,
    )


class TestBroadQuietAttackTies(unittest.TestCase):
    def test_one_exact_tie_can_follow_the_primary_slot(self) -> None:
        first = quiet_frontier((1, 1), count=8)
        tied = quiet_frontier((2, 2), count=8)
        weaker = quiet_frontier((3, 3), count=7)

        selected = root_candidates.broad_quiet_attack_frontier_moves(
            frontiers=(first, tied, weaker),
            minimum_rank=40,
            minimum_continuations=7,
            minimum_total_rank=280,
            limit=1,
            tied_limit=1,
        )

        self.assertEqual([first.gain_move, tied.gain_move], selected)

    def test_v0169_move_seven_keeps_both_equal_broad_hubs(self) -> None:
        board = replay(YIXIN_MOVE_7_PREFIX)
        g9 = parse_move("G9", board.size)
        f9 = parse_move("F9", board.size)
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
        ai._begin_move_search()

        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )

        self.assertEqual(
            root_candidates.RootCandidateMode.FRONTIER_DEFENSE,
            plan.mode,
        )
        self.assertEqual((g9, f9), ai._root_broad_quiet_attack_frontiers)
        self.assertIn(f9, plan.moves)
        self.assertIn(
            root_candidates.CandidateSource.BROAD_QUIET_ATTACK,
            ai._root_candidate_sources[f9],
        )
        self.assertEqual(10, len(plan.moves))
        self.assertLessEqual(len(plan.moves), ai.config.root_candidate_limit)


class TestDynamicReviewPairReserve(unittest.TestCase):
    def test_first_pair_runs_before_optional_pool_structure_scoring(self) -> None:
        board = Board()
        leader = (7, 7)
        challenger = (7, 8)
        later = (8, 7)
        result = RootResult(
            move=leader,
            score=100,
            principal_variation=(leader,),
            ranked_moves=(
                (leader, 100),
                (challenger, 90),
                (later, 80),
            ),
        )
        probe = RootSafetyProbeResult(
            trigger="dynamic_remaining_review",
            pvs_gap=10,
            main_rank_stable=True,
            completed_depth=5,
            nodes=1,
            candidates=(
                RootSafetyCandidateAnalysis(leader, 100),
                RootSafetyCandidateAnalysis(challenger, 90),
            ),
            leader_history=(leader, leader, leader),
            approved_move=leader,
            selection_basis="equal_window",
            requested_budget_seconds=0.5,
        )
        ai = SearchAI(BLACK, time_limit_seconds=60.0)
        ai._begin_move_search()

        with (
            patch.object(
                ai,
                "_dynamic_review_budget_seconds",
                return_value=0.6,
            ),
            patch(
                "engine.search.root_review.review_pool",
                return_value=[leader, challenger, later],
            ),
            patch(
                "engine.search.root_review.finalists",
                return_value=[leader, challenger, later],
            ),
            patch.object(
                ai,
                "_run_dynamic_pair_review",
                return_value=probe,
            ) as run_pair,
            patch.object(
                ai,
                "_frontier_balance_after_move",
                return_value=0,
            ) as structure_score,
        ):
            actual = ai._maybe_run_dynamic_root_review(
                board,
                result,
                [leader, challenger, later],
                completed_depth=5,
        )

        self.assertIsNotNone(actual)
        self.assertGreaterEqual(run_pair.call_count, 1)
        first_pair = run_pair.call_args_list[0]
        self.assertEqual(challenger, first_pair.args[2])
        self.assertGreaterEqual(first_pair.kwargs["budget_seconds"], 0.5)
        structure_score.assert_not_called()


if __name__ == "__main__":
    unittest.main()
