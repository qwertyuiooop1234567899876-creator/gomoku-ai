from __future__ import annotations

import unittest
from unittest.mock import patch

from engine import root_candidates, root_review
from engine.ai import RootSafetyCandidateAnalysis
from engine.board import BLACK, WHITE, Board
from engine.game import parse_move
from engine.search import SearchAI
from engine.search_types import RootResult, RootSafetyProbeResult, SearchConfig


SELFPLAY_MOVE_17_HISTORY = (
    "H8 G7 I7 G9 G8 H9 I8 J8 F9 I6 E8 F8 E7 E10 H6 H10"
).split()
YIXIN_MOVE_21_HISTORY = (
    "H8 H9 G7 I9 G9 I7 G8 G6 I8 F8 F10 E11 G11 G10 H12 E9 "
    "J8 K8 E10 I6"
).split()


def build_board(coordinates: list[str]) -> Board:
    board = Board()
    for index, coordinate in enumerate(coordinates):
        board.place(
            *parse_move(coordinate, board.size),
            BLACK if index % 2 == 0 else WHITE,
        )
    return board


class TestV0162RecordDriven(unittest.TestCase):
    def test_single_frontier_root_membership_ignores_history_noise(self) -> None:
        board = build_board(YIXIN_MOVE_21_HISTORY)
        target = parse_move("H5")
        clean_ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
        clean_ai._begin_move_search()
        clean_plan = clean_ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
        for move in board.get_legal_moves():
            if move != target:
                ai._history_scores[(BLACK, move[0], move[1])] = 1_000_000

        ai._begin_move_search()
        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )

        self.assertIn(target, plan.moves)
        self.assertEqual(set(clean_plan.moves), set(plan.moves))
        self.assertIn(
            root_candidates.CandidateSource.ORDINARY,
            ai._root_candidate_sources[target],
        )
        self.assertLessEqual(len(plan.moves), ai.config.root_candidate_limit)

    def test_active_counterattack_reaches_bounded_dynamic_review(self) -> None:
        board = build_board(SELFPLAY_MOVE_17_HISTORY)
        leader = parse_move("H5")
        strongest = parse_move("K9")
        pvs_sibling = parse_move("F10")
        frontier = parse_move("I9")
        active = parse_move("G10")
        ranked = (
            (leader, -10_900),
            (strongest, -79_900),
            (pvs_sibling, -98_200),
            (frontier, -99_000),
            (active, -100_700),
        )
        result = RootResult(leader, ranked[0][1], (leader,), ranked)
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
        ordinary = root_candidates.CandidateSource.ORDINARY
        ai._root_candidate_sources = {
            leader: frozenset(
                {root_candidates.CandidateSource.DUAL_FRONTIER_BRIDGE}
            ),
            strongest: frozenset(
                {root_candidates.CandidateSource.QUIET_PREVENTION}
            ),
            pvs_sibling: frozenset({ordinary}),
            frontier: frozenset(
                {ordinary, root_candidates.CandidateSource.THREAT_FRONTIER}
            ),
            active: frozenset(
                {ordinary, root_candidates.CandidateSource.ACTIVE_COUNTERATTACK}
            ),
        }
        ai._root_attack_priority = (active,)
        reviewed: list[tuple[int, int]] = []

        def run_pair(
            _board: Board,
            pair_result: RootResult,
            challenger: tuple[int, int],
            **_kwargs: object,
        ) -> RootSafetyProbeResult:
            reviewed.append(challenger)
            return RootSafetyProbeResult(
                trigger="dynamic_remaining_review",
                pvs_gap=0,
                main_rank_stable=True,
                completed_depth=5,
                nodes=1,
                candidates=(
                    RootSafetyCandidateAnalysis(pair_result.move, 0),
                    RootSafetyCandidateAnalysis(challenger, -1),
                ),
                leader_history=(pair_result.move, pair_result.move),
                approved_move=pair_result.move,
            )

        with (
            patch.object(ai, "_dynamic_review_budget_seconds", return_value=20.0),
            patch.object(ai, "_frontier_balance_after_move", return_value=0),
            patch.object(ai, "_run_dynamic_pair_review", side_effect=run_pair),
        ):
            probe = ai._maybe_run_dynamic_root_review(
                board,
                result,
                [move for move, _score in ranked],
                completed_depth=6,
            )

        self.assertIsNotNone(probe)
        self.assertIn(active, reviewed)
        self.assertLessEqual(
            len(reviewed) + 1,
            ai.config.root_dynamic_review_finalist_limit,
        )

    def test_critical_representative_does_not_consume_two_slots(self) -> None:
        leader = parse_move("H5")
        strongest = parse_move("K9")
        critical = parse_move("I9")
        active = parse_move("G10")
        result = RootResult(
            leader,
            -10_900,
            (leader,),
            (
                (leader, -10_900),
                (strongest, -79_900),
                (critical, -99_000),
                (active, -100_700),
            ),
        )
        pool = [move for move, _score in result.ranked_moves]

        finalists = root_review.finalists(
            SearchConfig(),
            result,
            pool,
            {move: 0 for move in pool},
            critical_groups=((critical,),),
            preferred_groups=((active,),),
        )

        self.assertEqual([leader, critical, strongest, active], finalists)
        self.assertEqual(len(finalists), len(set(finalists)))


if __name__ == "__main__":
    unittest.main()
