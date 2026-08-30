from __future__ import annotations

import unittest

from engine import root_candidates
from engine.board import BLACK, WHITE, Board
from engine.evaluator import ThreatProfile
from engine.game import parse_move
from engine.root_review import finalists, review_pool
from engine.search import SearchAI
from engine.search_types import RootResult


YIXIN_MOVE_31_PREFIX = (
    "H8 H9 G7 I9 G9 I7 G8 G6 I8 F8 F10 G10 E11 D12 G11 F11 "
    "E9 H12 D8 C7 E10 E12 D13 F9 E8 E7 F12 G13 C10 H11"
).split()
I4_PROOF_RISK_PREFIX = (
    "H8 I7 I6 H7 G7 G6 I8 J8 H6 F8 K3 K6 F6 E5 J4 I5 J7 H5"
).split()


def replay(coordinates: list[str]) -> Board:
    board = Board()
    for index, coordinate in enumerate(coordinates):
        board.place(
            *parse_move(coordinate, board.size),
            BLACK if index % 2 == 0 else WHITE,
        )
    return board


class TestDirectPressurePrevention(unittest.TestCase):
    def test_single_four_pressure_is_bounded_and_source_aware(self) -> None:
        direct = (3, 3)
        profiles = {
            direct: ThreatProfile(four_directions=1),
            (4, 4): ThreatProfile(open_three_directions=1),
        }

        selected = root_candidates.direct_pressure_prevention_moves(
            profiles=profiles,
            ordered_moves=((4, 4), direct),
            limit=1,
        )

        self.assertEqual([direct], selected)
        self.assertEqual(
            [],
            root_candidates.direct_pressure_prevention_moves(
                profiles={(4, 4): profiles[(4, 4)]},
                ordered_moves=((4, 4),),
                limit=1,
            ),
        )

        frontier = ((1, 1),)
        ordinary = tuple((2, column) for column in range(12))
        without_direct = root_candidates.frontier_defense_moves(
            frontier_moves=frontier,
            ordinary_moves=ordinary,
            counterattack_moves=(),
            limit=6,
        )
        with_direct = root_candidates.frontier_defense_moves(
            frontier_moves=frontier,
            ordinary_moves=ordinary,
            counterattack_moves=(),
            direct_pressure_prevention_moves=selected,
            limit=6,
        )

        self.assertEqual(6, len(with_direct))
        self.assertIn(direct, with_direct)
        self.assertEqual(
            without_direct,
            root_candidates.frontier_defense_moves(
                frontier_moves=frontier,
                ordinary_moves=ordinary,
                counterattack_moves=(),
                direct_pressure_prevention_moves=(),
                limit=6,
            ),
        )

    def test_lane_does_not_expand_when_no_ordinary_root_was_admitted(
        self,
    ) -> None:
        board = replay(I4_PROOF_RISK_PREFIX)
        ai = SearchAI(BLACK, max_depth=2, time_limit_seconds=4.0)
        ai._begin_move_search()

        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )

        self.assertEqual(
            [parse_move(name, board.size) for name in ("J5", "I4")],
            plan.moves,
        )
        self.assertEqual((), ai._root_pressure_prevention)

    def test_recorded_i12_enters_root_and_critical_review(self) -> None:
        board = replay(YIXIN_MOVE_31_PREFIX)
        target = parse_move("I12", board.size)
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
        self.assertIn(target, plan.moves)
        self.assertIn(
            root_candidates.CandidateSource.PRESSURE_PREVENTION,
            ai._root_candidate_sources[target],
        )
        self.assertIn(target, ai._root_pressure_prevention)
        self.assertLessEqual(len(plan.moves), ai.config.root_candidate_limit)

        ranked = tuple(
            (move, 10_000 - index * 100)
            for index, move in enumerate(plan.moves)
        )
        result = RootResult(
            move=ranked[0][0],
            score=ranked[0][1],
            principal_variation=(ranked[0][0],),
            ranked_moves=ranked,
        )
        critical_groups = ai._critical_root_review_groups(plan.moves)
        critical_moves = root_candidates.merge_unique(*critical_groups)
        pool = review_pool(
            ai.config,
            result,
            (),
            critical_moves=critical_moves,
            active_moves=(),
            quiet_moves=(),
            offensive_moves=(),
        )
        selected = finalists(
            ai.config,
            result,
            pool,
            {move: score for move, score in ranked},
            critical_groups=critical_groups,
        )

        self.assertIn(target, critical_moves)
        self.assertIn(target, selected)


if __name__ == "__main__":
    unittest.main()
