from __future__ import annotations

import unittest

from engine import root_candidates
from engine.board import BLACK, WHITE, Board
from engine.game import parse_move
from engine.search import SearchAI
from engine.search_types import SearchConfig


SELFPLAY_MOVE_17_PREFIX = (
    "H8 G7 I7 G9 G8 F8 H6 H10 E7 I11 J12 J8 H7 H9 H5 H4"
)
YIXIN_MOVE_21_PREFIX = (
    "H8 H9 G7 I9 G9 I7 G8 G6 I8 F8 F10 G10 E12 E11 H10 J8 "
    "H6 K7 F9 J7"
)


def replay_prefix(text: str) -> Board:
    board = Board()
    for index, coordinate in enumerate(text.split()):
        player = BLACK if index % 2 == 0 else WHITE
        board.place(*parse_move(coordinate, board.size), player)
    return board


class TestV0163MandatoryActiveCounterattack(unittest.TestCase):
    def test_active_counterattack_limit_rejects_negative_values(self) -> None:
        with self.assertRaises(ValueError):
            SearchConfig(root_mandatory_active_counterattack_limit=-1)

    def test_mandatory_root_reserves_each_defensive_source(self) -> None:
        direct = ((1, 1), (1, 2), (1, 3))
        forcing = ((2, 1), (2, 2))
        active = ((3, 1), (3, 2))

        moves = root_candidates.mandatory_defense_moves(
            defense_moves=direct,
            forcing_counterattack_moves=forcing,
            active_counterattack_moves=active,
            limit=3,
        )

        self.assertEqual(3, len(moves))
        self.assertTrue(set(direct).intersection(moves))
        self.assertTrue(set(forcing).intersection(moves))
        self.assertTrue(set(active).intersection(moves))

    def test_selfplay_singleton_defense_keeps_active_i6(self) -> None:
        board = replay_prefix(SELFPLAY_MOVE_17_PREFIX)
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
        ai._begin_move_search()

        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )
        sources = {entry.move: entry.sources for entry in plan.entries}
        direct = parse_move("I9", board.size)
        active = parse_move("I6", board.size)

        self.assertIn(direct, plan.moves)
        self.assertIn(active, plan.moves)
        self.assertIn(
            root_candidates.CandidateSource.MANDATORY_DEFENSE,
            sources[direct],
        )
        self.assertIn(
            root_candidates.CandidateSource.ACTIVE_COUNTERATTACK,
            sources[active],
        )
        self.assertLessEqual(len(plan.moves), ai.config.root_candidate_limit)

    def test_yixin_singleton_defense_keeps_active_h7(self) -> None:
        board = replay_prefix(YIXIN_MOVE_21_PREFIX)
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
        ai._begin_move_search()

        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )
        sources = {entry.move: entry.sources for entry in plan.entries}
        direct = parse_move("L7", board.size)
        active = parse_move("H7", board.size)

        self.assertIn(direct, plan.moves)
        self.assertIn(active, plan.moves)
        self.assertIn(
            root_candidates.CandidateSource.MANDATORY_DEFENSE,
            sources[direct],
        )
        self.assertIn(
            root_candidates.CandidateSource.ACTIVE_COUNTERATTACK,
            sources[active],
        )
        self.assertLessEqual(len(plan.moves), ai.config.root_candidate_limit)


if __name__ == "__main__":
    unittest.main()
