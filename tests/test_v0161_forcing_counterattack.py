from __future__ import annotations

import unittest

from engine import root_candidates
from engine.board import BLACK, WHITE, Board
from engine.game import parse_move
from engine.search import SearchAI


MOVE15_PREFIX = (
    "H8 H9 G9 I7 E11 F10 G8 I8 G10 G7 G11 G12 F11 H11"
)


def replay_prefix(text: str) -> Board:
    board = Board()
    for index, coordinate in enumerate(text.split()):
        player = BLACK if index % 2 == 0 else WHITE
        board.place(*parse_move(coordinate), player)
    return board


class TestV0161ForcingCounterattack(unittest.TestCase):
    def test_mandatory_root_reserves_direct_and_forcing_sources(self) -> None:
        defenses = ((1, 1), (1, 2), (1, 3), (1, 4))
        counterattacks = ((2, 1), (2, 2))

        moves = root_candidates.mandatory_defense_moves(
            defense_moves=defenses,
            forcing_counterattack_moves=counterattacks,
            limit=4,
        )

        self.assertEqual(4, len(moves))
        self.assertTrue(set(defenses).intersection(moves))
        self.assertTrue(set(counterattacks).intersection(moves))

    def test_latest_yixin_loss_keeps_tempo_defenses_in_root(self) -> None:
        board = replay_prefix(MOVE15_PREFIX)
        ai = SearchAI(
            player=BLACK,
            max_depth=8,
            time_limit_seconds=None,
        )
        ai._begin_move_search()

        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )
        sources = {entry.move: entry.sources for entry in plan.entries}
        direct_block = parse_move("I10")
        tempo_defenses = {parse_move("D11"), parse_move("C11")}

        self.assertIn(direct_block, plan.moves)
        self.assertTrue(tempo_defenses.issubset(plan.moves))
        self.assertIn(
            root_candidates.CandidateSource.MANDATORY_DEFENSE,
            sources[direct_block],
        )
        for move in tempo_defenses:
            self.assertIn(
                root_candidates.CandidateSource.FORCING_COUNTERATTACK,
                sources[move],
            )


if __name__ == "__main__":
    unittest.main()
