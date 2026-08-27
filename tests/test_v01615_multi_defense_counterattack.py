from __future__ import annotations

import unittest
from unittest.mock import patch

from engine import root_candidates
from engine.board import BLACK, WHITE, Board
from engine.evaluator import ThreatProfile
from engine.game import parse_move
from engine.search import SearchAI


YIXIN_MOVE_23_PREFIX = (
    "H8 H9 G8 F8 G9 G7 E9 F10 F9 E10 G10 G12 D9 C9 E8 H11 "
    "D7 C6 F7 G6 E6 F13"
)


def replay_prefix(text: str) -> Board:
    board = Board()
    for index, coordinate in enumerate(text.split()):
        player = BLACK if index % 2 == 0 else WHITE
        board.place(*parse_move(coordinate, board.size), player)
    return board


class TestV01615MultiDefenseCounterattack(unittest.TestCase):
    def _controlled_plan(
        self,
        direct_defenses: list[tuple[int, int]],
    ) -> tuple[root_candidates.RootCandidatePlan, tuple[int, int], tuple[int, int]]:
        board = Board()
        forcing_counterattack = (1, 0)
        quiet_counterattack = (1, 1)
        root_pool = [
            *direct_defenses,
            forcing_counterattack,
            quiet_counterattack,
        ]
        own_profiles = {
            forcing_counterattack: ThreatProfile(four_directions=1),
            quiet_counterattack: ThreatProfile(open_three_directions=1),
        }
        opponent_profiles = {
            move: ThreatProfile(immediate_win=True)
            for move in direct_defenses
        }
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
        ai._begin_move_search()

        with (
            patch.object(ai, "_root_profile_pool", return_value=root_pool),
            patch.object(ai, "_root_relevant_pool", return_value=root_pool),
            patch.object(
                ai,
                "_profile_moves_timed",
                side_effect=lambda _board, _moves, player: (
                    own_profiles if player == BLACK else opponent_profiles
                ),
            ),
            patch.object(ai, "_multi_threat_frontiers", return_value={}),
            patch.object(
                ai,
                "_order_specific_moves",
                side_effect=lambda _board, moves, *_args, **_kwargs: list(
                    moves
                ),
            ),
            patch.object(ai, "_run_defense_vct_probe", return_value=None),
        ):
            plan = ai._prepare_root_candidate_plan(board, root_pool)

        return plan, forcing_counterattack, quiet_counterattack

    def test_yixin_move_23_keeps_pressure_relevant_counterattack(self) -> None:
        board = replay_prefix(YIXIN_MOVE_23_PREFIX)
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
        ai._begin_move_search()

        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )
        sources = {entry.move: entry.sources for entry in plan.entries}
        direct_defenses = {parse_move("I10"), parse_move("E14")}
        counterattack = parse_move("C10")

        self.assertIs(
            root_candidates.RootCandidateMode.MANDATORY_DEFENSE,
            plan.mode,
        )
        self.assertTrue(direct_defenses.issubset(plan.moves))
        self.assertIn(counterattack, plan.moves)
        self.assertEqual(3, len(plan.moves))
        self.assertIn(
            root_candidates.CandidateSource.FORCING_COUNTERATTACK,
            sources[counterattack],
        )
        self.assertIsNotNone(plan.defense_probe)
        self.assertEqual(
            set(plan.moves),
            {candidate.move for candidate in plan.defense_probe.candidates},
        )

    def test_two_defenses_add_one_forcing_move_but_not_quiet_attack(self) -> None:
        direct_defenses = [(0, 0), (0, 1)]

        plan, forcing_counterattack, quiet_counterattack = (
            self._controlled_plan(direct_defenses)
        )

        self.assertEqual(
            [*direct_defenses, forcing_counterattack],
            plan.moves,
        )
        self.assertNotIn(quiet_counterattack, plan.moves)

    def test_three_direct_defenses_do_not_widen_the_root(self) -> None:
        direct_defenses = [(0, 0), (0, 1), (0, 2)]

        plan, forcing_counterattack, quiet_counterattack = (
            self._controlled_plan(direct_defenses)
        )

        self.assertEqual(direct_defenses, plan.moves)
        self.assertNotIn(forcing_counterattack, plan.moves)
        self.assertNotIn(quiet_counterattack, plan.moves)

    def test_certificate_intercepts_remain_singleton_only(self) -> None:
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)

        result = ai._mandatory_certificate_intercept_moves(
            Board(),
            forcing_moves=[(0, 0), (0, 1)],
            forcing_profiles={},
            opponent_frontiers=(object(),),
        )

        self.assertEqual([], result)


if __name__ == "__main__":
    unittest.main()
