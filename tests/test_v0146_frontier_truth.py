from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from engine.ai import ProofCandidateAnalysis
from engine.board import BLACK, WHITE, Board
from engine.game import format_move, parse_move
from engine.proof_search import ProofState
from engine.search import SearchAI
from engine.search_types import MATE_SCORE, RootResult


SELFPLAY_MOVE_EIGHT_PREFIX = "H8 I7 H6 H9 H7 H4 G7".split()
YIXIN_MOVE_NINE_PREFIX = "H8 H9 G9 I7 F10 G8 F7 I6".split()


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


class TestV0146FrontierTruthOrder(unittest.TestCase):
    def test_real_move_eight_prefers_opponent_pressure_before_false_pvs(
        self,
    ) -> None:
        for coordinates, expected, rejected in (
            (SELFPLAY_MOVE_EIGHT_PREFIX, "F8", "I9"),
            (
                [
                    rotate_coordinate_180(coordinate)
                    for coordinate in SELFPLAY_MOVE_EIGHT_PREFIX
                ],
                rotate_coordinate_180("F8"),
                rotate_coordinate_180("I9"),
            ),
        ):
            with self.subTest(expected=expected):
                board = build_board(coordinates)
                ai = SearchAI(
                    WHITE,
                    max_depth=8,
                    time_limit_seconds=None,
                )
                ai._begin_move_search()
                plan = ai._prepare_root_candidate_plan(
                    board,
                    board.get_legal_moves(),
                )
                search_order = [format_move(*move) for move in plan.moves]
                truth_order = [
                    format_move(*move)
                    for move in ai._root_frontier_priority
                ]

                self.assertLess(
                    truth_order.index(expected),
                    truth_order.index(rejected),
                )
                self.assertEqual(expected, truth_order[0])
                self.assertIn(expected, search_order)

    def test_false_mate_quarantine_survives_unknown_tiebreak(self) -> None:
        board = build_board(SELFPLAY_MOVE_EIGHT_PREFIX)
        ai = SearchAI(
            WHITE,
            max_depth=8,
            time_limit_seconds=None,
        )
        ai._begin_move_search()
        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )
        moves = {
            coordinate: parse_move(coordinate, board.size)
            for coordinate in ("F8", "F6", "I9", "I5")
        }
        contaminated = RootResult(
            move=moves["I9"],
            score=-MATE_SCORE + 7,
            principal_variation=(moves["I9"],),
            ranked_moves=(
                (moves["I9"], -MATE_SCORE + 7),
                (moves["I5"], -MATE_SCORE + 7),
                (moves["F6"], -MATE_SCORE + 7),
                (moves["F8"], -MATE_SCORE + 5),
            ),
            ranked_variations=tuple(
                (move, score, (move,))
                for move, score in (
                    (moves["I9"], -MATE_SCORE + 7),
                    (moves["I5"], -MATE_SCORE + 7),
                    (moves["F6"], -MATE_SCORE + 7),
                    (moves["F8"], -MATE_SCORE + 5),
                )
            ),
        )
        heuristic = {
            moves["I9"]: 600,
            moves["I5"]: 600,
            moves["F6"]: -400,
            moves["F8"]: -1_200,
        }
        ai._proof_candidates = tuple(
            ProofCandidateAnalysis(
                move=move,
                state=ProofState.UNKNOWN.value,
                completed=False,
                nodes=1,
                elapsed_seconds=1.0,
                cutoff_reason="deadline",
                threat_risk=4_613_004,
            )
            for move in moves.values()
        )
        ai.config = replace(ai.config, max_depth=1)

        with (
            patch.object(ai, "_search_root", return_value=contaminated),
            patch.object(
                ai,
                "_heuristic_root_score",
                side_effect=lambda _board, move: heuristic[move],
            ),
        ):
            outcome = ai._run_iterative_root_search(
                board,
                plan.moves,
                fallback_move=plan.moves[0],
                preserve_frontier_order=True,
                allow_near_loss_expansion=False,
                defense_probe=None,
            )

        self.assertEqual(moves["F8"], outcome.result.move)

    def test_yixin_move_nine_keeps_g10_as_bounded_candidate(self) -> None:
        board = build_board(YIXIN_MOVE_NINE_PREFIX)
        ai = SearchAI(
            BLACK,
            max_depth=8,
            time_limit_seconds=None,
        )
        ai._begin_move_search()
        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )
        coordinates = {format_move(*move) for move in plan.moves}

        self.assertIn("G10", coordinates)
        self.assertLessEqual(len(plan.moves), ai.config.root_candidate_limit)


if __name__ == "__main__":
    unittest.main()
