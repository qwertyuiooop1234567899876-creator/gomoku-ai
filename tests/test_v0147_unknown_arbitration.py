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


SELFPLAY_MOVES = (
    "H8 I7 H6 H7 G7 F8 F6 I9 E5 D4 G6 I6 I5 I8 I10 G8 "
    "J5 F9 E10 F10 F11 H9 G9 J7"
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


class TestV0147UnknownArbitration(unittest.TestCase):
    def test_material_proof_risk_can_correct_bounded_defense_probe(
        self,
    ) -> None:
        for coordinates, expected, rejected in (
            (SELFPLAY_MOVES[:16], "F9", "J5"),
            (
                rotated(SELFPLAY_MOVES[:16]),
                rotate_coordinate_180("F9"),
                rotate_coordinate_180("J5"),
            ),
        ):
            with self.subTest(expected=expected):
                board = build_board(coordinates)
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
                moves = {
                    coordinate: parse_move(coordinate, board.size)
                    for coordinate in (expected, rejected)
                }
                root = RootResult(
                    move=moves[rejected],
                    score=10_800,
                    principal_variation=(moves[rejected],),
                    ranked_moves=(
                        (moves[rejected], 10_800),
                        (moves[expected], 9_800),
                    ),
                    ranked_variations=(
                        (
                            moves[rejected],
                            10_800,
                            (moves[rejected],),
                        ),
                        (
                            moves[expected],
                            9_800,
                            (moves[expected],),
                        ),
                    ),
                )
                risks = {
                    moves[rejected]: 3_806_002,
                    moves[expected]: 2_204_402,
                }
                ai._proof_candidates = tuple(
                    ProofCandidateAnalysis(
                        move=move,
                        state=ProofState.UNKNOWN.value,
                        completed=False,
                        nodes=1,
                        elapsed_seconds=1.0,
                        cutoff_reason="deadline",
                        threat_risk=risk,
                    )
                    for move, risk in risks.items()
                )
                ai.config = replace(ai.config, max_depth=1)

                with patch.object(ai, "_search_root", return_value=root):
                    outcome = ai._run_iterative_root_search(
                        board,
                        plan.moves,
                        fallback_move=plan.moves[0],
                        preserve_frontier_order=False,
                        allow_near_loss_expansion=False,
                        defense_probe=plan.defense_probe,
                    )

                self.assertEqual(moves[expected], outcome.result.move)

    def test_counterattack_truth_precedes_false_static_leader(self) -> None:
        for coordinates, expected, rejected in (
            (SELFPLAY_MOVES[:19], "H9", "F10"),
            (
                rotated(SELFPLAY_MOVES[:19]),
                rotate_coordinate_180("H9"),
                rotate_coordinate_180("F10"),
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
                truth = [
                    format_move(*move)
                    for move in ai._root_frontier_priority
                ]

                self.assertIn(expected, truth)
                self.assertIn(rejected, truth)
                self.assertLess(
                    truth.index(expected),
                    truth.index(rejected),
                )
                self.assertTrue(plan.preserve_frontier_order)

    def test_false_mate_quarantine_uses_counterattack_truth_band(
        self,
    ) -> None:
        board = build_board(SELFPLAY_MOVES[:19])
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
        f10 = parse_move("F10", board.size)
        h9 = parse_move("H9", board.size)
        g9 = parse_move("G9", board.size)
        contaminated = RootResult(
            move=f10,
            score=MATE_SCORE - 5,
            principal_variation=(f10,),
            ranked_moves=(
                (f10, MATE_SCORE - 5),
                (h9, MATE_SCORE - 7),
                (g9, MATE_SCORE - 9),
            ),
            ranked_variations=(
                (f10, MATE_SCORE - 5, (f10,)),
                (h9, MATE_SCORE - 7, (h9,)),
                (g9, MATE_SCORE - 9, (g9,)),
            ),
        )
        heuristic = {
            f10: 79_200,
            h9: 51_300,
            g9: 51_300,
        }
        ai._proof_candidates = tuple(
            ProofCandidateAnalysis(
                move=move,
                state=ProofState.UNKNOWN.value,
                completed=False,
                nodes=1,
                elapsed_seconds=1.0,
                cutoff_reason="deadline",
                threat_risk=5_029_408,
            )
            for move in (f10, h9, g9)
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

        self.assertEqual(h9, outcome.result.move)

    def test_quiet_frontier_sibling_adds_k7_before_root_cap(self) -> None:
        for coordinates, expected in (
            (SELFPLAY_MOVES[:23], "K7"),
            (
                rotated(SELFPLAY_MOVES[:23]),
                rotate_coordinate_180("K7"),
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
                candidates = {
                    format_move(*move) for move in plan.moves
                }

                self.assertIn(expected, candidates)
                self.assertLessEqual(
                    len(plan.moves),
                    ai.config.root_candidate_limit,
                )

    def test_quiet_frontier_sibling_gets_deeper_equal_window_review(
        self,
    ) -> None:
        for coordinates, expected, rejected in (
            (SELFPLAY_MOVES[:23], "K7", "J7"),
            (
                rotated(SELFPLAY_MOVES[:23]),
                rotate_coordinate_180("K7"),
                rotate_coordinate_180("J7"),
            ),
        ):
            with self.subTest(expected=expected):
                board = build_board(coordinates)
                ai = SearchAI(
                    WHITE,
                    max_depth=8,
                    time_limit_seconds=60.0,
                )
                ai._begin_move_search()
                plan = ai._prepare_root_candidate_plan(
                    board,
                    board.get_legal_moves(),
                )
                expected_move = parse_move(expected, board.size)
                rejected_move = parse_move(rejected, board.size)
                root = RootResult(
                    move=rejected_move,
                    score=89_300,
                    principal_variation=(rejected_move,),
                    ranked_moves=(
                        (rejected_move, 89_300),
                        (expected_move, -11_900),
                    ),
                    ranked_variations=(
                        (rejected_move, 89_300, (rejected_move,)),
                        (expected_move, -11_900, (expected_move,)),
                    ),
                )

                probe = ai._run_quiet_sibling_probe(
                    board,
                    root,
                    plan.moves,
                    completed_depth=8,
                )

                self.assertIsNotNone(probe)
                assert probe is not None
                self.assertGreaterEqual(probe.completed_depth, 5)
                self.assertTrue(probe.rank_stable)
                self.assertEqual(expected_move, probe.best_move)


if __name__ == "__main__":
    unittest.main()
