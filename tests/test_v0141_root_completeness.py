from __future__ import annotations

import unittest

from engine.board import BLACK, Board
from engine.game import format_move, parse_move
from engine.search import SearchAI


BLACK_EIGHTEEN_PREFIX = """
H8 H9 G9 I7 F10 G8 F7 I6 F9 F8 I8 H7 G7 J7 I9 J5 K4 J10
""".split()

BLACK_THIRTY_PREFIX = """
H8 I7 G7 I9 I8 J8 H10 H6 K9 H9 G8 G9 F9 H7 F8 E8 F10 F7
G6 K7 G5 G4 F11 F12 G10 E10 J9 J7 L7 I10
""".split()


def rotate_coordinate_180(coordinate: str, size: int = 15) -> str:
    row, column = parse_move(coordinate, size)
    return format_move(size - 1 - row, size - 1 - column)


def build_board(coordinates: list[str]) -> Board:
    board = Board()
    for index, coordinate in enumerate(coordinates):
        board.place(
            *parse_move(coordinate, board.size),
            BLACK if index % 2 == 0 else 2,
        )
    return board


def board_state(board: Board) -> tuple[object, ...]:
    return (
        tuple(tuple(row) for row in board.grid),
        tuple(board.move_history),
        board.zobrist_hash,
        board.empty_count,
    )


class TestV0141RootCandidateCompleteness(unittest.TestCase):
    def _assert_kept_with_rotation(
        self,
        prefix: list[str],
        required_coordinate: str,
    ) -> None:
        for coordinates, required in (
            (prefix, required_coordinate),
            (
                [rotate_coordinate_180(item) for item in prefix],
                rotate_coordinate_180(required_coordinate),
            ),
        ):
            with self.subTest(required=required):
                board = build_board(coordinates)
                before = board_state(board)
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
                candidate_coordinates = {
                    format_move(*move) for move in plan.moves
                }

                self.assertIn(required, candidate_coordinates)
                self.assertLessEqual(
                    len(plan.moves),
                    ai.config.root_candidate_limit,
                )
                self.assertEqual(before, board_state(board))

    def test_black_nineteen_keeps_forcing_counter_defense(self) -> None:
        self._assert_kept_with_rotation(
            BLACK_EIGHTEEN_PREFIX,
            "F6",
        )

    def test_black_thirty_one_keeps_quiet_frontier_prevention(self) -> None:
        self._assert_kept_with_rotation(
            BLACK_THIRTY_PREFIX,
            "H4",
        )

    def test_initial_and_final_proof_share_one_total_cap(self) -> None:
        ai = SearchAI(
            BLACK,
            max_depth=1,
            time_limit_seconds=60.0,
        )
        ai._begin_move_search()

        self.assertEqual(8.0, ai._final_proof_reserve_seconds())
        self.assertEqual(7.0, ai._proof_budget_seconds())


if __name__ == "__main__":
    unittest.main()
