from __future__ import annotations

import unittest

from engine.board import BLACK, DIRECTIONS, EMPTY, WHITE, Board
from engine.evaluator import is_winning_move, other_side
from engine.game import format_move, parse_move
from engine.search import SearchAI


H11_POSITION = (
    "H8", "I7", "I6", "H7", "G7", "F6",
    "J7", "K8", "J8", "J9", "I8", "I10",
)


def build_board(coordinates: tuple[str, ...]) -> Board:
    board = Board()
    for index, coordinate in enumerate(coordinates):
        board.place(
            *parse_move(coordinate, board.size),
            BLACK if index % 2 == 0 else WHITE,
        )
    return board


def reference_quick_score(
    board: Board,
    move: tuple[int, int],
    player: int,
) -> int:
    """V0.14.2 ordering expression kept as an equivalence oracle."""
    row, column = move
    opponent = other_side(player)
    score = 0
    distance_weights = (0, 24, 8, 3, 1)
    for row_step, column_step in DIRECTIONS:
        for sign in (-1, 1):
            for distance in range(1, 5):
                neighbor_row = row + sign * distance * row_step
                neighbor_column = column + sign * distance * column_step
                if not board.is_inside(neighbor_row, neighbor_column):
                    break
                cell = board.grid[neighbor_row][neighbor_column]
                weight = distance_weights[distance]
                if cell == player:
                    score += weight * 3
                elif cell == opponent:
                    score += weight * 4
                elif cell == EMPTY:
                    score += weight
    center = (board.size - 1) / 2
    score -= int((row - center) ** 2 + (column - center) ** 2)
    return score


class TestV0143ExactPVSAcceleration(unittest.TestCase):
    def test_cached_quick_scores_equal_v0142_expression(self) -> None:
        for size in (8, 15):
            board = Board(size)
            stones = (
                (size // 2, size // 2, BLACK),
                (size // 2 - 1, size // 2, WHITE),
                (1, size - 2, BLACK),
            )
            for row, column, player in stones:
                board.place(row, column, player)
            ai = SearchAI(max_depth=1, time_limit_seconds=None)
            for player in (BLACK, WHITE):
                for move in board.get_legal_moves():
                    self.assertEqual(
                        reference_quick_score(board, move, player),
                        ai._quick_order_score(board, move, player),
                    )

    def test_quick_score_cache_is_position_scoped(self) -> None:
        board = Board()
        move = parse_move("H8", board.size)
        nearby = parse_move("H7", board.size)
        ai = SearchAI(max_depth=1, time_limit_seconds=None)
        empty_score = ai._quick_order_score(board, move, BLACK)
        board.place(*nearby, BLACK)
        occupied_score = ai._quick_order_score(board, move, BLACK)
        self.assertNotEqual(empty_score, occupied_score)
        board.undo()
        self.assertEqual(
            empty_score,
            ai._quick_order_score(board, move, BLACK),
        )

    def test_batched_win_scan_preserves_subset_order_and_board(self) -> None:
        board = Board()
        for coordinate in ("B8", "C8", "D8", "E8"):
            board.place(*parse_move(coordinate, board.size), BLACK)
        candidates = [
            parse_move(coordinate, board.size)
            for coordinate in ("F8", "A1", "A8")
        ]
        before = (
            tuple(tuple(row) for row in board.grid),
            tuple(board.move_history),
            board.zobrist_hash,
            board.empty_count,
        )
        ai = SearchAI(max_depth=1, time_limit_seconds=None)
        expected = [
            move
            for move in candidates
            if is_winning_move(board, *move, BLACK)
        ]
        self.assertEqual(
            expected,
            ai._timed_winning_moves(board, BLACK, candidates),
        )
        self.assertEqual(
            before,
            (
                tuple(tuple(row) for row in board.grid),
                tuple(board.move_history),
                board.zobrist_hash,
                board.empty_count,
            ),
        )

    def test_fixed_h11_search_tree_is_identical(self) -> None:
        board = build_board(H11_POSITION)
        ai = SearchAI(
            player=BLACK,
            max_depth=3,
            time_limit_seconds=None,
            diagnostics=True,
            top_n=12,
        )
        selected = ai.choose_move(board)
        analysis = ai.last_analysis
        assert analysis is not None
        self.assertEqual("H11", format_move(*selected))
        self.assertEqual(48, analysis.nodes)
        self.assertEqual(38, analysis.cutoffs)
        self.assertEqual(47, analysis.extensions)
        self.assertEqual(
            ("H11", "L7", "M6"),
            tuple(format_move(*move) for move in analysis.principal_variation),
        )
        self.assertEqual(
            (("H11", 11_300), ("L7", 11_300)),
            tuple(
                (format_move(*candidate.move), candidate.score)
                for candidate in analysis.top_candidates
            ),
        )


if __name__ == "__main__":
    unittest.main()
