from __future__ import annotations

import unittest

from engine.board import BLACK, WHITE, Board
from engine.game import parse_move
from engine.search import SearchAI
from engine.search_types import BoundType, MATE_SCORE


def reordered_same_position() -> tuple[Board, Board]:
    orders = (
        "H8 H9 G8 G9 F8 F9 E8 E9 D8 D9".split(),
        "D8 D9 E8 E9 F8 F9 G8 G9 H8 H9".split(),
    )
    boards: list[Board] = []
    for coordinates in orders:
        board = Board()
        for index, coordinate in enumerate(coordinates):
            board.place(
                *parse_move(coordinate),
                BLACK if index % 2 == 0 else WHITE,
            )
        boards.append(board)
    return boards[0], boards[1]


class TestSearchTranspositionInvariants(unittest.TestCase):
    def test_recent_move_order_is_part_of_selective_tt_key(self) -> None:
        first, second = reordered_same_position()
        ai = SearchAI(BLACK, max_depth=2, time_limit_seconds=None)

        self.assertEqual(first.grid, second.grid)
        self.assertEqual(first.zobrist_hash, second.zobrist_hash)
        self.assertNotEqual(
            ai._raw_candidates(
                first,
                first.get_legal_moves(),
                at_root=False,
            ),
            ai._raw_candidates(
                second,
                second.get_legal_moves(),
                at_root=False,
            ),
        )
        self.assertNotEqual(
            ai._position_key(first, BLACK),
            ai._position_key(second, BLACK),
        )

    def test_mate_scores_are_normalized_across_root_ply(self) -> None:
        ai = SearchAI(BLACK)

        stored_win = ai._score_to_tt(MATE_SCORE - 7, ply=7)
        stored_loss = ai._score_to_tt(-MATE_SCORE + 7, ply=7)

        self.assertEqual(MATE_SCORE - 2, ai._score_from_tt(stored_win, 2))
        self.assertEqual(-MATE_SCORE + 2, ai._score_from_tt(stored_loss, 2))
        self.assertEqual(12_345, ai._score_from_tt(12_345, 9))

    def test_weaker_equal_depth_entry_cannot_replace_exact_tt_data(self) -> None:
        ai = SearchAI(BLACK)
        key = 123
        ai._store_tt(
            key,
            depth=3,
            extension_depth=2,
            score=42,
            alpha_original=-100,
            beta_original=100,
            principal_variation=((1, 1),),
            best_move=(1, 1),
            ply=0,
        )
        ai._store_tt(
            key,
            depth=3,
            extension_depth=2,
            score=100,
            alpha_original=-100,
            beta_original=100,
            principal_variation=((2, 2),),
            best_move=(2, 2),
            ply=0,
        )

        entry = ai._transposition_table[key]
        self.assertIs(BoundType.EXACT, entry.bound)
        self.assertEqual(42, entry.score)
        self.assertEqual((1, 1), entry.best_move)

    def test_search_heuristics_reorder_but_do_not_select_members(self) -> None:
        board, _other = reordered_same_position()
        ai = SearchAI(
            BLACK,
            max_depth=2,
            time_limit_seconds=None,
            branch_candidate_limit=4,
        )
        baseline = ai._ordered_moves(
            board,
            BLACK,
            at_root=False,
            ply=3,
            tt_move=None,
        )
        promoted = baseline[-1]
        ai._history_scores[(BLACK, *promoted)] = 1_000_000
        ai._killer_moves[3] = [promoted]

        reordered = ai._ordered_moves(
            board,
            BLACK,
            at_root=False,
            ply=3,
            tt_move=promoted,
        )

        self.assertEqual(set(baseline), set(reordered))
        self.assertEqual(promoted, reordered[0])


if __name__ == "__main__":
    unittest.main()
