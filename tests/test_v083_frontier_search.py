import unittest

from engine.board import BLACK, WHITE, Board
from engine.game import parse_move
from engine.search import SearchAI


class TestV083FrontierGuidedSearch(unittest.TestCase):
    def _build_friend_game_position(self) -> Board:
        """构造 K9/J7/H6/J10 四个前沿候选同时出现的实战局面。"""
        board = Board()
        moves = (
            "H8", "H7",
            "I7", "G9",
            "G8", "I8",
            "J9", "G6",
            "J8",
        )
        for index, coordinate in enumerate(moves):
            row, column = parse_move(coordinate, board.size)
            board.place(
                row,
                column,
                BLACK if index % 2 == 0 else WHITE,
            )
        return board

    def test_multiple_frontiers_are_decided_by_pvs(self) -> None:
        board = self._build_friend_game_position()
        grid_before = [row.copy() for row in board.grid]
        history_before = board.move_history.copy()
        hash_before = board.zobrist_hash

        ai = SearchAI(
            player=WHITE,
            max_depth=3,
            time_limit_seconds=None,
            diagnostics=True,
        )
        move = ai.choose_move(board)

        expected = parse_move("J7", board.size)
        self.assertEqual(expected, move)
        assert ai.last_analysis is not None
        self.assertIn("多重威胁启动点候选的 PVS", ai.last_analysis.reason)
        self.assertGreater(ai.last_analysis.nodes, 0)
        self.assertEqual(3, ai.last_analysis.search_depth)
        self.assertEqual(grid_before, board.grid)
        self.assertEqual(history_before, board.move_history)
        self.assertEqual(hash_before, board.zobrist_hash)


if __name__ == "__main__":
    unittest.main()
