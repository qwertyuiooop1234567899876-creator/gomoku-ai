import unittest

from engine.board import BLACK, WHITE, Board
from engine.search import SearchAI


class TestV082ThreatFrontier(unittest.TestCase):
    @staticmethod
    def _parse(coordinate: str) -> tuple[int, int]:
        return int(coordinate[1:]) - 1, ord(coordinate[0]) - ord("A")

    def _build_g9_position(self) -> Board:
        board = Board()
        moves = (
            "H8", "H7",
            "G7", "I9",
            "F8", "E8",
            "E9", "H6",
            "F10",
        )
        for index, coordinate in enumerate(moves):
            row, column = self._parse(coordinate)
            board.place(
                row,
                column,
                BLACK if index % 2 == 0 else WHITE,
            )
        return board

    def test_search_blocks_g9_multi_threat_launch(self) -> None:
        board = self._build_g9_position()
        ai = SearchAI(
            player=WHITE,
            max_depth=2,
            time_limit_seconds=None,
            diagnostics=True,
        )

        self.assertEqual(self._parse("G9"), ai.choose_move(board))
        assert ai.last_analysis is not None
        self.assertIn("多重威胁启动点", ai.last_analysis.reason)

    def test_g9_frontier_contains_multiple_forcing_replies(self) -> None:
        board = self._build_g9_position()
        ai = SearchAI(
            player=WHITE,
            max_depth=2,
            time_limit_seconds=None,
        )
        root_pool = ai._root_profile_pool(board, board.get_legal_moves())
        profiles = ai._profile_moves_timed(board, root_pool, BLACK)
        frontiers = ai._multi_threat_frontiers(
            board,
            root_pool,
            BLACK,
            profiles=profiles,
        )

        g9 = self._parse("G9")
        self.assertIn(g9, frontiers)
        self.assertGreaterEqual(len(frontiers[g9]), 2)
        self.assertIn(self._parse("I7"), frontiers[g9])
        self.assertIn(self._parse("F9"), frontiers[g9])

    def test_frontier_detection_does_not_modify_board(self) -> None:
        board = self._build_g9_position()
        grid_before = [row.copy() for row in board.grid]
        history_before = board.move_history.copy()
        hash_before = board.zobrist_hash
        ai = SearchAI(
            player=WHITE,
            max_depth=2,
            time_limit_seconds=None,
        )
        root_pool = ai._root_profile_pool(board, board.get_legal_moves())
        profiles = ai._profile_moves_timed(board, root_pool, BLACK)
        ai._multi_threat_frontiers(
            board,
            root_pool,
            BLACK,
            profiles=profiles,
        )

        self.assertEqual(grid_before, board.grid)
        self.assertEqual(history_before, board.move_history)
        self.assertEqual(hash_before, board.zobrist_hash)


if __name__ == "__main__":
    unittest.main()
