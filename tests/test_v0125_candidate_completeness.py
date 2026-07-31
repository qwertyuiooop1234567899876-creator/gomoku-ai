import unittest

from engine import root_candidates, root_safety
from engine.board import BLACK, WHITE, Board
from engine.game import format_move, parse_move
from engine.search import SearchAI


def build_position(coordinates: tuple[str, ...]) -> Board:
    board = Board()
    for index, coordinate in enumerate(coordinates):
        board.place(
            *parse_move(coordinate, board.size),
            BLACK if index % 2 == 0 else WHITE,
        )
    return board


def board_state(board: Board) -> tuple[object, ...]:
    return (
        tuple(tuple(row) for row in board.grid),
        tuple(board.move_history),
        board.zobrist_hash,
        board.empty_count,
    )


WHITE_TWENTY_SIX_POSITION = (
    "H8", "I7", "G7", "I9", "I8", "G8", "F6", "J8",
    "H10", "H9", "K9", "I10", "F7", "F8", "E7", "D7",
    "G5", "D8", "F9", "E8", "C8", "J11", "K12", "D9",
    "D6",
)

BLACK_TWENTY_FIVE_POSITION = (
    "H8", "H9", "G7", "I9", "G9", "I7", "I8", "G8",
    "F7", "J8", "E7", "D7", "H10", "F8", "K9", "D8",
    "D6", "E8", "C8", "D11", "F9", "E11", "I11", "E9",
)


class TestCandidateMergePolicy(unittest.TestCase):
    def test_frontier_merge_reserves_tactical_sources_within_cap(
        self,
    ) -> None:
        frontier = ((1, 1),)
        ordinary = tuple((2, column) for column in range(12))
        counterattacks = ((3, 1), (3, 2))

        merged = root_candidates.frontier_defense_moves(
            frontier_moves=frontier,
            ordinary_moves=ordinary,
            counterattack_moves=counterattacks,
            limit=6,
        )

        self.assertEqual(6, len(merged))
        self.assertEqual(frontier[0], merged[0])
        self.assertTrue(set(counterattacks).issubset(merged))
        self.assertTrue(set(merged).intersection(ordinary))

    def test_vcf_merge_keeps_every_line_point_without_widening_root(
        self,
    ) -> None:
        candidates = [(0, column) for column in range(12)]
        line = (
            (1, 1), (1, 2), (2, 2), (2, 3),
            (3, 3), (3, 4), (4, 4),
        )

        merged = root_safety.merge_vcf_intercepts(
            candidates,
            line,
            limit=12,
        )

        self.assertEqual(12, len(merged))
        self.assertTrue(set(line).issubset(merged))


class TestV0125CandidateCompleteness(unittest.TestCase):
    def test_white_twenty_six_keeps_prevention_and_counterplay(
        self,
    ) -> None:
        board = build_position(WHITE_TWENTY_SIX_POSITION)
        before = board_state(board)
        ai = SearchAI(
            player=WHITE,
            max_depth=2,
            time_limit_seconds=None,
        )
        ai._begin_move_search()

        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )
        coordinates = {format_move(*move) for move in plan.moves}

        self.assertEqual(before, board_state(board))
        self.assertLessEqual(
            len(plan.moves),
            ai.config.root_candidate_limit,
        )
        self.assertIn("E6", coordinates)
        self.assertTrue({"E5", "H4"}.issubset(coordinates))
        self.assertTrue({"J9", "J10"}.intersection(coordinates))
        self.assertIn("普通候选", plan.reason)

    def test_black_twenty_five_scans_every_point_on_vcf_line(
        self,
    ) -> None:
        board = build_position(BLACK_TWENTY_FIVE_POSITION)
        before = board_state(board)
        ai = SearchAI(
            player=BLACK,
            max_depth=2,
            time_limit_seconds=None,
        )
        ai._begin_move_search()
        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )

        scan = ai._run_root_opponent_vcf_scan(board, plan.moves)

        self.assertIsNotNone(scan)
        assert scan is not None
        self.assertEqual(before, board_state(board))
        self.assertTrue(scan.baseline_line)
        self.assertLessEqual(
            len(scan.candidates),
            ai.config.root_candidate_limit,
        )
        self.assertTrue(
            set(scan.baseline_line).issubset(scan.candidates)
        )
        defender_plies = {
            format_move(*move)
            for move in scan.baseline_line[1::2]
        }
        candidate_coordinates = {
            format_move(*move)
            for move in scan.candidates
        }
        self.assertTrue({"D9", "B12"}.issubset(defender_plies))
        self.assertTrue(
            defender_plies.issubset(candidate_coordinates)
        )


if __name__ == "__main__":
    unittest.main()
