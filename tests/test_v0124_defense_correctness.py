import unittest

from engine import root_safety
from engine.ai import RootVCFCandidateAnalysis
from engine.board import BLACK, WHITE, Board
from engine.game import format_move, parse_move
from engine.search import SearchAI
from engine.search_types import VCFTimeout


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


BLACK_ELEVEN_POSITION = (
    "H8", "H9", "G7", "I9", "G9",
    "I7", "I8", "G8", "F7", "J8",
)

BLACK_SEVENTEEN_POSITION = (
    *BLACK_ELEVEN_POSITION,
    "H10", "K9", "L10", "J9", "L9", "J7",
)

BLACK_FORTY_THREE_POSITION = (
    "H8", "I7", "G7", "I9", "I8", "J8", "H6", "F8",
    "K7", "H7", "G8", "G9", "H9", "J7", "H10", "I10",
    "J6", "I6", "K8", "I5", "K9", "K10", "L8", "J10",
    "L9", "M9", "I3", "L10", "M10", "J11", "J9", "K11",
    "J12", "N8", "O7", "I11", "L11", "K12", "K5", "K6",
    "H11", "H12",
)


class TestRootVCFSafetyPolicy(unittest.TestCase):
    def test_completed_survivor_precedes_unknown_and_proven_loss(
        self,
    ) -> None:
        survivor = (1, 1)
        unknown = (1, 2)
        losing = (1, 3)
        analyses = (
            RootVCFCandidateAnalysis(
                survivor,
                root_safety.RootCandidateSafety
                .SURVIVES_VCF_SCAN.value,
                True,
                10,
                0.01,
            ),
            RootVCFCandidateAnalysis(
                unknown,
                root_safety.RootCandidateSafety.UNKNOWN.value,
                False,
                5,
                0.01,
            ),
            RootVCFCandidateAnalysis(
                losing,
                root_safety.RootCandidateSafety.PROVEN_LOSS.value,
                True,
                10,
                0.01,
            ),
        )

        selected = root_safety.apply_vcf_scan(
            [losing, unknown, survivor],
            analyses,
        )

        self.assertEqual([survivor], selected)

    def test_all_proven_losses_keep_original_resistance_set(self) -> None:
        first = (1, 1)
        second = (1, 2)
        analyses = tuple(
            RootVCFCandidateAnalysis(
                move,
                root_safety.RootCandidateSafety.PROVEN_LOSS.value,
                True,
                10,
                0.01,
            )
            for move in (first, second)
        )

        self.assertEqual(
            [first, second],
            root_safety.apply_vcf_scan(
                [first, second],
                analyses,
            ),
        )

    def test_intercepts_cover_full_alternating_line(self) -> None:
        line = ((1, 1), (1, 2), (2, 2), (2, 3), (3, 3))

        self.assertEqual(
            line,
            root_safety.vcf_intercept_moves(line),
        )

    def test_candidate_timeout_stays_unknown_and_restores_board(
        self,
    ) -> None:
        board = Board()
        board.place(7, 7, BLACK)
        before = board_state(board)
        calls = [0]

        def find_vcf(
            _board: Board,
            _attacker: int,
            _deadline: float | None,
        ) -> tuple[tuple[int, int], ...] | None:
            calls[0] += 1
            if calls[0] == 2:
                raise VCFTimeout
            return None

        scanner = root_safety.RootVCFSafetyScanner(
            find_vcf=find_vcf,
            node_count=lambda: 0,
        )
        first = (7, 8)
        second = (8, 7)

        result = scanner.scan(
            board,
            [first, second],
            mover=WHITE,
            opponent=BLACK,
            budget_seconds=None,
            hard_deadline=None,
        )

        by_move = {
            candidate.move: candidate
            for candidate in result.analyses
        }
        self.assertEqual(
            root_safety.RootCandidateSafety.UNKNOWN.value,
            by_move[first].status,
        )
        self.assertFalse(by_move[first].completed)
        self.assertEqual(
            root_safety.RootCandidateSafety
            .SURVIVES_VCF_SCAN.value,
            by_move[second].status,
        )
        self.assertTrue(by_move[second].completed)
        self.assertEqual(before, board_state(board))


class TestV0124DefenseRegressions(unittest.TestCase):
    def test_black_forty_three_rejects_l6_by_safety_property(
        self,
    ) -> None:
        board = build_position(BLACK_FORTY_THREE_POSITION)
        before = board_state(board)
        ai = SearchAI(
            player=BLACK,
            max_depth=2,
            time_limit_seconds=None,
            diagnostics=True,
        )

        selected = ai.choose_move(board)

        self.assertEqual(before, board_state(board))
        self.assertIsNotNone(ai.last_analysis)
        assert ai.last_analysis is not None
        by_coordinate = {
            format_move(*candidate.move): candidate
            for candidate in ai.last_analysis.root_vcf_candidates
        }
        self.assertEqual(
            root_safety.RootCandidateSafety.PROVEN_LOSS.value,
            by_coordinate["L6"].status,
        )
        self.assertEqual(
            root_safety.RootCandidateSafety
            .SURVIVES_VCF_SCAN.value,
            by_coordinate[format_move(*selected)].status,
        )
        self.assertNotEqual("L6", format_move(*selected))
        self.assertTrue(ai.last_analysis.root_vcf_complete)
        self.assertTrue(
            ai.last_analysis.root_vcf_baseline_line
        )

    def test_black_eleven_keeps_active_counterattack_e7(
        self,
    ) -> None:
        board = build_position(BLACK_ELEVEN_POSITION)
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

        self.assertEqual(before, board_state(board))
        self.assertIn(parse_move("E7", board.size), plan.moves)
        self.assertIn("主动反击点", plan.reason)

    def test_black_seventeen_branches_record_verified_vcf_difference(
        self,
    ) -> None:
        losing = build_position(
            (
                *BLACK_SEVENTEEN_POSITION,
                "J10", "H6", "G5", "K7", "L7",
            )
        )
        before = board_state(losing)
        ai = SearchAI(player=WHITE, time_limit_seconds=None)
        ai._begin_move_search()

        losing_line = ai._find_vcf(losing, WHITE)

        self.assertIsNotNone(losing_line)
        self.assertEqual("J6", format_move(*losing_line[0]))
        self.assertEqual(before, board_state(losing))

        for defense in ("H7", "L7"):
            with self.subTest(defense=defense):
                surviving = build_position(
                    (
                        *BLACK_SEVENTEEN_POSITION,
                        "J6", "H6", "G5", "K7", defense,
                    )
                )
                branch_before = board_state(surviving)
                branch_ai = SearchAI(
                    player=WHITE,
                    time_limit_seconds=None,
                )
                branch_ai._begin_move_search()

                self.assertIsNone(
                    branch_ai._find_vcf(surviving, WHITE)
                )
                self.assertEqual(
                    branch_before,
                    board_state(surviving),
                )


if __name__ == "__main__":
    unittest.main()
