import unittest

from engine.board import BLACK, WHITE, Board
from engine.game import format_move, parse_move
from engine.proof_search import ProofBudget, ProofSearch, ProofState
from engine.threats import (
    ThreatAnalyzer,
    ThreatCandidateBatch,
    ThreatKind,
)


def board_state(board: Board) -> tuple[object, ...]:
    return (
        tuple(tuple(row) for row in board.grid),
        tuple(board.move_history),
        board.zobrist_hash,
        board.empty_count,
    )


def build_position(coordinates: tuple[str, ...]) -> Board:
    board = Board()
    player = BLACK
    for coordinate in coordinates:
        board.place(
            *parse_move(coordinate, board.size),
            player,
        )
        player = WHITE if player == BLACK else BLACK
    return board


G9_POSITION = (
    "H8", "H7",
    "G7", "I9",
    "F8", "E8",
    "E9", "H6",
    "F10",
)

H11_POSITION = (
    "H8", "I7",
    "I6", "H7",
    "G7", "F6",
    "J7", "K8",
    "J8", "J9",
    "I8", "I10",
)

I4_POSITION = (
    "H8", "I7",
    "I6", "H7",
    "G7", "G6",
    "I8", "J8",
    "H6", "F8",
    "K3", "K6",
    "F6", "E5",
    "J4", "I5",
    "J7", "H5",
)

E9_YIXIN_POSITION = (
    "H8", "I7",
    "I8", "H9",
    "H7", "G8",
    "F7", "G6",
    "G7", "F6",
    "H6", "H4",
    "G5", "E7",
    "I5", "F8",
    "G9", "D8",
)

J10_YIXIN_POSITION = E9_YIXIN_POSITION

E10_YIXIN_POSITION = (
    "H8", "I7",
    "I8", "J8",
    "H6", "H7",
    "G7", "F8",
    "F6", "E5",
    "I9", "J10",
    "J9", "K9",
    "I11", "I10",
    "L10", "E7",
    "G9", "G8",
    "H9", "F9",
)


class TestV090ThreatFrontierRegressions(unittest.TestCase):
    def test_g9_frontier_records_dependent_future_threats(self) -> None:
        board = build_position(G9_POSITION)
        before = board_state(board)

        frontiers = ThreatAnalyzer().generate_attack_frontiers(
            board,
            BLACK,
            frontier_limit=64,
        )
        by_coordinate = {
            format_move(*frontier.gain_move): frontier
            for frontier in frontiers
        }

        self.assertIn("G9", by_coordinate)
        g9 = by_coordinate["G9"]
        continuations = {
            format_move(*move)
            for move in g9.continuations
        }
        self.assertIn("I7", continuations)
        self.assertIn("F9", continuations)
        self.assertGreaterEqual(len(continuations), 2)
        self.assertFalse(g9.coverage_complete)
        self.assertFalse(hasattr(g9, "proof_state"))
        self.assertEqual(before, board_state(board))

    def test_mirrored_g9_shape_uses_the_same_frontier_rules(self) -> None:
        original = build_position(G9_POSITION)
        mirrored = Board(size=original.size)
        for row, column, player in original.move_history:
            mirrored.place(
                row,
                original.size - 1 - column,
                player,
            )
        before = board_state(mirrored)
        mirrored_gain = (
            parse_move("G9", original.size)[0],
            original.size - 1
            - parse_move("G9", original.size)[1],
        )

        frontiers = ThreatAnalyzer().generate_attack_frontiers(
            mirrored,
            BLACK,
            frontier_limit=64,
        )
        by_move = {
            frontier.gain_move: frontier
            for frontier in frontiers
        }

        self.assertIn(mirrored_gain, by_move)
        self.assertGreaterEqual(
            len(by_move[mirrored_gain].continuations),
            2,
        )
        self.assertEqual(before, board_state(mirrored))

    def test_i4_position_generates_e7_as_quiet_frontier(self) -> None:
        board = build_position(I4_POSITION)
        i4 = parse_move("I4", board.size)
        board.place(*i4, BLACK)
        before = board_state(board)

        frontiers = ThreatAnalyzer().generate_attack_frontiers(
            board,
            WHITE,
            frontier_limit=64,
        )
        by_coordinate = {
            format_move(*frontier.gain_move): frontier
            for frontier in frontiers
        }

        self.assertIn("E7", by_coordinate)
        e7 = by_coordinate["E7"]
        self.assertIs(ThreatKind.QUIET, e7.kind)
        self.assertIn(
            "D6",
            {
                format_move(*move)
                for move in e7.continuations
            },
        )
        self.assertFalse(e7.coverage_complete)
        self.assertEqual(before, board_state(board))

    def test_h11_and_l7_replies_have_exact_single_defenses(self) -> None:
        analyzer = ThreatAnalyzer()
        cases = (
            ("L7", "H11", "G12"),
            ("H11", "L7", "M6"),
        )

        for defense, attack, forced_reply in cases:
            with self.subTest(defense=defense):
                board = build_position(H11_POSITION)
                board.place(
                    *parse_move(defense, board.size),
                    BLACK,
                )
                before = board_state(board)

                threat = analyzer.describe_move(
                    board,
                    parse_move(attack, board.size),
                    WHITE,
                )

                self.assertIs(ThreatKind.FOUR, threat.kind)
                self.assertEqual(
                    (forced_reply,),
                    tuple(
                        format_move(*move)
                        for move in threat.required_defenses
                    ),
                )
                self.assertTrue(threat.coverage_complete)
                self.assertEqual(before, board_state(board))

    def test_e9_yixin_position_is_reconstructed_before_black_move_19(
        self,
    ) -> None:
        board = build_position(E9_YIXIN_POSITION)

        self.assertEqual(18, len(board.move_history))
        self.assertEqual(BLACK, board.move_history[0][2])
        self.assertEqual(WHITE, board.move_history[-1][2])
        self.assertTrue(board.is_empty(*parse_move("E9", board.size)))
        self.assertTrue(board.is_empty(*parse_move("H5", board.size)))
        last_row, last_column, _ = board.move_history[-1]
        self.assertFalse(board.check_win(last_row, last_column))

    def test_j10_yixin_position_keeps_both_near_tie_candidates_legal(
        self,
    ) -> None:
        board = build_position(J10_YIXIN_POSITION)
        before = board_state(board)

        self.assertTrue(board.is_empty(*parse_move("J10", board.size)))
        self.assertTrue(board.is_empty(*parse_move("J4", board.size)))
        self.assertEqual(18, len(board.move_history))
        self.assertEqual(before, board_state(board))

    def test_e10_yixin_position_keeps_pvs_and_risk_moves_legal(
        self,
    ) -> None:
        board = build_position(E10_YIXIN_POSITION)
        before = board_state(board)

        self.assertTrue(board.is_empty(*parse_move("I6", board.size)))
        self.assertTrue(board.is_empty(*parse_move("E10", board.size)))
        self.assertEqual(22, len(board.move_history))
        self.assertEqual(before, board_state(board))


class TestV090CandidateProbeBoundary(unittest.TestCase):
    def test_i4_probe_unknown_is_preserved_and_restores_board(self) -> None:
        board = build_position(I4_POSITION)
        before = board_state(board)

        result = ProofSearch(
            budget=ProofBudget(
                max_nodes=0,
                max_attacker_moves=7,
            )
        ).search_after_move(
            board,
            move=parse_move("I4", board.size),
            mover=BLACK,
            attacker=WHITE,
        )

        self.assertIs(ProofState.UNKNOWN, result.state)
        self.assertFalse(result.completed)
        self.assertEqual("node_limit", result.cutoff_reason)
        self.assertEqual(before, board_state(board))

    def test_candidate_probe_can_return_a_strict_opponent_win(self) -> None:
        board = Board(size=9)
        for column in (1, 2, 3, 4):
            board.place(4, column, WHITE)
        before = board_state(board)

        result = ProofSearch().search_after_move(
            board,
            move=(0, 0),
            mover=BLACK,
            attacker=WHITE,
            side_to_move=WHITE,
        )

        self.assertIs(ProofState.PROVEN_WIN, result.state)
        self.assertIn(result.best_move, ((4, 0), (4, 5)))
        self.assertEqual(before, board_state(board))

    def test_exact_candidate_descriptions_are_built_lazily(self) -> None:
        class CountingAnalyzer(ThreatAnalyzer):
            def __init__(self) -> None:
                super().__init__()
                self.described: list[tuple[int, int]] = []

            def generate_attack_candidates(
                self,
                board: Board,
                player: int,
                *,
                stop_requested=None,  # type: ignore[no-untyped-def]
            ) -> ThreatCandidateBatch:
                batch = super().generate_attack_candidates(
                    board,
                    player,
                    stop_requested=stop_requested,
                )
                by_move = {
                    candidate.move: candidate
                    for candidate in batch.candidates
                }
                ordered = (
                    by_move[(4, 5)],
                    by_move[(4, 1)],
                )
                return ThreatCandidateBatch(
                    candidates=ordered,
                    coverage_complete=False,
                )

            def describe_candidate(
                self,
                board: Board,
                candidate,  # type: ignore[no-untyped-def]
                player: int,
                *,
                stop_requested=None,  # type: ignore[no-untyped-def]
            ):  # type: ignore[no-untyped-def]
                self.described.append(candidate.move)
                return super().describe_candidate(
                    board,
                    candidate,
                    player,
                    stop_requested=stop_requested,
                )

        board = Board(size=9)
        for column in (2, 3, 4):
            board.place(4, column, BLACK)
        analyzer = CountingAnalyzer()

        result = ProofSearch(
            budget=ProofBudget(
                max_nodes=100,
                max_attacker_moves=1,
            ),
            analyzer=analyzer,
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertIs(ProofState.PROVEN_WIN, result.state)
        self.assertEqual([(4, 5)], analyzer.described)


if __name__ == "__main__":
    unittest.main()
