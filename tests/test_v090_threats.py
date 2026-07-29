import unittest

from engine.board import BLACK, WHITE, Board
from engine.proof_search import (
    ProofBudget,
    ProofSearch,
    ProofState,
    ProofTable,
)
from engine.threats import ThreatAnalyzer, ThreatKind


def board_state(board: Board) -> tuple[object, ...]:
    return (
        tuple(tuple(row) for row in board.grid),
        tuple(board.move_history),
        board.zobrist_hash,
        board.empty_count,
    )


def build_horizontal_open_three() -> Board:
    board = Board(size=9)
    board.place(4, 2, BLACK)
    board.place(4, 3, BLACK)
    return board


def build_crossing_double_three() -> Board:
    board = Board(size=9)
    for move in ((4, 3), (4, 5), (3, 4), (5, 4)):
        board.place(*move, BLACK)
    return board


class TestExactThreatDescriptions(unittest.TestCase):
    def test_analyzer_reports_work_counts_without_changing_results(
        self,
    ) -> None:
        board = build_horizontal_open_three()
        before = board_state(board)
        analyzer = ThreatAnalyzer()

        threat = analyzer.describe_move(board, (4, 4), BLACK)
        analyzer.generate_attack_candidates(board, BLACK)
        analyzer.generate_attack_frontiers(
            board,
            BLACK,
            frontier_limit=2,
        )

        stats = analyzer.stats()
        self.assertIs(ThreatKind.OPEN_THREE, threat.kind)
        self.assertEqual(1, stats.exact_descriptions)
        self.assertEqual(1, stats.candidate_batches)
        self.assertEqual(1, stats.frontier_batches)
        self.assertGreater(stats.cache_queries, 0)
        self.assertGreater(stats.cache_stores, 0)
        self.assertEqual(before, board_state(board))

    def test_complete_exact_description_is_reused_from_cache(
        self,
    ) -> None:
        board = build_horizontal_open_three()
        before = board_state(board)
        analyzer = ThreatAnalyzer()

        first = analyzer.describe_move(board, (4, 4), BLACK)
        second = analyzer.describe_move(board, (4, 4), BLACK)

        stats = analyzer.stats()
        self.assertEqual(first, second)
        self.assertEqual(1, stats.exact_descriptions)
        self.assertGreaterEqual(stats.cache_hits, 2)
        self.assertEqual(before, board_state(board))

    def test_complete_candidate_batch_is_reused_from_cache(
        self,
    ) -> None:
        board = build_crossing_double_three()
        before = board_state(board)
        analyzer = ThreatAnalyzer()

        first = analyzer.generate_attack_candidates(board, BLACK)
        second = analyzer.generate_attack_candidates(board, BLACK)

        stats = analyzer.stats()
        self.assertEqual(first, second)
        self.assertTrue(first.generation_completed)
        self.assertGreaterEqual(stats.cache_hits, 1)
        self.assertEqual(before, board_state(board))

    def test_open_three_has_exhaustive_two_move_defense_set(self) -> None:
        board = build_horizontal_open_three()
        before = board_state(board)

        analyzer = ThreatAnalyzer()
        threat = analyzer.describe_move(
            board,
            (4, 4),
            BLACK,
        )

        self.assertIs(ThreatKind.OPEN_THREE, threat.kind)
        self.assertEqual(((4, 1), (4, 5)), threat.winning_continuations)
        self.assertEqual(((4, 1), (4, 5)), threat.required_defenses)
        self.assertEqual(
            ((4, 0), (4, 1), (4, 5), (4, 6)),
            threat.rest_squares,
        )
        self.assertEqual(((0, 1),), threat.source_lines)
        self.assertTrue(threat.coverage_complete)
        self.assertTrue(threat.analysis_completed)
        self.assertEqual((), threat.unclassified_defenses)
        self.assertEqual(
            threat.legal_reply_count,
            len(threat.required_defenses)
            + len(threat.counter_wins)
            + threat.refuted_reply_count,
        )
        self.assertEqual(before, board_state(board))

    def test_vertical_translated_threat_uses_same_rules(self) -> None:
        board = Board(size=11)
        board.place(3, 7, BLACK)
        board.place(4, 7, BLACK)
        before = board_state(board)

        threat = ThreatAnalyzer().describe_move(
            board,
            (5, 7),
            BLACK,
        )

        self.assertIs(ThreatKind.OPEN_THREE, threat.kind)
        self.assertEqual(((2, 7), (6, 7)), threat.winning_continuations)
        self.assertEqual(((2, 7), (6, 7)), threat.required_defenses)
        self.assertEqual(((1, 0),), threat.source_lines)
        self.assertTrue(threat.coverage_complete)
        self.assertEqual(before, board_state(board))

    def test_crossing_double_three_has_no_single_complete_defense(self) -> None:
        board = build_crossing_double_three()
        before = board_state(board)

        threat = ThreatAnalyzer().describe_move(
            board,
            (4, 4),
            BLACK,
        )

        self.assertIs(ThreatKind.DOUBLE_THREE, threat.kind)
        self.assertEqual(
            ((2, 4), (4, 2), (4, 6), (6, 4)),
            threat.winning_continuations,
        )
        self.assertEqual((), threat.required_defenses)
        self.assertEqual(((0, 1), (1, 0)), threat.source_lines)
        self.assertTrue(threat.coverage_complete)
        self.assertEqual(
            threat.legal_reply_count,
            threat.refuted_reply_count,
        )
        refutations = {
            item.defense_move: item
            for item in threat.defense_refutations
        }
        self.assertIn((0, 0), refutations)
        self.assertFalse(
            refutations[(0, 0)].continuation_is_immediate
        )
        self.assertGreaterEqual(
            len(refutations[(0, 0)].winning_points),
            2,
        )
        self.assertEqual(before, board_state(board))

    def test_defense_set_includes_a_reply_that_creates_counter_threats(
        self,
    ) -> None:
        board = build_horizontal_open_three()
        for column in (2, 3, 4):
            board.place(1, column, WHITE)
        before = board_state(board)

        threat = ThreatAnalyzer().describe_move(
            board,
            (4, 4),
            BLACK,
        )

        # White B2 creates an open four. It does not occupy a black cost
        # square, but it refutes the planned continuation by moving faster.
        self.assertIn((1, 1), threat.required_defenses)
        self.assertNotIn((8, 8), threat.required_defenses)
        self.assertTrue(threat.coverage_complete)
        self.assertEqual(before, board_state(board))

    def test_quiet_move_is_complete_analysis_but_incomplete_coverage(
        self,
    ) -> None:
        board = Board(size=9)
        board.place(4, 4, BLACK)
        before = board_state(board)

        threat = ThreatAnalyzer().describe_move(
            board,
            (0, 0),
            BLACK,
        )

        self.assertIs(ThreatKind.QUIET, threat.kind)
        self.assertTrue(threat.analysis_completed)
        self.assertFalse(threat.coverage_complete)
        self.assertEqual((), threat.winning_continuations)
        self.assertEqual(
            threat.legal_reply_count,
            len(threat.unclassified_defenses),
        )
        self.assertEqual(before, board_state(board))

    def test_interrupted_description_is_incomplete_and_restores_board(
        self,
    ) -> None:
        board = build_horizontal_open_three()
        before = board_state(board)

        analyzer = ThreatAnalyzer()
        threat = analyzer.describe_move(
            board,
            (4, 4),
            BLACK,
            stop_requested=lambda: True,
        )

        self.assertFalse(threat.analysis_completed)
        self.assertFalse(threat.coverage_complete)
        completed = analyzer.describe_move(board, (4, 4), BLACK)
        stats = analyzer.stats()
        self.assertTrue(completed.analysis_completed)
        self.assertEqual(2, stats.exact_descriptions)
        self.assertGreaterEqual(stats.cache_skips, 1)
        self.assertEqual(before, board_state(board))

    def test_interruption_during_reply_classification_is_recoverable(
        self,
    ) -> None:
        board = build_horizontal_open_three()
        before = board_state(board)
        analyzer = ThreatAnalyzer()
        line_candidate_count = len(
            analyzer._line_candidates(board, (4, 4))
        )
        calls = 0

        def stop_after_two_replies() -> bool:
            nonlocal calls
            calls += 1
            return calls > line_candidate_count + 2

        threat = analyzer.describe_move(
            board,
            (4, 4),
            BLACK,
            stop_requested=stop_after_two_replies,
        )

        self.assertFalse(threat.analysis_completed)
        self.assertFalse(threat.coverage_complete)
        self.assertGreater(len(threat.unclassified_defenses), 0)
        self.assertEqual(before, board_state(board))


class TestExactThreatProofSearch(unittest.TestCase):
    def test_double_three_is_proven_through_complete_defense_coverage(
        self,
    ) -> None:
        board = build_crossing_double_three()
        before = board_state(board)

        result = ProofSearch(
            budget=ProofBudget(
                max_nodes=1_000,
                max_attacker_moves=2,
            )
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertIs(ProofState.PROVEN_WIN, result.state)
        self.assertEqual((4, 4), result.best_move)
        self.assertEqual((4, 4), result.principal_variation[0])
        self.assertGreaterEqual(len(result.principal_variation), 4)
        self.assertEqual(2, result.searched_attacker_moves)
        self.assertTrue(result.completed)
        self.assertEqual(before, board_state(board))

    def test_double_three_needs_budget_for_nonterminal_continuation(
        self,
    ) -> None:
        board = build_crossing_double_three()
        before = board_state(board)

        result = ProofSearch(
            budget=ProofBudget(
                max_nodes=1_000,
                max_attacker_moves=1,
            )
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertIs(ProofState.UNKNOWN, result.state)
        self.assertEqual("attacker_depth_limit", result.cutoff_reason)
        self.assertEqual(before, board_state(board))

    def test_shallow_double_three_unknown_does_not_block_deeper_proof(
        self,
    ) -> None:
        board = build_crossing_double_three()
        before = board_state(board)
        table = ProofTable()

        shallow = ProofSearch(
            budget=ProofBudget(
                max_nodes=1_000,
                max_attacker_moves=1,
            ),
            table=table,
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )
        deeper = ProofSearch(
            budget=ProofBudget(
                max_nodes=1_000,
                max_attacker_moves=2,
            ),
            table=table,
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertIs(ProofState.UNKNOWN, shallow.state)
        self.assertIs(ProofState.PROVEN_WIN, deeper.state)
        self.assertEqual(before, board_state(board))

    def test_single_open_three_is_not_promoted_to_proven_win(self) -> None:
        board = build_horizontal_open_three()
        before = board_state(board)

        result = ProofSearch(
            budget=ProofBudget(
                max_nodes=1_000,
                max_attacker_moves=1,
            )
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertIs(ProofState.UNKNOWN, result.state)
        self.assertFalse(result.completed)
        self.assertEqual(before, board_state(board))

    def test_defender_immediate_win_overrides_exact_attack_obligation(
        self,
    ) -> None:
        board = build_crossing_double_three()
        for column in (0, 1, 2, 3):
            board.place(1, column, WHITE)
        board.place(4, 4, BLACK)
        before = board_state(board)

        result = ProofSearch().search(
            board,
            attacker=BLACK,
            side_to_move=WHITE,
        )

        self.assertIs(ProofState.PROVEN_LOSS, result.state)
        self.assertEqual((1, 4), result.best_move)
        self.assertEqual(before, board_state(board))


if __name__ == "__main__":
    unittest.main()
