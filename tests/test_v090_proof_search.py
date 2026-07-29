import unittest

from engine.board import BLACK, WHITE, Board
from engine.proof_search import (
    ProofBudget,
    ProofSearch,
    ProofState,
    ProofTable,
    combine_and_states,
    combine_or_states,
)
from engine.search import SearchAI
from engine.threats import ThreatAnalyzer, ThreatBatch, ThreatKind


def board_state(board: Board) -> tuple[object, ...]:
    return (
        tuple(tuple(row) for row in board.grid),
        tuple(board.move_history),
        board.zobrist_hash,
        board.empty_count,
    )


def build_open_four_launch() -> Board:
    board = Board(size=9)
    for column in (2, 3, 4):
        board.place(4, column, BLACK)
    return board


class TestProofStateCombination(unittest.TestCase):
    def test_or_node_accepts_one_winning_witness(self) -> None:
        state = combine_or_states(
            (
                ProofState.PROVEN_LOSS,
                ProofState.UNKNOWN,
                ProofState.PROVEN_WIN,
            ),
            coverage_complete=False,
        )
        self.assertIs(ProofState.PROVEN_WIN, state)

    def test_or_node_requires_complete_set_to_prove_loss(self) -> None:
        children = (
            ProofState.PROVEN_LOSS,
            ProofState.PROVEN_LOSS,
        )
        self.assertIs(
            ProofState.PROVEN_LOSS,
            combine_or_states(children, coverage_complete=True),
        )
        self.assertIs(
            ProofState.UNKNOWN,
            combine_or_states(children, coverage_complete=False),
        )

    def test_and_node_accepts_one_refuting_defense(self) -> None:
        state = combine_and_states(
            (
                ProofState.PROVEN_WIN,
                ProofState.PROVEN_LOSS,
                ProofState.UNKNOWN,
            ),
            coverage_complete=False,
        )
        self.assertIs(ProofState.PROVEN_LOSS, state)

    def test_and_node_requires_every_complete_defense_to_lose(self) -> None:
        children = (
            ProofState.PROVEN_WIN,
            ProofState.PROVEN_WIN,
        )
        self.assertIs(
            ProofState.PROVEN_WIN,
            combine_and_states(children, coverage_complete=True),
        )
        self.assertIs(
            ProofState.UNKNOWN,
            combine_and_states(children, coverage_complete=False),
        )

    def test_unknown_is_neither_safe_nor_a_loss(self) -> None:
        self.assertIs(
            ProofState.UNKNOWN,
            combine_or_states(
                (ProofState.UNKNOWN,),
                coverage_complete=True,
            ),
        )
        self.assertIs(
            ProofState.UNKNOWN,
            combine_and_states(
                (ProofState.UNKNOWN,),
                coverage_complete=True,
            ),
        )


class TestThreatDescriptions(unittest.TestCase):
    def test_open_four_is_a_candidate_not_a_proof_state(self) -> None:
        board = build_open_four_launch()
        before = board_state(board)

        threat = ThreatAnalyzer().describe_move(
            board,
            (4, 5),
            BLACK,
        )

        self.assertIs(ThreatKind.OPEN_FOUR, threat.kind)
        self.assertEqual(((4, 1), (4, 6)), threat.winning_continuations)
        self.assertTrue(threat.coverage_complete)
        self.assertFalse(hasattr(threat, "proof_state"))
        self.assertEqual(before, board_state(board))


class TestProofSearchFoundation(unittest.TestCase):
    def test_immediate_win_is_proven(self) -> None:
        board = Board(size=9)
        for column in (1, 2, 3, 4):
            board.place(4, column, BLACK)

        result = ProofSearch().search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertIs(ProofState.PROVEN_WIN, result.state)
        self.assertIn(result.best_move, ((4, 0), (4, 5)))
        self.assertTrue(result.completed)

    def test_open_four_launch_is_proven_by_and_or_search(self) -> None:
        board = build_open_four_launch()
        before = board_state(board)

        result = ProofSearch(
            budget=ProofBudget(
                max_nodes=100,
                max_attacker_moves=1,
            )
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertIs(ProofState.PROVEN_WIN, result.state)
        self.assertIn(result.best_move, ((4, 1), (4, 5)))
        self.assertGreaterEqual(result.nodes, 2)
        self.assertEqual(before, board_state(board))

    def test_zero_node_budget_returns_unknown(self) -> None:
        board = build_open_four_launch()
        before = board_state(board)
        table = ProofTable()

        result = ProofSearch(
            budget=ProofBudget(
                max_nodes=0,
                max_attacker_moves=2,
            ),
            table=table,
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertIs(ProofState.UNKNOWN, result.state)
        self.assertFalse(result.completed)
        self.assertEqual("node_limit", result.cutoff_reason)
        self.assertEqual(0, len(table))
        self.assertEqual(before, board_state(board))

    def test_node_cutoff_after_simulation_restores_every_board_field(self) -> None:
        board = build_open_four_launch()
        board.place(4, 1, WHITE)
        before = board_state(board)

        result = ProofSearch(
            budget=ProofBudget(
                max_nodes=2,
                max_attacker_moves=2,
            )
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertIs(ProofState.UNKNOWN, result.state)
        self.assertEqual("node_limit", result.cutoff_reason)
        self.assertEqual(before, board_state(board))

    def test_deadline_after_simulation_restores_and_does_not_cache(self) -> None:
        class SingleThreatAnalyzer(ThreatAnalyzer):
            def generate_attack_threats(
                self,
                board: Board,
                player: int,
                *,
                stop_requested=None,  # type: ignore[no-untyped-def]
            ) -> ThreatBatch:
                return ThreatBatch(
                    threats=(
                        self.describe_move(board, (4, 5), player),
                    ),
                    coverage_complete=False,
                )

        class StepClock:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self) -> float:
                self.calls += 1
                return 0.0 if self.calls <= 3 else 2.0

        board = build_open_four_launch()
        board.place(4, 1, WHITE)
        before = board_state(board)
        table = ProofTable()

        result = ProofSearch(
            budget=ProofBudget(
                max_nodes=100,
                max_attacker_moves=2,
                deadline=1.0,
            ),
            analyzer=SingleThreatAnalyzer(),
            table=table,
            clock=StepClock(),
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertIs(ProofState.UNKNOWN, result.state)
        self.assertEqual("deadline", result.cutoff_reason)
        self.assertEqual(0, len(table))
        self.assertEqual(before, board_state(board))

    def test_shallow_unknown_does_not_block_deeper_proof(self) -> None:
        board = build_open_four_launch()
        table = ProofTable()

        shallow = ProofSearch(
            budget=ProofBudget(
                max_nodes=100,
                max_attacker_moves=0,
            ),
            table=table,
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )
        deeper = ProofSearch(
            budget=ProofBudget(
                max_nodes=100,
                max_attacker_moves=1,
            ),
            table=table,
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertIs(ProofState.UNKNOWN, shallow.state)
        self.assertIs(ProofState.PROVEN_WIN, deeper.state)
        self.assertGreater(len(table), 0)

    def test_defender_immediate_counterwin_refutes_double_threat(self) -> None:
        board = Board(size=9)
        for column in (2, 3, 4, 5):
            board.place(5, column, BLACK)
        for column in (0, 1, 2, 3):
            board.place(1, column, WHITE)
        before = board_state(board)

        # Black has two next-move wins, but White is to move and can end the
        # game immediately. The black threat must not be reported as proven.
        result = ProofSearch().search(
            board,
            attacker=BLACK,
            side_to_move=WHITE,
        )

        self.assertIs(ProofState.PROVEN_LOSS, result.state)
        self.assertEqual((1, 4), result.best_move)
        self.assertEqual(before, board_state(board))

    def test_proven_result_can_be_reused_from_proof_tt(self) -> None:
        board = build_open_four_launch()
        table = ProofTable()
        searcher = ProofSearch(
            budget=ProofBudget(
                max_nodes=100,
                max_attacker_moves=1,
            ),
            table=table,
        )

        first = searcher.search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )
        second = searcher.search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertIs(ProofState.PROVEN_WIN, first.state)
        self.assertIs(ProofState.PROVEN_WIN, second.state)
        self.assertGreater(len(table), 0)
        self.assertEqual(1, second.transposition_hits)
        stats = table.stats()
        self.assertGreaterEqual(stats.queries, 2)
        self.assertGreaterEqual(stats.hits, 1)
        self.assertGreaterEqual(stats.stores, 1)
        self.assertEqual(len(table), stats.size)

    def test_shallow_proven_win_reuses_at_deeper_budget(self) -> None:
        board = build_open_four_launch()
        table = ProofTable()

        shallow = ProofSearch(
            budget=ProofBudget(
                max_nodes=100,
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
                max_nodes=100,
                max_attacker_moves=3,
            ),
            table=table,
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        stats = table.stats()
        self.assertIs(ProofState.PROVEN_WIN, shallow.state)
        self.assertIs(ProofState.PROVEN_WIN, deeper.state)
        self.assertEqual(1, deeper.transposition_hits)
        self.assertGreaterEqual(stats.compatible_hits, 1)

    def test_unknown_proof_tt_store_is_counted_but_not_cached(
        self,
    ) -> None:
        board = build_open_four_launch()
        table = ProofTable()

        result = ProofSearch(
            budget=ProofBudget(
                max_nodes=100,
                max_attacker_moves=0,
            ),
            table=table,
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        stats = table.stats()
        self.assertIs(ProofState.UNKNOWN, result.state)
        self.assertEqual(0, stats.stores)
        self.assertEqual(1, stats.skipped_stores)
        self.assertEqual(0, stats.size)

    def test_proof_tt_is_independent_from_pvs_tt(self) -> None:
        board = build_open_four_launch()
        pvs = SearchAI(player=BLACK)
        pvs._transposition_table[123] = object()  # type: ignore[assignment]
        before_pvs_table = dict(pvs._transposition_table)
        proof_table = ProofTable()

        ProofSearch(
            budget=ProofBudget(
                max_nodes=100,
                max_attacker_moves=1,
            ),
            table=proof_table,
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertEqual(before_pvs_table, pvs._transposition_table)
        self.assertGreater(len(proof_table), 0)
        self.assertIsNot(proof_table._entries, pvs._transposition_table)

    def test_exception_during_child_search_restores_board(self) -> None:
        class ExplodingProofSearch(ProofSearch):
            def __init__(self) -> None:
                super().__init__(
                    budget=ProofBudget(
                        max_nodes=100,
                        max_attacker_moves=1,
                    )
                )
                self.calls = 0

            def _search_node(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("test explosion")
                return super()._search_node(*args, **kwargs)

        board = build_open_four_launch()
        before = board_state(board)

        with self.assertRaisesRegex(RuntimeError, "test explosion"):
            ExplodingProofSearch().search(
                board,
                attacker=BLACK,
                side_to_move=BLACK,
            )

        self.assertEqual(before, board_state(board))


if __name__ == "__main__":
    unittest.main()
