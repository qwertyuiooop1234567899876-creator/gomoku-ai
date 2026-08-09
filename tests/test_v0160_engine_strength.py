from __future__ import annotations

import unittest
from unittest.mock import patch

from engine import root_candidates
from engine.ai import DecisionAnalysis
from engine.board import BLACK, WHITE, Board
from engine.proof_search import (
    ProofBudget,
    ProofKey,
    ProofSearch,
    ProofState,
    ProofTable,
)
from engine.search import SearchAI


def board_state(board: Board) -> tuple[object, ...]:
    return (
        tuple(tuple(row) for row in board.grid),
        tuple(board.move_history),
        board.zobrist_hash,
        board.empty_count,
    )


class TestV0160ProofContinuation(unittest.TestCase):
    def test_hint_rotates_interrupted_move_without_caching_unknown(self) -> None:
        table = ProofTable()
        key = ProofKey(123, BLACK, BLACK, (), 8, 1)
        moves = ((1, 1), (2, 2), (3, 3))

        table.defer_move(key, moves[0], generation=table.next_generation())

        self.assertEqual((moves[1], moves[2], moves[0]), table.order_moves(key, moves))
        self.assertIsNone(table.get(key))
        self.assertEqual(0, len(table))
        stats = table.stats()
        self.assertEqual(1, stats.hint_hits)
        self.assertEqual(1, stats.hint_stores)
        self.assertEqual(1, stats.hint_size)

    def test_repeated_bounded_proof_reenters_a_different_root_branch(self) -> None:
        board = Board(size=9)
        for column in (2, 3, 4):
            board.place(4, column, BLACK)
        board.place(4, 1, WHITE)
        before = board_state(board)
        table = ProofTable()

        results = [
            ProofSearch(
                budget=ProofBudget(
                    max_nodes=2,
                    max_attacker_moves=2,
                ),
                table=table,
            ).search(
                board,
                attacker=BLACK,
                side_to_move=BLACK,
            )
            for _ in range(2)
        ]

        self.assertTrue(all(result.state is ProofState.UNKNOWN for result in results))
        self.assertTrue(all(result.cutoff_reason == "node_limit" for result in results))
        self.assertGreaterEqual(table.stats().hint_hits, 1)
        self.assertEqual(before, board_state(board))

    def test_proof_generations_are_shared_by_the_table(self) -> None:
        table = ProofTable()

        self.assertEqual(1, table.next_generation())
        self.assertEqual(2, table.next_generation())


class TestV0160CandidateDiversity(unittest.TestCase):
    def test_overfull_required_set_keeps_every_source_represented(self) -> None:
        frontier = tuple((0, column) for column in range(8))
        forcing = tuple((1, column) for column in range(4))
        prevention = ((2, 0),)
        bridge = ((3, 0),)
        counterattack = ((4, 0),)

        merged = root_candidates.merge_with_required(
            ordered_groups=(
                frontier,
                forcing,
                prevention,
                bridge,
                counterattack,
            ),
            required_groups=(
                frontier,
                forcing,
                prevention,
                bridge,
                counterattack,
            ),
            limit=10,
        )

        self.assertEqual(10, len(merged))
        for group in (
            frontier,
            forcing,
            prevention,
            bridge,
            counterattack,
        ):
            self.assertTrue(set(group).intersection(merged))

    def test_initial_proof_slots_cover_distinct_tactical_sources(self) -> None:
        moves = tuple((5, column) for column in range(7))
        frontier = root_candidates.CandidateSource.THREAT_FRONTIER
        forcing = root_candidates.CandidateSource.OWN_FORCING
        prevention = root_candidates.CandidateSource.QUIET_PREVENTION
        bridge = root_candidates.CandidateSource.DUAL_FRONTIER_BRIDGE
        sources = {
            moves[0]: frozenset({frontier}),
            moves[1]: frozenset({frontier}),
            moves[2]: frozenset({frontier}),
            moves[3]: frozenset({forcing}),
            moves[4]: frozenset({prevention}),
            moves[5]: frozenset({bridge}),
            moves[6]: frozenset({root_candidates.CandidateSource.ORDINARY}),
        }

        selected = root_candidates.source_diverse_subset(
            moves,
            sources,
            limit=4,
        )

        self.assertEqual(moves[0], selected[0])
        self.assertEqual(
            {moves[0], moves[3], moves[4], moves[5]},
            set(selected),
        )

    def test_unprobed_diverse_root_candidates_remain_explicit_unknown(self) -> None:
        board = Board()
        moves = [(5, column) for column in range(7)]
        ai = SearchAI(player=BLACK)
        ai._root_candidate_sources = {
            move: frozenset({root_candidates.CandidateSource.ORDINARY})
            for move in moves
        }

        with (
            patch.object(ai, "_proof_budget_seconds", return_value=0.000_001),
            patch.object(ai, "_threat_risk_after_move", return_value=0),
        ):
            ai._run_proof_arbitration(
                board,
                moves,
                search_own_win=False,
            )

        self.assertEqual(set(moves), {item.move for item in ai._proof_candidates})
        unprobed = [
            item
            for item in ai._proof_candidates
            if item.cutoff_reason == "initial_proof_unprobed"
        ]
        self.assertEqual(3, len(unprobed))
        self.assertTrue(all(item.state == ProofState.UNKNOWN.value for item in unprobed))

    def test_candidate_sources_are_serialized_for_future_record_audits(self) -> None:
        move = (4, 4)
        payload = DecisionAnalysis(
            selected_move=move,
            reason="test",
            candidate_count=1,
            root_candidate_sources=(
                (move, ("mandatory_defense", "quiet_prevention")),
            ),
            proof_hint_queries=3,
            proof_hint_hits=1,
        ).to_dict()

        self.assertEqual(
            ["mandatory_defense", "quiet_prevention"],
            payload["root_candidate_sources"][0]["sources"],
        )
        self.assertEqual(3, payload["proof_hint_queries"])
        self.assertEqual(1, payload["proof_hint_hits"])


if __name__ == "__main__":
    unittest.main()
