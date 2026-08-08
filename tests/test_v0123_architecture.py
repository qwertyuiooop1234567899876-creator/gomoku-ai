import inspect
import unittest

from engine import root_candidates, root_policy, root_safety
from engine.board import BLACK, Board
from engine.proof_search import ProofState
from engine.search import RootResult, SearchAI, SearchConfig
from engine.search_types import (
    RootResult as SharedRootResult,
    SearchConfig as SharedSearchConfig,
)
from engine.vcf import VCFSearch


class TestV0123SearchContracts(unittest.TestCase):
    def test_legacy_search_imports_reexport_shared_types(self) -> None:
        self.assertIs(SearchConfig, SharedSearchConfig)
        self.assertIs(RootResult, SharedRootResult)

    def test_choose_move_remains_a_small_coordinator(self) -> None:
        source = inspect.getsource(SearchAI.choose_move)
        self.assertLessEqual(len(source.splitlines()), 160)
        self.assertIn("_try_tactical_shortcut", source)
        self.assertIn("_prepare_root_candidate_plan", source)
        self.assertIn("_run_iterative_root_search", source)

    def test_vcf_search_is_not_embedded_in_search_ai(self) -> None:
        self.assertNotIn("_vcf_search", SearchAI.__dict__)
        self.assertTrue(callable(VCFSearch.find))


class TestV0123CandidatePolicy(unittest.TestCase):
    def test_candidate_modes_are_classified_without_scoring(self) -> None:
        self.assertIs(
            root_candidates.classify_mode(
                own_forcing_moves=[(1, 1)],
                opponent_forcing_moves=[(2, 2)],
                opponent_frontier_moves=[],
            ),
            root_candidates.RootCandidateMode.MERGED_FORCING,
        )
        self.assertIs(
            root_candidates.classify_mode(
                own_forcing_moves=[],
                opponent_forcing_moves=[],
                opponent_frontier_moves=[(3, 3)],
            ),
            root_candidates.RootCandidateMode.FRONTIER_DEFENSE,
        )

    def test_candidate_provenance_preserves_order_and_sources(self) -> None:
        entries = root_candidates.with_sources(
            (
                (
                    root_candidates.CandidateSource.MANDATORY_DEFENSE,
                    [(2, 2), (3, 3)],
                ),
                (
                    root_candidates.CandidateSource.OWN_FORCING,
                    [(3, 3), (4, 4)],
                ),
            )
        )
        self.assertEqual(
            [(2, 2), (3, 3), (4, 4)],
            [entry.move for entry in entries],
        )
        self.assertEqual(
            frozenset(
                {
                    root_candidates.CandidateSource.MANDATORY_DEFENSE,
                    root_candidates.CandidateSource.OWN_FORCING,
                }
            ),
            entries[1].sources,
        )


class TestV0123RootPolicy(unittest.TestCase):
    def test_quarantine_uses_one_explicit_heuristic_scale(self) -> None:
        first = (1, 1)
        second = (2, 2)
        result = RootResult(
            move=first,
            score=999_999_994,
            principal_variation=(first,),
            ranked_moves=(
                (first, 999_999_994),
                (second, 50_000),
            ),
            ranked_variations=(
                (first, 999_999_994, (first,)),
                (second, 50_000, (second,)),
            ),
        )
        revised, quarantined = (
            root_policy.quarantine_unproven_scores(
                result,
                proof_states={
                    first: ProofState.UNKNOWN.value,
                    second: ProofState.UNKNOWN.value,
                },
                heuristic_score=lambda move: {
                    first: 100,
                    second: 200,
                }[move],
            )
        )
        self.assertTrue(quarantined)
        self.assertEqual(second, revised.move)
        self.assertEqual(((second, 200), (first, 100)), revised.ranked_moves)

    def test_root_safety_states_are_not_proof_aliases(self) -> None:
        self.assertEqual(
            {
                "proven_loss",
                "survives_vcf_scan",
                "unknown",
            },
            {state.value for state in root_safety.RootCandidateSafety},
        )

    def test_unknown_risk_needs_unknown_on_both_moves(self) -> None:
        first = (1, 1)
        second = (2, 2)
        pvs = RootResult(first, 10, (first,), ((first, 10), (second, 9)))
        revised = RootResult(
            second,
            9,
            (second,),
            ((second, 9), (first, 10)),
        )
        self.assertTrue(
            root_policy.is_unknown_risk_override(
                pvs,
                revised,
                proof_states={
                    first: ProofState.UNKNOWN.value,
                    second: ProofState.UNKNOWN.value,
                },
            )
        )
        self.assertFalse(
            root_policy.is_unknown_risk_override(
                pvs,
                revised,
                proof_states={
                    first: ProofState.PROVEN_LOSS.value,
                    second: ProofState.UNKNOWN.value,
                },
            )
        )


class TestV0123VCFModule(unittest.TestCase):
    def test_immediate_vcf_result_restores_board(self) -> None:
        board = Board()
        for column in range(5, 9):
            board.place(7, column, BLACK)
        before = (
            tuple(tuple(row) for row in board.grid),
            tuple(board.move_history),
            board.zobrist_hash,
            board.empty_count,
        )
        nodes = [0]
        search = VCFSearch(
            position_key=lambda position, player: (
                position.zobrist_hash ^ player
            ),
            forcing_candidates=lambda _board, _player: [],
            check_timeout=lambda: None,
            count_node=lambda: nodes.__setitem__(0, nodes[0] + 1),
        )

        line = search.find(board, BLACK, 3)

        self.assertIsNotNone(line)
        self.assertEqual(1, len(line))
        self.assertEqual(1, nodes[0])
        after = (
            tuple(tuple(row) for row in board.grid),
            tuple(board.move_history),
            board.zobrist_hash,
            board.empty_count,
        )
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
