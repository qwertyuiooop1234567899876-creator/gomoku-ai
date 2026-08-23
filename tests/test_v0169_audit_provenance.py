from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from engine.ai import ProofCandidateAnalysis, RootVCFCandidateAnalysis
from engine.board import BLACK, WHITE, Board
from engine.proof_search import ProofResult, ProofState
from engine.search import SearchAI
from engine.search_diagnostics import build_search_analysis
from engine.search_types import RootResult


def proof_result(
    state: ProofState,
    *,
    principal_variation: tuple[tuple[int, int], ...] = (),
) -> ProofResult:
    return ProofResult(
        state=state,
        attacker=WHITE,
        side_to_move=WHITE,
        best_move=(
            principal_variation[0] if principal_variation else None
        ),
        principal_variation=principal_variation,
        required_defenses=(),
        nodes=1,
        transposition_hits=0,
        searched_attacker_moves=1,
        completed=state is not ProofState.UNKNOWN,
        cutoff_reason=(
            "deadline" if state is ProofState.UNKNOWN else None
        ),
        elapsed_seconds=0.0,
    )


def root_result(move: tuple[int, int]) -> RootResult:
    return RootResult(
        move=move,
        score=100,
        principal_variation=(move,),
        ranked_moves=((move, 100),),
    )


def build_payload(ai: SearchAI, selected: tuple[int, int]) -> dict[str, object]:
    return build_search_analysis(
        ai,
        selected_move=selected,
        reason="test",
        candidate_count=1,
        ranked_moves=[(selected, 100)],
        completed_depth=1,
        principal_variation=(selected,),
        search_completed=True,
    ).to_dict()


class TestFinalProofEmergencyVCFProvenance(unittest.TestCase):
    def _ai(self, *, diagnostics: bool = True) -> SearchAI:
        ai = SearchAI(
            BLACK,
            max_depth=2,
            time_limit_seconds=60.0,
            diagnostics=diagnostics,
        )
        ai.config = replace(ai.config, proof_final_candidate_limit=1)
        ai._begin_move_search()
        return ai

    def test_unattempted_gate_is_explicit_and_serializable(self) -> None:
        board = Board()
        move = (7, 7)
        ai = self._ai()
        ai._proof_candidates = (
            ProofCandidateAnalysis(
                move=move,
                state=ProofState.PROVEN_LOSS.value,
                completed=True,
                nodes=1,
                elapsed_seconds=0.0,
            ),
        )

        with patch.object(ai, "_final_proof_budget_seconds", return_value=1.0):
            revised = ai._run_final_proof_audit(board, root_result(move))

        provenance = ai._final_proof_emergency_vcf
        self.assertIsNotNone(provenance)
        assert provenance is not None
        self.assertFalse(provenance.attempted)
        self.assertGreater(provenance.reserved_seconds, 0.0)
        self.assertEqual(0.0, provenance.used_seconds)
        self.assertEqual((), provenance.candidates)
        self.assertEqual(move, provenance.selected_before)
        self.assertEqual(move, provenance.selected_after)
        self.assertFalse(provenance.changed_selection)
        self.assertEqual(move, revised.move)

        payload = build_payload(ai, revised.move)
        serialized = payload["final_proof_emergency_vcf"]
        assert isinstance(serialized, dict)
        self.assertFalse(serialized["attempted"])
        self.assertEqual([], serialized["candidates"])

    def test_unattempted_gate_ignores_an_earlier_final_proof_change(
        self,
    ) -> None:
        board = Board()
        leader = (7, 7)
        strict_survivor = (7, 8)
        ai = self._ai()
        ai._proof_candidates = (
            ProofCandidateAnalysis(
                move=leader,
                state=ProofState.PROVEN_WIN.value,
                completed=True,
                nodes=1,
                elapsed_seconds=0.0,
            ),
            ProofCandidateAnalysis(
                move=strict_survivor,
                state=ProofState.PROVEN_LOSS.value,
                completed=True,
                nodes=1,
                elapsed_seconds=0.0,
            ),
        )
        result = RootResult(
            move=leader,
            score=100,
            principal_variation=(leader,),
            ranked_moves=((leader, 100), (strict_survivor, 90)),
        )

        with patch.object(ai, "_final_proof_budget_seconds", return_value=1.0):
            revised = ai._run_final_proof_audit(board, result)

        provenance = ai._final_proof_emergency_vcf
        self.assertIsNotNone(provenance)
        assert provenance is not None
        self.assertEqual(strict_survivor, revised.move)
        self.assertFalse(provenance.attempted)
        self.assertEqual(strict_survivor, provenance.selected_before)
        self.assertEqual(strict_survivor, provenance.selected_after)
        self.assertFalse(provenance.changed_selection)

    def test_unknown_emergency_gate_stays_unknown_and_is_recorded(self) -> None:
        board = Board()
        leader = (7, 7)
        emergency = (7, 8)
        ai = self._ai()

        class LosingProofSearch:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def search_after_move(
                self,
                _board: Board,
                **_kwargs: object,
            ) -> ProofResult:
                return proof_result(
                    ProofState.PROVEN_WIN,
                    principal_variation=(emergency,),
                )

        class UnknownScanner:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def scan_candidate(
                self,
                _board: Board,
                move: tuple[int, int],
                **_kwargs: object,
            ) -> RootVCFCandidateAnalysis:
                return RootVCFCandidateAnalysis(
                    move=move,
                    status="unknown",
                    completed=False,
                    nodes=7,
                    elapsed_seconds=0.01,
                )

        with (
            patch("engine.search.ProofSearch", LosingProofSearch),
            patch(
                "engine.search.root_safety.RootVCFSafetyScanner",
                UnknownScanner,
            ),
            patch.object(ai, "_final_proof_budget_seconds", return_value=2.0),
        ):
            revised = ai._run_final_proof_audit(board, root_result(leader))

        provenance = ai._final_proof_emergency_vcf
        self.assertIsNotNone(provenance)
        assert provenance is not None
        self.assertTrue(provenance.attempted)
        self.assertEqual(1, provenance.candidate_count)
        self.assertEqual(1, len(provenance.candidates))
        candidate = provenance.candidates[0]
        self.assertEqual(emergency, candidate.move)
        self.assertEqual("unknown", candidate.status)
        self.assertFalse(candidate.completed)
        self.assertEqual("deadline", candidate.cutoff_reason)
        self.assertEqual(emergency, revised.move)
        self.assertEqual(emergency, provenance.selected_before)
        self.assertEqual(emergency, provenance.selected_after)
        self.assertFalse(provenance.changed_selection)
        self.assertEqual(ProofState.UNKNOWN.value, ai._final_proof_state)
        self.assertFalse(ai._final_proof_completed)

    def test_emergency_gate_skips_a_proven_loss_fallback(self) -> None:
        board = Board()
        leader = (7, 7)
        first_emergency = (7, 8)
        second_emergency = (8, 7)
        ai = self._ai()

        class LosingProofSearch:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def search_after_move(
                self,
                _board: Board,
                **_kwargs: object,
            ) -> ProofResult:
                return proof_result(
                    ProofState.PROVEN_WIN,
                    principal_variation=(first_emergency, second_emergency),
                )

        class FirstLosingScanner:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def scan_candidate(
                self,
                _board: Board,
                move: tuple[int, int],
                **_kwargs: object,
            ) -> RootVCFCandidateAnalysis:
                return RootVCFCandidateAnalysis(
                    move=move,
                    status=(
                        "proven_loss"
                        if move == first_emergency
                        else "unknown"
                    ),
                    completed=move == first_emergency,
                    nodes=1,
                    elapsed_seconds=0.0,
                )

        with (
            patch("engine.search.ProofSearch", LosingProofSearch),
            patch(
                "engine.search.root_safety.RootVCFSafetyScanner",
                FirstLosingScanner,
            ),
            patch.object(ai, "_final_proof_budget_seconds", return_value=2.0),
        ):
            revised = ai._run_final_proof_audit(board, root_result(leader))

        provenance = ai._final_proof_emergency_vcf
        self.assertIsNotNone(provenance)
        assert provenance is not None
        self.assertTrue(provenance.attempted)
        self.assertEqual(first_emergency, provenance.selected_before)
        self.assertEqual(second_emergency, provenance.selected_after)
        self.assertTrue(provenance.changed_selection)
        self.assertEqual(second_emergency, revised.move)

    def test_candidate_provenance_is_bounded_with_explicit_truncation(self) -> None:
        board = Board()
        leader = (7, 7)
        intercepts = ((7, 8), (8, 7), (8, 8))
        ai = self._ai()

        class LosingProofSearch:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def search_after_move(
                self,
                _board: Board,
                **_kwargs: object,
            ) -> ProofResult:
                return proof_result(
                    ProofState.PROVEN_WIN,
                    principal_variation=intercepts,
                )

        class LosingScanner:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def scan_candidate(
                self,
                _board: Board,
                move: tuple[int, int],
                **_kwargs: object,
            ) -> RootVCFCandidateAnalysis:
                return RootVCFCandidateAnalysis(
                    move=move,
                    status="proven_loss",
                    completed=True,
                    nodes=1,
                    elapsed_seconds=0.0,
                )

        with (
            patch("engine.search.ProofSearch", LosingProofSearch),
            patch(
                "engine.search.root_safety.RootVCFSafetyScanner",
                LosingScanner,
            ),
            patch.object(ai, "_final_proof_budget_seconds", return_value=2.0),
        ):
            ai._run_final_proof_audit(board, root_result(leader))

        provenance = ai._final_proof_emergency_vcf
        self.assertIsNotNone(provenance)
        assert provenance is not None
        self.assertEqual(len(intercepts), provenance.candidate_count)
        self.assertEqual(1, len(provenance.candidates))
        self.assertTrue(provenance.candidates_truncated)


class TestRootReviewUnpairedFinalists(unittest.TestCase):
    def test_triggered_unpaired_audit_is_compact_and_serializable(self) -> None:
        leader = (7, 7)
        skipped = (7, 8)
        later = (8, 7)
        ai = SearchAI(BLACK, time_limit_seconds=None, diagnostics=False)
        ai._begin_move_search()
        ai._record_root_review_unpaired_finalists(
            [leader, skipped, later],
            entered_moves={leader},
            reason="insufficient_pair_budget",
            remaining_budget_seconds=0.25,
        )

        payload = build_payload(ai, leader)
        self.assertEqual([], payload["root_review_finalists"])
        unpaired = payload["root_review_unpaired_finalists"]
        assert isinstance(unpaired, list)
        self.assertEqual(2, len(unpaired))
        self.assertEqual("I8", unpaired[0]["coordinate"])
        self.assertEqual("H9", unpaired[1]["coordinate"])
        self.assertTrue(
            all(
                item["reason"] == "insufficient_pair_budget"
                for item in unpaired
            )
        )
        self.assertTrue(
            all(item["remaining_budget_seconds"] == 0.25 for item in unpaired)
        )

    def test_dynamic_review_budget_break_records_every_unentered_finalist(
        self,
    ) -> None:
        board = Board()
        leader = (7, 7)
        challenger = (7, 8)
        skipped = (8, 7)
        result = RootResult(
            move=leader,
            score=100,
            principal_variation=(leader,),
            ranked_moves=(
                (leader, 100),
                (challenger, 90),
                (skipped, 80),
            ),
        )
        ai = SearchAI(BLACK, time_limit_seconds=60.0, diagnostics=False)
        ai._begin_move_search()

        with (
            patch.object(
                ai,
                "_dynamic_review_budget_seconds",
                return_value=0.4,
            ),
            patch(
                "engine.search.root_review.review_pool",
                return_value=[leader, challenger, skipped],
            ),
            patch(
                "engine.search.root_review.finalists",
                return_value=[leader, challenger, skipped],
            ),
        ):
            probe = ai._maybe_run_dynamic_root_review(
                board,
                result,
                [leader, challenger, skipped],
                completed_depth=3,
            )

        self.assertIsNone(probe)
        self.assertEqual(
            (leader, challenger, skipped),
            tuple(item.move for item in ai._root_review_unpaired_finalists),
        )
        self.assertTrue(
            all(
                item.reason == "insufficient_pair_budget"
                for item in ai._root_review_unpaired_finalists
            )
        )
        payload = build_payload(ai, leader)
        serialized = payload["root_review_unpaired_finalists"]
        assert isinstance(serialized, list)
        self.assertEqual(3, len(serialized))
        self.assertTrue(
            all(
                item["reason"] == "insufficient_pair_budget"
                for item in serialized
            )
        )


if __name__ == "__main__":
    unittest.main()
