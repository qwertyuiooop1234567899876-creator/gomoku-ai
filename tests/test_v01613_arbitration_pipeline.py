from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from engine import root_candidates
from engine.ai import RootSafetyCandidateAnalysis
from engine.board import BLACK, Board
from engine.proof_search import ProofResult, ProofState
from engine.search import SearchAI
from engine.search_diagnostics import build_search_analysis
from engine.search_types import RootResult, RootSafetyProbeResult


class UnknownProofSearch:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def search_after_move(
        self,
        _board: Board,
        *,
        move: tuple[int, int],
        mover: int,
        attacker: int,
        side_to_move: int,
    ) -> ProofResult:
        return ProofResult(
            state=ProofState.UNKNOWN,
            attacker=attacker,
            side_to_move=side_to_move,
            best_move=None,
            principal_variation=(move,),
            required_defenses=(),
            nodes=1,
            transposition_hits=0,
            searched_attacker_moves=1,
            completed=False,
            cutoff_reason="deadline",
            elapsed_seconds=0.0,
        )


class StrictProofSearch:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def search_after_move(
        self,
        _board: Board,
        *,
        move: tuple[int, int],
        mover: int,
        attacker: int,
        side_to_move: int,
    ) -> ProofResult:
        state = (
            ProofState.PROVEN_WIN
            if move == (7, 8)
            else ProofState.PROVEN_LOSS
        )
        return ProofResult(
            state=state,
            attacker=attacker,
            side_to_move=side_to_move,
            best_move=None,
            principal_variation=(move,),
            required_defenses=(),
            nodes=1,
            transposition_hits=0,
            searched_attacker_moves=1,
            completed=True,
            cutoff_reason=None,
            elapsed_seconds=0.0,
        )


def root_result(
    leader: tuple[int, int],
    challenger: tuple[int, int],
    later: tuple[int, int] | None = None,
) -> RootResult:
    ranked = [(leader, 100), (challenger, 100)]
    if later is not None:
        ranked.append((later, 90))
    return RootResult(leader, 100, (leader,), tuple(ranked))


def stable_probe(
    leader: tuple[int, int],
    challenger: tuple[int, int],
    *,
    depth: int = 3,
) -> RootSafetyProbeResult:
    return RootSafetyProbeResult(
        trigger="dynamic_remaining_review",
        pvs_gap=0,
        main_rank_stable=True,
        completed_depth=depth,
        nodes=10,
        candidates=(
            RootSafetyCandidateAnalysis(leader, 20_200),
            RootSafetyCandidateAnalysis(challenger, 10_200),
        ),
        leader_history=(leader, leader),
        approved_move=leader,
        selection_basis="equal_window",
        requested_budget_seconds=1.0,
    )


class TestV01613ArbitrationPipeline(unittest.TestCase):
    def test_later_timeout_keeps_last_completed_pair(self) -> None:
        board = Board()
        leader = (7, 7)
        challenger = (7, 8)
        later = (8, 7)
        result = root_result(leader, challenger, later)
        first = stable_probe(leader, challenger)
        ai = SearchAI(BLACK, time_limit_seconds=60.0)
        ai._begin_move_search()

        with (
            patch.object(
                ai,
                "_dynamic_review_budget_seconds",
                return_value=2.0,
            ),
            patch(
                "engine.search.root_review.review_pool",
                return_value=[leader, challenger, later],
            ),
            patch(
                "engine.search.root_review.finalists",
                return_value=[leader, challenger, later],
            ),
            patch.object(
                ai,
                "_run_dynamic_pair_review",
                side_effect=(first, None),
            ) as run_pair,
        ):
            actual = ai._maybe_run_dynamic_root_review(
                board,
                result,
                [leader, challenger, later],
                completed_depth=6,
            )

        self.assertEqual(2, run_pair.call_count)
        self.assertIsNotNone(actual)
        assert actual is not None
        self.assertEqual(leader, actual.approved_move)
        self.assertEqual("equal_window", actual.selection_basis)

    def test_confirmed_review_blocks_unknown_pressure_tiebreak(self) -> None:
        board = Board()
        board.place(7, 7, BLACK)
        leader = (7, 8)
        pressure = (8, 7)
        ai = SearchAI(BLACK, time_limit_seconds=60.0)
        ai.config = replace(ai.config, proof_final_candidate_limit=2)
        ai._begin_move_search()
        ai._root_pressure_prevention = (pressure,)
        ai._root_candidate_sources = {
            leader: frozenset(
                {root_candidates.CandidateSource.ORDINARY}
            ),
            pressure: frozenset(
                {root_candidates.CandidateSource.PRESSURE_PREVENTION}
            ),
        }
        ai._root_review_confirmed_move = leader

        with patch("engine.search.ProofSearch", UnknownProofSearch):
            revised = ai._run_final_proof_audit(
                board,
                root_result(leader, pressure),
            )

        self.assertEqual(leader, revised.move)
        self.assertEqual(
            "checked_unknown_review_confirmed",
            ai._final_proof_selection_basis,
        )
        self.assertFalse(ai._final_proof_overrode_review)

    def test_shallow_or_boundary_probe_does_not_confirm_leader(self) -> None:
        leader = (7, 8)
        challenger = (8, 7)
        ai = SearchAI(BLACK, time_limit_seconds=60.0)
        ai._begin_move_search()
        result = root_result(leader, challenger)

        ai._register_root_review_confirmation(
            result,
            stable_probe(leader, challenger, depth=2),
        )
        self.assertIsNone(ai._root_review_confirmed_move)

        boundary = replace(
            stable_probe(leader, challenger),
            candidates=(
                RootSafetyCandidateAnalysis(leader, -100_000_000),
                RootSafetyCandidateAnalysis(challenger, -100_000_000),
            ),
        )
        ai._register_root_review_confirmation(result, boundary)
        self.assertIsNone(ai._root_review_confirmed_move)

    def test_strict_proof_can_still_override_confirmed_review(self) -> None:
        board = Board()
        board.place(7, 7, BLACK)
        leader = (7, 8)
        survivor = (8, 7)
        ai = SearchAI(BLACK, time_limit_seconds=60.0)
        ai.config = replace(ai.config, proof_final_candidate_limit=2)
        ai._begin_move_search()
        ai._root_review_confirmed_move = leader

        with patch("engine.search.ProofSearch", StrictProofSearch):
            revised = ai._run_final_proof_audit(
                board,
                root_result(leader, survivor),
            )

        self.assertEqual(survivor, revised.move)
        self.assertEqual("strict_survivor", ai._final_proof_selection_basis)
        self.assertTrue(ai._final_proof_overrode_review)

    def test_stage_diagnostics_explain_confirmation_without_change(self) -> None:
        leader = (7, 8)
        challenger = (8, 7)
        ai = SearchAI(BLACK, time_limit_seconds=60.0, diagnostics=True)
        ai._begin_move_search()
        result = root_result(leader, challenger)
        probe = stable_probe(leader, challenger)

        revised = ai._apply_and_record_root_safety(result, probe)
        payload = build_search_analysis(
            ai,
            selected_move=revised.move,
            reason="test",
            candidate_count=2,
            ranked_moves=list(revised.ranked_moves),
            completed_depth=3,
            principal_variation=revised.principal_variation,
            search_completed=False,
            stop_reason="test",
        ).to_dict()

        self.assertEqual("I8", payload["root_review_incoming_coordinate"])
        self.assertEqual("I8", payload["root_review_approved_coordinate"])
        self.assertFalse(payload["root_review_result_changed"])
        self.assertEqual(
            "confirmed_current",
            payload["root_review_apply_reason"],
        )
        self.assertEqual("I8", payload["root_review_confirmed_coordinate"])


if __name__ == "__main__":
    unittest.main()
