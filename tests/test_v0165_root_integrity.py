from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from engine import root_candidates, root_policy
from engine.ai import DecisionAnalysis, DefenseCandidateAnalysis
from engine.board import BLACK, WHITE, Board
from engine.game import parse_move
from engine.proof_search import ProofResult, ProofState
from engine.search import SearchAI
from engine.search_types import (
    DefenseProbeResult,
    RootResult,
    SearchConfig,
)
from engine.threats import ThreatFrontier, ThreatKind


SELFPLAY_MOVE_21_PREFIX = (
    "H8 G7 I7 G9 G8 H9 I8 J8 F9 I6 E8 F8 E7 E10 H6 G10 "
    "F10 J9 E6 E9"
)


def replay_prefix(text: str) -> Board:
    board = Board()
    for index, coordinate in enumerate(text.split()):
        board.place(
            *parse_move(coordinate, board.size),
            BLACK if index % 2 == 0 else WHITE,
        )
    return board


def quiet_frontier(
    gain: tuple[int, int],
    *,
    ranks: tuple[int, ...],
) -> ThreatFrontier:
    continuations = tuple(
        (gain[0] + 1, gain[1] + index + 1)
        for index in range(len(ranks))
    )
    return ThreatFrontier(
        gain_move=gain,
        kind=ThreatKind.QUIET,
        continuations=continuations,
        continuation_kinds=tuple(ThreatKind.FOUR for _ in ranks),
        continuation_ranks=ranks,
    )


class TestV0165RootIntegrity(unittest.TestCase):
    def test_quiet_attack_frontier_is_bounded_without_lowering_strong_gate(
        self,
    ) -> None:
        strongest = quiet_frontier((1, 1), ranks=(60, 60, 40, 40, 40))
        second = quiet_frontier((2, 2), ranks=(60, 40, 40, 40))
        too_shallow = quiet_frontier((3, 3), ranks=(60, 40, 40))
        too_weak = quiet_frontier((4, 4), ranks=(40, 40, 40, 40))

        selected = root_candidates.quiet_attack_frontier_moves(
            frontiers=(too_weak, second, too_shallow, strongest),
            minimum_rank=60,
            minimum_continuations=4,
            limit=1,
        )

        self.assertEqual([strongest.gain_move], selected)

    def test_move_21_keeps_medium_quiet_attack_frontier_in_root(self) -> None:
        board = replay_prefix(SELFPLAY_MOVE_21_PREFIX)
        target = parse_move("G6", board.size)
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
        ai._begin_move_search()

        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )

        self.assertIn(target, plan.moves)
        self.assertIn(
            root_candidates.CandidateSource.QUIET_ATTACK_FRONTIER,
            ai._root_candidate_sources[target],
        )
        self.assertLessEqual(len(plan.moves), ai.config.root_candidate_limit)

    def test_post_filter_mandatory_root_gets_one_second_vct_gate(self) -> None:
        board = Board()
        moves = [(7, 7), (7, 8), (8, 8)]
        probe = DefenseProbeResult(
            completed_depth=3,
            nodes=3,
            candidates=tuple(
                DefenseCandidateAnalysis(
                    move=move,
                    score=100 - index,
                    status="survives_probe",
                )
                for index, move in enumerate(reversed(moves))
            ),
        )
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
        ai._root_candidate_sources = {
            moves[0]: frozenset(
                {root_candidates.CandidateSource.MANDATORY_DEFENSE}
            ),
            moves[1]: frozenset(
                {root_candidates.CandidateSource.ROOT_EXPANSION}
            ),
            moves[2]: frozenset(
                {root_candidates.CandidateSource.ROOT_EXPANSION}
            ),
        }

        with patch.object(
            ai,
            "_run_defense_vct_probe",
            return_value=probe,
        ) as run_probe:
            ordered, actual = ai._maybe_run_post_filter_defense_probe(
                board,
                moves,
                candidate_mode=(
                    root_candidates.RootCandidateMode.MANDATORY_DEFENSE
                ),
                existing_probe=None,
            )

        self.assertIs(probe, actual)
        run_probe.assert_called_once_with(board, BLACK, moves)
        self.assertEqual(list(reversed(moves)), ordered)

    def test_post_filter_vct_does_not_repeat_or_broaden_ordinary_roots(
        self,
    ) -> None:
        board = Board()
        moves = [(7, 7), (7, 8), (8, 8)]
        existing = DefenseProbeResult(0, 0, ())
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)

        with patch.object(ai, "_run_defense_vct_probe") as run_probe:
            unchanged, actual = ai._maybe_run_post_filter_defense_probe(
                board,
                moves,
                candidate_mode=root_candidates.RootCandidateMode.ORDINARY,
                existing_probe=existing,
            )

        self.assertEqual(moves, unchanged)
        self.assertIs(existing, actual)
        run_probe.assert_not_called()

    def test_risk_override_borrows_only_a_bounded_final_proof_slice(
        self,
    ) -> None:
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=60.0)
        ai.config = replace(
            ai.config,
            root_risk_override_shared_fraction=0.5,
        )
        ai._begin_move_search()

        with patch.object(
            ai,
            "_final_proof_reserve_seconds",
            return_value=6.0,
        ):
            budget = ai._risk_override_budget_seconds()

        self.assertGreaterEqual(budget, ai.config.root_safety_min_seconds)
        self.assertLessEqual(budget, 3.0)

    def test_proof_candidate_slice_has_an_explicit_upper_bound(self) -> None:
        self.assertEqual(
            2.0,
            root_policy.proof_candidate_slice_seconds(
                remaining_seconds=8.0,
                checks_left=2,
                maximum_seconds=2.0,
            ),
        )
        with self.assertRaises(ValueError):
            SearchConfig(proof_candidate_max_seconds=0.0)

    def test_final_proof_reserve_scales_with_effective_root(self) -> None:
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=60.0)
        ai._begin_move_search()
        ai._set_final_proof_expected_candidates([(7, 7), (7, 8)])

        self.assertEqual(4.0, ai._final_proof_reserve_seconds())
        self.assertLessEqual(
            ai._proof_budget_seconds(),
            ai.config.proof_initial_max_seconds,
        )

    def test_final_proof_selection_basis_is_serialized(self) -> None:
        payload = DecisionAnalysis(
            selected_move=(7, 7),
            reason="test",
            candidate_count=1,
            final_proof_selection_basis="emergency_unknown",
        ).to_dict()

        self.assertEqual(
            "emergency_unknown",
            payload["final_proof_selection_basis"],
        )

    def test_unchecked_certificate_intercept_is_explicit_unknown(self) -> None:
        board = Board()
        board.place(7, 7, BLACK)
        losing = (7, 8)
        intercept = (8, 8)
        root = RootResult(
            move=losing,
            score=10_000,
            principal_variation=(losing,),
            ranked_moves=((losing, 10_000),),
            ranked_variations=((losing, 10_000, (losing,)),),
        )
        ai = SearchAI(BLACK, time_limit_seconds=60.0)
        ai.config = replace(ai.config, proof_final_candidate_limit=1)
        ai._begin_move_search()
        ai._root_candidate_sources = {
            losing: frozenset({root_candidates.CandidateSource.ORDINARY})
        }

        class LosingProofSearch:
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
                    state=ProofState.PROVEN_WIN,
                    attacker=attacker,
                    side_to_move=side_to_move,
                    best_move=intercept,
                    principal_variation=(intercept,),
                    required_defenses=(),
                    nodes=1,
                    transposition_hits=0,
                    searched_attacker_moves=1,
                    completed=True,
                    cutoff_reason=None,
                    elapsed_seconds=0.0,
                )

        with (
            patch("engine.search.ProofSearch", LosingProofSearch),
            patch.object(ai, "_final_proof_budget_seconds", return_value=1.0),
        ):
            revised = ai._run_final_proof_audit(board, root)

        analyses = {item.move: item for item in ai._proof_candidates}
        self.assertEqual(intercept, revised.move)
        self.assertIn(
            root_candidates.CandidateSource.CERTIFICATE_INTERCEPT,
            ai._root_candidate_sources[intercept],
        )
        self.assertEqual(ProofState.UNKNOWN.value, analyses[intercept].state)
        self.assertFalse(analyses[intercept].completed)
        self.assertEqual(
            "final_proof_budget_exhausted",
            analyses[intercept].cutoff_reason,
        )
        self.assertEqual("emergency_unknown", ai._final_proof_selection_basis)
        self.assertEqual(ProofState.UNKNOWN.value, ai._final_proof_state)


if __name__ == "__main__":
    unittest.main()
