from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from engine import root_candidates, root_safety
from engine.ai import ProofCandidateAnalysis, RootSafetyCandidateAnalysis
from engine.board import BLACK, WHITE, Board
from engine.game import parse_move
from engine.proof_search import ProofResult, ProofState
from engine.search import SearchAI
from engine.search_diagnostics import build_search_analysis
from engine.search_types import (
    HEURISTIC_SCORE_LIMIT,
    RootResult,
    RootSafetyProbeResult,
    SearchConfig,
)
from engine.threats import ThreatFrontier, ThreatKind


YIXIN_MOVE_15_PREFIX = (
    "H8 I8 G7 I9 I7 G9 H9 H7 G6 H10 G5 G4 F8 H6"
).split()
SELFPLAY_MOVE_23_PREFIX = (
    "H8 G7 I7 G9 G8 I8 H9 H7 F7 I10 J9 F8 H10 J6 "
    "E6 D5 H11 H12 G6 E8 E9 D6"
).split()
SELFPLAY_MOVE_29_PREFIX = (
    "H8 G7 I7 G9 G8 I8 H9 H7 F7 I10 J9 F8 H10 J6 "
    "E6 D5 H11 H12 G6 E8 E9 D6 I9 J8 K9 L9 J10 D7"
).split()


def replay(coordinates: list[str]) -> Board:
    board = Board()
    for index, coordinate in enumerate(coordinates):
        board.place(
            *parse_move(coordinate, board.size),
            BLACK if index % 2 == 0 else WHITE,
        )
    return board


def review_probe(
    first: tuple[int, int],
    second: tuple[int, int],
    *,
    depth: int,
    first_score: int,
    second_score: int,
    leaders: tuple[tuple[int, int], ...],
    requested: float = 0.0,
) -> RootSafetyProbeResult:
    return RootSafetyProbeResult(
        trigger="dynamic_remaining_review",
        pvs_gap=0,
        main_rank_stable=True,
        completed_depth=depth,
        nodes=10,
        candidates=(
            RootSafetyCandidateAnalysis(first, first_score),
            RootSafetyCandidateAnalysis(second, second_score),
        ),
        leader_history=leaders,
        approved_move=first,
        selection_basis="pvs_fallback",
        requested_budget_seconds=requested,
    )


class TestV0168CandidateEvidence(unittest.TestCase):
    def test_broad_quiet_attack_gets_one_source_aware_root_slot(self) -> None:
        board = replay(YIXIN_MOVE_15_PREFIX)
        target = parse_move("F7", board.size)
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
        ai._begin_move_search()

        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )

        self.assertEqual(
            root_candidates.RootCandidateMode.FRONTIER_DEFENSE,
            plan.mode,
        )
        self.assertEqual(1, plan.moves.count(target))
        self.assertIn(
            root_candidates.CandidateSource.BROAD_QUIET_ATTACK,
            ai._root_candidate_sources[target],
        )
        self.assertIn(
            target,
            root_candidates.merge_unique(
                *ai._critical_root_review_groups(tuple(plan.moves))
            ),
        )
        self.assertLessEqual(len(plan.moves), ai.config.root_candidate_limit)

    def test_broad_quiet_attack_lane_keeps_breadth_and_total_gates(self) -> None:
        weak = ThreatFrontier(
            gain_move=(1, 1),
            kind=ThreatKind.QUIET,
            continuations=tuple((2, index) for index in range(7)),
            continuation_kinds=(ThreatKind.OPEN_THREE,) * 7,
            continuation_ranks=(40,) * 7,
        )
        broad = ThreatFrontier(
            gain_move=(3, 3),
            kind=ThreatKind.QUIET,
            continuations=tuple((4, index) for index in range(8)),
            continuation_kinds=(ThreatKind.OPEN_THREE,) * 8,
            continuation_ranks=(40,) * 8,
        )

        actual = root_candidates.broad_quiet_attack_frontier_moves(
            frontiers=(weak, broad),
            minimum_rank=40,
            minimum_continuations=8,
            minimum_total_rank=320,
            limit=1,
        )

        self.assertEqual([broad.gain_move], actual)

    def test_ordinary_root_keeps_member_and_adds_pressure_evidence(self) -> None:
        board = replay(SELFPLAY_MOVE_23_PREFIX)
        target = parse_move("D7", board.size)
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
        ai._begin_move_search()

        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )

        self.assertEqual(
            root_candidates.RootCandidateMode.ORDINARY,
            plan.mode,
        )
        self.assertEqual(1, plan.moves.count(target))
        self.assertEqual((target,), ai._root_pressure_prevention)
        self.assertEqual(
            {
                root_candidates.CandidateSource.ORDINARY,
                root_candidates.CandidateSource.PRESSURE_PREVENTION,
            },
            set(ai._root_candidate_sources[target]),
        )
        self.assertLessEqual(len(plan.moves), ai.config.root_candidate_limit)


class TestV0168EmergencyVCFGate(unittest.TestCase):
    def test_single_candidate_gate_proves_recorded_f9_loss(self) -> None:
        board = replay(SELFPLAY_MOVE_29_PREFIX)
        target = parse_move("F9", board.size)
        history = tuple(board.move_history)
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=60.0)
        ai._begin_move_search()
        scanner = root_safety.RootVCFSafetyScanner(
            find_vcf=lambda position, attacker, deadline: (
                ai._find_vcf_with_deadline(
                    position,
                    attacker,
                    deadline=deadline,
                    root_safety_channel=True,
                )
            ),
            node_count=lambda: ai._counters.root_vcf_nodes,
        )

        analysis = scanner.scan_candidate(
            board,
            target,
            mover=BLACK,
            opponent=WHITE,
            budget_seconds=1.0,
            hard_deadline=None,
        )

        self.assertEqual(
            root_safety.RootCandidateSafety.PROVEN_LOSS.value,
            analysis.status,
        )
        self.assertTrue(analysis.completed)
        self.assertEqual(parse_move("D8", board.size), analysis.principal_variation[0])
        self.assertEqual(history, tuple(board.move_history))
        self.assertTrue(board.is_empty(*target))

    def test_final_audit_rejects_vcf_losing_late_intercept(self) -> None:
        board = replay(SELFPLAY_MOVE_29_PREFIX)
        leader = parse_move("C8", board.size)
        emergency = parse_move("F9", board.size)
        result = RootResult(
            move=leader,
            score=-10_000,
            principal_variation=(leader,),
            ranked_moves=((leader, -10_000),),
        )
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=60.0)
        ai.config = replace(
            ai.config,
            proof_final_candidate_limit=1,
            proof_emergency_vcf_time_fraction=0.5,
            proof_emergency_vcf_max_seconds=1.0,
        )
        ai._begin_move_search()

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
                    best_move=emergency,
                    principal_variation=(emergency,),
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
            patch.object(ai, "_final_proof_budget_seconds", return_value=2.0),
        ):
            revised = ai._run_final_proof_audit(board, result)

        self.assertNotEqual(emergency, revised.move)
        self.assertIn(emergency, ai._final_proof_rejected)
        self.assertEqual("proved_loss_fallback", ai._final_proof_selection_basis)
        late = {
            item.move: item
            for item in ai._root_vcf_scan.analyses  # type: ignore[union-attr]
        }[emergency]
        self.assertEqual(
            root_safety.RootCandidateSafety.PROVEN_LOSS.value,
            late.status,
        )


class TestV0168BoundaryDualChannel(unittest.TestCase):
    def setUp(self) -> None:
        self.board = Board()
        self.first = (7, 7)
        self.second = (7, 8)
        self.result = RootResult(
            move=self.first,
            score=0,
            principal_variation=(self.first,),
            ranked_moves=((self.first, 0), (self.second, 0)),
        )
        self.initial = review_probe(
            self.first,
            self.second,
            depth=2,
            first_score=HEURISTIC_SCORE_LIMIT,
            second_score=HEURISTIC_SCORE_LIMIT,
            leaders=(self.first,),
            requested=1.0,
        )

    def test_same_clamp_uses_reserved_secondary_channel(self) -> None:
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=60.0)
        ai._begin_move_search()
        tactical = replace(self.initial, completed_depth=3)
        secondary = review_probe(
            self.second,
            self.first,
            depth=4,
            first_score=900,
            second_score=100,
            leaders=(self.second, self.second, self.second),
        )

        with (
            patch.object(
                ai,
                "_boundary_tie_escalation_budget_seconds",
                return_value=4.0,
            ),
            patch.object(
                ai,
                "_run_dynamic_pair_review",
                side_effect=(tactical, secondary),
            ) as run_pair,
        ):
            actual = ai._escalate_boundary_tie_review(
                self.board,
                self.result,
                self.second,
                initial_probe=self.initial,
                completed_depth=6,
                fallback_move=self.first,
            )

        self.assertEqual(2, run_pair.call_count)
        secondary_call = run_pair.call_args_list[1]
        self.assertFalse(
            secondary_call.kwargs["quiet_frontier_extension_override"]
        )
        self.assertTrue(secondary_call.kwargs["reject_mate_like"])
        self.assertGreaterEqual(
            secondary_call.kwargs["budget_seconds"],
            ai.config.root_boundary_secondary_min_seconds,
        )
        self.assertEqual(self.second, actual.approved_move)
        self.assertEqual(
            "boundary_secondary_equal_window",
            actual.selection_basis,
        )
        self.assertEqual(
            "boundary_secondary_equal_window",
            ai._root_review_trace[-1][1].selection_basis,
        )
        self.assertEqual(5.0, actual.requested_budget_seconds)

    def test_distinguishable_tactical_result_does_not_run_secondary(self) -> None:
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=60.0)
        ai._begin_move_search()
        distinguishable = review_probe(
            self.first,
            self.second,
            depth=3,
            first_score=500,
            second_score=100,
            leaders=(self.first,),
        )
        with (
            patch.object(
                ai,
                "_boundary_tie_escalation_budget_seconds",
                return_value=4.0,
            ),
            patch.object(
                ai,
                "_run_dynamic_pair_review",
                return_value=distinguishable,
            ) as run_pair,
        ):
            actual = ai._escalate_boundary_tie_review(
                self.board,
                self.result,
                self.second,
                initial_probe=self.initial,
                completed_depth=6,
                fallback_move=self.first,
            )

        self.assertEqual(1, run_pair.call_count)
        self.assertEqual("boundary_tie_pvs_fallback", actual.selection_basis)

    def test_secondary_channel_requires_its_minimum_budget(self) -> None:
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=60.0)
        ai._begin_move_search()
        with (
            patch.object(
                ai,
                "_boundary_tie_escalation_budget_seconds",
                return_value=1.5,
            ),
            patch.object(
                ai,
                "_run_dynamic_pair_review",
                return_value=self.initial,
            ) as run_pair,
        ):
            actual = ai._escalate_boundary_tie_review(
                self.board,
                self.result,
                self.second,
                initial_probe=self.initial,
                completed_depth=6,
                fallback_move=self.first,
            )

        self.assertEqual(1, run_pair.call_count)
        self.assertEqual("boundary_tie_pvs_fallback", actual.selection_basis)

    def test_secondary_leader_cannot_override_strict_proof_conflict(self) -> None:
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=60.0)
        ai._begin_move_search()
        secondary = review_probe(
            self.second,
            self.first,
            depth=4,
            first_score=900,
            second_score=100,
            leaders=(self.second, self.second, self.second),
        )
        ai._proof_candidates = (
            ProofCandidateAnalysis(
                move=self.second,
                state=ProofState.PROVEN_WIN.value,
                completed=True,
                nodes=1,
                elapsed_seconds=0.0,
            ),
        )

        self.assertIsNone(ai._boundary_secondary_approved_move(secondary))


class TestV0168ReviewAudit(unittest.TestCase):
    def test_triggered_pair_audit_is_bounded_and_source_aware(self) -> None:
        first = (7, 7)
        second = (7, 8)
        probe = review_probe(
            first,
            second,
            depth=4,
            first_score=100,
            second_score=50,
            leaders=(first, first, first),
        )
        ai = SearchAI(BLACK, time_limit_seconds=None, diagnostics=True)
        ai._begin_move_search()
        ai._root_review_finalists = (first, second)
        ai._root_review_trace = [("primary", probe)]
        ai._root_candidate_sources = {
            first: frozenset({root_candidates.CandidateSource.ORDINARY}),
            second: frozenset(
                {
                    root_candidates.CandidateSource.ORDINARY,
                    root_candidates.CandidateSource.PRESSURE_PREVENTION,
                }
            ),
        }

        analysis = build_search_analysis(
            ai,
            selected_move=first,
            reason="test",
            candidate_count=2,
            ranked_moves=[(first, 100), (second, 50)],
            completed_depth=4,
            principal_variation=(first,),
            search_completed=True,
        )
        payload = analysis.to_dict()

        self.assertEqual(2, len(payload["root_review_finalists"]))
        self.assertEqual(1, len(payload["root_review_pairs"]))
        self.assertEqual("primary", payload["root_review_pairs"][0]["channel"])
        coverage = {
            item["source"]: {
                move["coordinate"] for move in item["moves"]
            }
            for item in payload["root_review_source_coverage"]
        }
        self.assertIn("pressure_prevention", coverage)
        self.assertEqual({"I8"}, coverage["pressure_prevention"])

    def test_disabled_diagnostics_omit_full_pair_audit(self) -> None:
        first = (7, 7)
        second = (7, 8)
        ai = SearchAI(BLACK, time_limit_seconds=None, diagnostics=False)
        ai._begin_move_search()
        ai._root_review_finalists = (first, second)
        ai._root_review_trace = [(
            "primary",
            review_probe(
                first,
                second,
                depth=3,
                first_score=10,
                second_score=0,
                leaders=(first, first),
            ),
        )]

        analysis = build_search_analysis(
            ai,
            selected_move=first,
            reason="test",
            candidate_count=2,
            ranked_moves=[(first, 10), (second, 0)],
            completed_depth=3,
            principal_variation=(first,),
            search_completed=True,
        )

        self.assertEqual((), analysis.root_review_pairs)
        self.assertEqual((), analysis.root_review_finalists)

    def test_new_budget_and_breadth_config_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            replace(
                SearchConfig(),
                root_boundary_secondary_time_fraction=1.0,
            )
        with self.assertRaises(ValueError):
            replace(
                SearchConfig(),
                root_broad_quiet_attack_min_continuations=1,
            )


if __name__ == "__main__":
    unittest.main()
