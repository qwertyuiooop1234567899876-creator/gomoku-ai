from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from engine import root_candidates, root_review
from engine.ai import RootSafetyCandidateAnalysis
from engine.board import BLACK, WHITE, Board
from engine.game import parse_move
from engine.proof_search import ProofResult, ProofState
from engine.search import SearchAI
from engine.search_types import RootResult, RootSafetyProbeResult, SearchConfig


YIXIN_MOVE_21_PREFIX = (
    "H8 H9 G7 I9 G9 I7 G8 G6 I8 F8 "
    "I6 J8 H6 F10 H10 G11 H5 H7 K8 J7"
).split()

SELFPLAY_MOVE_19_PREFIX = (
    "H8 G7 I7 G9 G8 H9 I8 J8 F9 I6 E8 F8 E7 E10 H6 G10 F10 J9"
).split()


def build_board(coordinates: list[str]) -> Board:
    board = Board()
    for index, coordinate in enumerate(coordinates):
        board.place(
            *parse_move(coordinate, board.size),
            BLACK if index % 2 == 0 else WHITE,
        )
    return board


class TestRecordDrivenCandidateCompleteness(unittest.TestCase):
    def test_active_open_three_outside_quick_profile_prefix_is_sourced(
        self,
    ) -> None:
        board = build_board(SELFPLAY_MOVE_19_PREFIX)
        target = parse_move("E6", board.size)
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
        ai._begin_move_search()

        profile_prefix = ai._root_profile_pool(
            board,
            board.get_legal_moves(),
        )
        self.assertNotIn(target, profile_prefix)

        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )

        self.assertIn(target, plan.moves)
        self.assertGreaterEqual(
            plan.own_profiles[target].open_three_directions,
            1,
        )
        self.assertIn(
            root_candidates.CandidateSource.ACTIVE_COUNTERATTACK,
            ai._root_candidate_sources[target],
        )
        self.assertLessEqual(len(plan.moves), ai.config.root_candidate_limit)

    def test_open_three_pressure_point_survives_multi_frontier_mode(self) -> None:
        board = build_board(YIXIN_MOVE_21_PREFIX)
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
        ai._begin_move_search()

        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )

        defensive = parse_move("F11", board.size)
        self.assertIn(defensive, plan.moves)
        self.assertIn(
            root_candidates.CandidateSource.PRESSURE_PREVENTION,
            ai._root_candidate_sources[defensive],
        )
        for existing_source in ("K7", "L7"):
            self.assertNotIn(
                root_candidates.CandidateSource.PRESSURE_PREVENTION,
                ai._root_candidate_sources[parse_move(existing_source)],
            )
        self.assertLessEqual(len(plan.moves), ai.config.root_candidate_limit)

    def test_audited_pressure_unknown_is_conservative_fallback(self) -> None:
        leader = parse_move("J9")
        pressure = parse_move("F11")

        self.assertEqual(
            pressure,
            root_review.preferred_unknown_move(
                (leader, pressure),
                (pressure,),
            ),
        )
        self.assertEqual(
            leader,
            root_review.preferred_unknown_move((leader,), (pressure,)),
        )

    def test_final_audit_keeps_pressure_fallback_unknown(self) -> None:
        board = Board()
        board.place(*parse_move("H8"), BLACK)
        leader = parse_move("J9")
        pressure = parse_move("F11")
        ai = SearchAI(BLACK, time_limit_seconds=60.0)
        ai.config = replace(ai.config, proof_final_candidate_limit=2)
        ai._begin_move_search()
        ai._root_pressure_prevention = (pressure,)
        ai._root_candidate_sources = {
            leader: frozenset(
                {root_candidates.CandidateSource.THREAT_FRONTIER}
            ),
            pressure: frozenset(
                {root_candidates.CandidateSource.PRESSURE_PREVENTION}
            ),
        }
        result = RootResult(
            leader,
            1_000,
            (leader,),
            ((leader, 1_000), (pressure, 900)),
        )

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

        with patch("engine.search.ProofSearch", UnknownProofSearch):
            revised = ai._run_final_proof_audit(board, result)

        self.assertEqual(pressure, revised.move)
        self.assertEqual(ProofState.UNKNOWN.value, ai._final_proof_state)
        self.assertFalse(ai._final_proof_completed)

    def test_critical_frontier_challenger_precedes_score_only_challenger(
        self,
    ) -> None:
        leader = parse_move("I6")
        strongest = parse_move("F6")
        critical = parse_move("I8")
        quiet = parse_move("G5")
        result = RootResult(
            leader,
            99_000,
            (leader,),
            (
                (leader, 99_000),
                (strongest, 98_200),
                (quiet, 10_000),
                (critical, -2_000),
            ),
        )
        pool = [move for move, _score in result.ranked_moves]

        finalists = root_review.finalists(
            SearchConfig(root_dynamic_review_finalist_limit=4),
            result,
            pool,
            {move: 0 for move in pool},
            critical_groups=((critical,),),
        )

        self.assertEqual([leader, critical, strongest], finalists[:3])
        self.assertLessEqual(len(finalists), 4)

    def test_unreviewed_root_expansion_cannot_replace_base_leader(self) -> None:
        board = Board()
        board.place(*parse_move("H8"), WHITE)
        base = parse_move("H7")
        sibling = parse_move("I7")
        expansion = parse_move("M8")
        ai = SearchAI(BLACK, max_depth=1, time_limit_seconds=None)
        ai.config = replace(
            ai.config,
            root_survival_min_depth=1,
            root_unverified_advantage_threshold=900_000,
            use_aspiration=False,
        )
        ai._begin_move_search()
        ordinary = root_candidates.CandidateSource.ORDINARY
        ai._root_candidate_sources = {
            base: frozenset({ordinary}),
            sibling: frozenset({ordinary}),
        }
        initial = RootResult(
            base,
            950_000,
            (base,),
            ((base, 950_000), (sibling, 100)),
        )
        expanded = RootResult(
            expansion,
            1_200_000,
            (expansion,),
            (
                (expansion, 1_200_000),
                (base, 940_000),
                (sibling, 100),
            ),
        )

        with (
            patch.object(ai, "_search_root", side_effect=(initial, expanded)),
            patch.object(
                ai,
                "_expand_unverified_advantage_root_candidates",
                return_value=[base, sibling, expansion],
            ),
        ):
            outcome = ai._run_iterative_root_search(
                board,
                [base, sibling],
                fallback_move=base,
                preserve_frontier_order=False,
                allow_near_loss_expansion=True,
                defense_probe=None,
            )

        self.assertEqual("unverified_advantage", outcome.root_expansion_reason)
        self.assertEqual(base, outcome.result.move)
        self.assertTrue(ai._root_expansion_hold_applied)
        self.assertIn(expansion, outcome.candidates)
        self.assertIn(
            root_candidates.CandidateSource.ROOT_EXPANSION,
            ai._root_candidate_sources[expansion],
        )

    def test_expansion_equal_window_review_must_include_base_move(self) -> None:
        base = parse_move("H7")
        expansion = parse_move("M8")
        other_expansion = parse_move("B11")
        ai = SearchAI(BLACK, time_limit_seconds=None)
        ai._root_candidate_sources = {
            base: frozenset({root_candidates.CandidateSource.ORDINARY}),
            expansion: frozenset(
                {root_candidates.CandidateSource.ROOT_EXPANSION}
            ),
            other_expansion: frozenset(
                {root_candidates.CandidateSource.ROOT_EXPANSION}
            ),
        }

        def probe_with(challenger: tuple[int, int]) -> RootSafetyProbeResult:
            return RootSafetyProbeResult(
                trigger="dynamic_remaining_review",
                pvs_gap=0,
                main_rank_stable=True,
                completed_depth=5,
                nodes=1,
                candidates=(
                    RootSafetyCandidateAnalysis(expansion, 1),
                    RootSafetyCandidateAnalysis(challenger, 0),
                ),
                leader_history=(expansion, expansion),
                approved_move=expansion,
            )

        ai._root_safety_probe = probe_with(other_expansion)
        self.assertFalse(ai._expansion_leader_has_base_review(expansion))
        ai._root_safety_probe = probe_with(base)
        self.assertTrue(ai._expansion_leader_has_base_review(expansion))


if __name__ == "__main__":
    unittest.main()
