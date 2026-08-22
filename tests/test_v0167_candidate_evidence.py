from __future__ import annotations

import unittest
from unittest.mock import patch

from engine import root_candidates, root_review
from engine.board import BLACK, WHITE, Board
from engine.game import parse_move
from engine.proof_search import ProofResult, ProofState
from engine.search import SearchAI
from engine.search_types import RootResult
from engine.threats import ThreatFrontier, ThreatKind


MOVE_23_PREFIX = (
    "H8 H9 G7 I9 G9 I7 G8 G6 I8 F8 F10 E11 G11 G10 "
    "H12 E9 J8 K8 I13 J14 E10 I6"
).split()
MOVE_25_PREFIX = [*MOVE_23_PREFIX, "F9", "D11"]
MOVE_27_PREFIX = [*MOVE_25_PREFIX, "J7", "D10"]


def build_board(coordinates: list[str]) -> Board:
    board = Board()
    for index, coordinate in enumerate(coordinates):
        board.place(
            *parse_move(coordinate, board.size),
            BLACK if index % 2 == 0 else WHITE,
        )
    return board


class TestV0167PressureEvidence(unittest.TestCase):
    def test_covered_pressure_move_keeps_membership_and_evidence(self) -> None:
        board = build_board(MOVE_25_PREFIX)
        target = parse_move("H5", board.size)
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
        ai._begin_move_search()

        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )

        self.assertEqual(1, plan.moves.count(target))
        self.assertIn(
            root_candidates.CandidateSource.ORDINARY,
            ai._root_candidate_sources[target],
        )
        self.assertIn(
            root_candidates.CandidateSource.PRESSURE_PREVENTION,
            ai._root_candidate_sources[target],
        )
        self.assertEqual((target,), ai._root_pressure_prevention)
        self.assertLessEqual(len(plan.moves), ai.config.root_candidate_limit)

    def test_pressure_evidence_reaches_review_and_final_audit(self) -> None:
        board = build_board(MOVE_25_PREFIX)
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
        ai._begin_move_search()
        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )
        moves = {
            name: parse_move(name, board.size)
            for name in ("J7", "F6", "L9", "J9", "H5", "H6")
        }
        result = RootResult(
            move=moves["J7"],
            score=-102_000,
            principal_variation=(moves["J7"],),
            ranked_moves=(
                (moves["J7"], -102_000),
                (moves["F6"], -102_000),
                (moves["L9"], -102_000),
                (moves["J9"], -102_000),
                (moves["H5"], -102_200),
                (moves["H6"], -103_000),
            ),
        )
        available = tuple(move for move, _score in result.ranked_moves)
        critical_groups = ai._critical_root_review_groups(available)
        critical_moves = root_candidates.merge_unique(*critical_groups)
        pool = root_review.review_pool(
            ai.config,
            result,
            (),
            critical_moves=critical_moves,
            active_moves=(),
            quiet_moves=(),
            offensive_moves=(),
        )
        finalists = root_review.finalists(
            ai.config,
            result,
            pool,
            {move: 0 for move in pool},
            critical_groups=critical_groups,
        )

        self.assertIn(moves["H5"], critical_groups[0])
        self.assertIn(moves["H5"], pool)
        self.assertIn(moves["H5"], finalists)
        self.assertLess(
            finalists.index(moves["H5"]),
            ai.config.root_dynamic_review_finalist_limit,
        )

        checked: list[tuple[int, int]] = []

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
                checked.append(move)
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

        with (
            patch("engine.search.ProofSearch", UnknownProofSearch),
            patch.object(ai, "_final_proof_budget_seconds", return_value=8.0),
        ):
            ai._run_final_proof_audit(board, result)

        self.assertEqual(moves["J7"], checked[0])
        self.assertEqual(moves["H5"], checked[1])
        self.assertLess(
            checked.index(moves["H5"]),
            ai.config.proof_final_candidate_limit,
        )


class TestV0167MandatoryCertificateIntercept(unittest.TestCase):
    def test_linked_quiet_intercept_enters_singleton_mandatory_root(self) -> None:
        board = build_board(MOVE_27_PREFIX)
        target = parse_move("F11", board.size)
        ai = SearchAI(BLACK, max_depth=8, time_limit_seconds=None)
        ai._begin_move_search()

        plan = ai._prepare_root_candidate_plan(
            board,
            board.get_legal_moves(),
        )

        self.assertEqual(
            root_candidates.RootCandidateMode.MANDATORY_DEFENSE,
            plan.mode,
        )
        self.assertIn(target, plan.moves)
        self.assertIn(
            root_candidates.CandidateSource.CERTIFICATE_INTERCEPT,
            ai._root_candidate_sources[target],
        )
        self.assertEqual(4, len(plan.moves))
        self.assertIsNone(plan.defense_probe)

        with patch.object(
            ai,
            "_run_defense_vct_probe",
            return_value=None,
        ) as probe:
            unchanged, _result = ai._maybe_run_post_filter_defense_probe(
                board,
                plan.moves,
                candidate_mode=plan.mode,
                existing_probe=plan.defense_probe,
            )
            self.assertEqual(plan.moves, unchanged)
            probe.assert_not_called()

            ai._maybe_run_post_filter_defense_probe(
                board,
                plan.moves[:3],
                candidate_mode=plan.mode,
                existing_probe=None,
            )
            probe.assert_called_once()

    def test_unlinked_quiet_frontier_is_not_a_certificate_intercept(self) -> None:
        forcing = (7, 7)
        linked = ThreatFrontier(
            gain_move=(6, 6),
            kind=ThreatKind.QUIET,
            continuations=(forcing,),
            continuation_kinds=(ThreatKind.DOUBLE_FOUR,),
            continuation_ranks=(95,),
            coverage_complete=False,
        )
        unrelated = ThreatFrontier(
            gain_move=(5, 5),
            kind=ThreatKind.QUIET,
            continuations=((4, 4),),
            continuation_kinds=(ThreatKind.DOUBLE_FOUR,),
            continuation_ranks=(95,),
            coverage_complete=False,
        )

        actual = root_candidates.forcing_certificate_intercept_moves(
            frontiers=(unrelated, linked),
            forcing_moves=(forcing,),
            strong_rank=80,
            limit=1,
        )

        self.assertEqual([linked.gain_move], actual)
        self.assertNotIn(unrelated.gain_move, actual)


if __name__ == "__main__":
    unittest.main()
