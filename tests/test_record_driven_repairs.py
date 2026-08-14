from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from engine import root_candidates, root_review
from engine.board import BLACK, WHITE, Board
from engine.game import parse_move
from engine.proof_search import ProofResult, ProofState
from engine.search import SearchAI
from engine.search_types import RootResult, SearchConfig


YIXIN_MOVE_21_PREFIX = (
    "H8 H9 G7 I9 G9 I7 G8 G6 I8 F8 "
    "I6 J8 H6 F10 H10 G11 H5 H7 K8 J7"
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


if __name__ == "__main__":
    unittest.main()
