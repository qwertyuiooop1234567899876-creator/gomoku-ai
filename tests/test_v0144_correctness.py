from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from engine import root_policy
from engine.ai import ProofCandidateAnalysis
from engine.board import BLACK, WHITE, Board
from engine.game import format_move, parse_move
from engine.proof_search import ProofResult, ProofState
from engine.search import SearchAI
from engine.search_types import MATE_SCORE, RootResult


BLACK_SIXTEEN_PREFIX = """
H8 H9 G9 I7 F10 G8 F7 F8 E11 D12 E10 D10 E9 E8 D8 G11
""".split()

BLACK_TWENTY_FOUR_PREFIX = """
H8 H9 G9 I7 F10 G8 F7 F8 E11 D12 E10 D10 E9 E8 D8 G11
C7 B6 E12 E13 D7 E7 D9 C11
""".split()


def rotate_coordinate_180(coordinate: str, size: int = 15) -> str:
    row, column = parse_move(coordinate, size)
    return format_move(size - 1 - row, size - 1 - column)


def build_board(coordinates: list[str]) -> Board:
    board = Board()
    for index, coordinate in enumerate(coordinates):
        board.place(
            *parse_move(coordinate, board.size),
            BLACK if index % 2 == 0 else WHITE,
        )
    return board


def board_state(board: Board) -> tuple[object, ...]:
    return (
        tuple(tuple(row) for row in board.grid),
        tuple(board.move_history),
        board.zobrist_hash,
        board.empty_count,
    )


class TestV0144RootCompleteness(unittest.TestCase):
    def test_unverified_advantage_expands_to_quiet_survival_move(self) -> None:
        for coordinates, required in (
            (BLACK_SIXTEEN_PREFIX, "C9"),
            (
                [rotate_coordinate_180(item) for item in BLACK_SIXTEEN_PREFIX],
                rotate_coordinate_180("C9"),
            ),
        ):
            with self.subTest(required=required):
                board = build_board(coordinates)
                before = board_state(board)
                ai = SearchAI(
                    BLACK,
                    max_depth=1,
                    time_limit_seconds=None,
                )
                ai.config = replace(
                    ai.config,
                    root_survival_min_depth=1,
                )
                ai._begin_move_search()
                plan = ai._prepare_root_candidate_plan(
                    board,
                    board.get_legal_moves(),
                )

                outcome = ai._run_iterative_root_search(
                    board,
                    plan.moves,
                    fallback_move=plan.moves[0],
                    preserve_frontier_order=plan.preserve_frontier_order,
                    allow_near_loss_expansion=(
                        plan.allow_near_loss_expansion
                    ),
                    defense_probe=plan.defense_probe,
                )
                expanded_coordinates = {
                    format_move(*move) for move in outcome.candidates
                }

                self.assertIn(required, expanded_coordinates)
                self.assertEqual(
                    "unverified_advantage",
                    outcome.root_expansion_reason,
                )
                self.assertLessEqual(
                    len(outcome.candidates),
                    ai.config.root_candidate_limit * 2,
                )
                self.assertEqual(before, board_state(board))

    def test_full_relevant_scan_keeps_both_remote_open_four_blocks(
        self,
    ) -> None:
        for coordinates, required in (
            (BLACK_TWENTY_FOUR_PREFIX, {"B10", "F14"}),
            (
                [
                    rotate_coordinate_180(item)
                    for item in BLACK_TWENTY_FOUR_PREFIX
                ],
                {
                    rotate_coordinate_180("B10"),
                    rotate_coordinate_180("F14"),
                },
            ),
        ):
            with self.subTest(required=required):
                board = build_board(coordinates)
                before = board_state(board)
                ai = SearchAI(
                    BLACK,
                    max_depth=8,
                    time_limit_seconds=None,
                )
                ai._begin_move_search()

                plan = ai._prepare_root_candidate_plan(
                    board,
                    board.get_legal_moves(),
                )
                candidate_coordinates = {
                    format_move(*move) for move in plan.moves
                }

                self.assertTrue(required.issubset(candidate_coordinates))
                self.assertEqual(before, board_state(board))


class TestV0144ProofSemantics(unittest.TestCase):
    def test_frontier_quarantine_uses_real_heuristic_scale(self) -> None:
        first = (1, 1)
        second = (2, 2)
        result = RootResult(
            move=first,
            score=MATE_SCORE - 4,
            principal_variation=(first,),
            ranked_moves=(
                (first, MATE_SCORE - 4),
                (second, 5_000),
            ),
            ranked_variations=(
                (first, MATE_SCORE - 4, (first,)),
                (second, 5_000, (second,)),
            ),
        )

        revised, quarantined = root_policy.quarantine_unproven_scores(
            result,
            proof_states={
                first: ProofState.UNKNOWN.value,
                second: ProofState.UNKNOWN.value,
            },
            heuristic_score=lambda move: {
                first: 100,
                second: 5_000,
            }[move],
            preserve_order=True,
        )

        self.assertTrue(quarantined)
        self.assertEqual(second, revised.move)
        self.assertEqual(((second, 5_000), (first, 100)), revised.ranked_moves)

    def test_final_audit_continues_after_unknown_to_find_safe_move(
        self,
    ) -> None:
        board = Board()
        board.place(7, 7, BLACK)
        unknown = (7, 8)
        safe = (8, 8)
        before = board_state(board)
        ai = SearchAI(BLACK, time_limit_seconds=60.0)
        ai.config = replace(
            ai.config,
            proof_final_candidate_limit=2,
        )
        ai._begin_move_search()
        root = RootResult(
            move=unknown,
            score=10_000,
            principal_variation=(unknown,),
            ranked_moves=((unknown, 10_000), (safe, 9_000)),
            ranked_variations=(
                (unknown, 10_000, (unknown,)),
                (safe, 9_000, (safe,)),
            ),
        )

        class FakeProofSearch:
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
                    ProofState.UNKNOWN
                    if move == unknown
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
                    completed=state is not ProofState.UNKNOWN,
                    cutoff_reason=(
                        "deadline"
                        if state is ProofState.UNKNOWN
                        else None
                    ),
                    elapsed_seconds=0.0,
                )

        with patch("engine.search.ProofSearch", FakeProofSearch):
            revised = ai._run_final_proof_audit(board, root)

        self.assertEqual(safe, revised.move)
        self.assertEqual(ProofState.PROVEN_LOSS.value, ai._final_proof_state)
        self.assertTrue(ai._final_proof_completed)
        self.assertEqual(before, board_state(board))

    def test_final_unknown_is_reported_as_unconfirmed(self) -> None:
        board = Board()
        board.place(7, 7, BLACK)
        move = (7, 8)
        ai = SearchAI(BLACK, time_limit_seconds=60.0)
        ai.config = replace(
            ai.config,
            proof_final_candidate_limit=1,
        )
        ai._begin_move_search()
        root = RootResult(
            move=move,
            score=10_000,
            principal_variation=(move,),
            ranked_moves=((move, 10_000),),
            ranked_variations=((move, 10_000, (move,)),),
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
            revised = ai._run_final_proof_audit(board, root)

        self.assertEqual(move, revised.move)
        self.assertTrue(ai._final_proof_checked)
        self.assertEqual(ProofState.UNKNOWN.value, ai._final_proof_state)
        self.assertFalse(ai._final_proof_completed)


if __name__ == "__main__":
    unittest.main()
