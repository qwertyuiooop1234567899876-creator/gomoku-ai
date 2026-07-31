import unittest

from engine.ai import ProofCandidateAnalysis
from engine.board import BLACK, WHITE, Board
from engine.game import format_move, parse_move
from engine.proof_search import ProofState
from engine.search import MATE_SCORE, RootResult, SearchAI


def build_black_forty_one_position() -> Board:
    board = Board()
    coordinates = (
        "H8", "G7", "I7", "G9", "G8", "F8", "E7", "H10",
        "H6", "J8", "I9", "I6", "J12", "F9", "F7", "H9",
        "H5", "G6", "G5", "H4", "E9", "I5", "E10", "E8",
        "G10", "J6", "K7", "J7", "J9", "E3", "F5", "K6",
        "F6", "D5", "J4", "I8", "L5", "G3", "F2", "C9",
    )
    for index, coordinate in enumerate(coordinates):
        board.place(
            *parse_move(coordinate, board.size),
            BLACK if index % 2 == 0 else WHITE,
        )
    return board


class TestV0122MateScoreQuarantine(unittest.TestCase):
    def test_black_forty_one_unproven_mates_leave_terminal_band(
        self,
    ) -> None:
        board = build_black_forty_one_position()
        before = (
            tuple(tuple(row) for row in board.grid),
            tuple(board.move_history),
            board.zobrist_hash,
        )
        ai = SearchAI(player=BLACK, time_limit_seconds=None)
        coordinates = ("M6", "K8", "F10", "D8", "F4", "F3")
        moves = tuple(
            parse_move(coordinate, board.size)
            for coordinate in coordinates
        )
        ai._proof_candidates = tuple(
            ProofCandidateAnalysis(
                move=move,
                state=ProofState.UNKNOWN.value,
                completed=False,
                nodes=1,
                elapsed_seconds=0.01,
            )
            for move in moves
        )
        result = RootResult(
            move=moves[0],
            score=MATE_SCORE - 6,
            principal_variation=(moves[0],),
            ranked_moves=tuple(
                (move, MATE_SCORE - 6 - index)
                for index, move in enumerate(moves)
            ),
            ranked_variations=tuple(
                (
                    move,
                    MATE_SCORE - 6 - index,
                    (move,),
                )
                for index, move in enumerate(moves)
            ),
        )

        revised = ai._quarantine_unproven_root_scores(board, result)

        self.assertEqual("F3", format_move(*revised.move))
        self.assertNotEqual("M6", format_move(*revised.move))
        self.assertTrue(ai._root_mate_scores_quarantined)
        self.assertTrue(
            all(
                abs(score) < MATE_SCORE - 10_000
                for _, score in revised.ranked_moves
            )
        )
        self.assertEqual(
            before,
            (
                tuple(tuple(row) for row in board.grid),
                tuple(board.move_history),
                board.zobrist_hash,
            ),
        )

    def test_proven_opponent_win_keeps_negative_mate_score(
        self,
    ) -> None:
        board = Board()
        board.place(7, 7, BLACK)
        move = (7, 8)
        ai = SearchAI(player=WHITE, time_limit_seconds=None)
        ai._proof_candidates = (
            ProofCandidateAnalysis(
                move=move,
                state=ProofState.PROVEN_WIN.value,
                completed=True,
                nodes=10,
                elapsed_seconds=0.01,
            ),
        )
        result = RootResult(
            move=move,
            score=-MATE_SCORE + 4,
            principal_variation=(move,),
            ranked_moves=((move, -MATE_SCORE + 4),),
            ranked_variations=(
                (move, -MATE_SCORE + 4, (move,)),
            ),
        )

        revised = ai._quarantine_unproven_root_scores(board, result)

        self.assertIs(result, revised)
        self.assertFalse(ai._root_mate_scores_quarantined)

    def test_mixed_root_is_recalibrated_on_one_scale(self) -> None:
        board = build_black_forty_one_position()
        ai = SearchAI(player=BLACK, time_limit_seconds=None)
        m6 = parse_move("M6", board.size)
        f3 = parse_move("F3", board.size)
        ai._proof_candidates = (
            ProofCandidateAnalysis(
                move=m6,
                state=ProofState.UNKNOWN.value,
                completed=False,
                nodes=1,
                elapsed_seconds=0.01,
            ),
            ProofCandidateAnalysis(
                move=f3,
                state=ProofState.UNKNOWN.value,
                completed=False,
                nodes=1,
                elapsed_seconds=0.01,
            ),
        )
        result = RootResult(
            move=m6,
            score=MATE_SCORE - 6,
            principal_variation=(m6,),
            ranked_moves=(
                (m6, MATE_SCORE - 6),
                (f3, -12_100),
            ),
            ranked_variations=(
                (m6, MATE_SCORE - 6, (m6,)),
                (f3, -12_100, (f3,)),
            ),
        )

        revised = ai._quarantine_unproven_root_scores(board, result)

        self.assertEqual("F3", format_move(*revised.move))
        self.assertGreater(
            dict(revised.ranked_moves)[f3],
            dict(revised.ranked_moves)[m6],
        )

    def test_safety_proof_does_not_turn_pvs_into_proven_win(
        self,
    ) -> None:
        board = Board()
        board.place(7, 7, BLACK)
        move = (7, 8)
        ai = SearchAI(player=WHITE, time_limit_seconds=None)
        ai._proof_candidates = (
            ProofCandidateAnalysis(
                move=move,
                state=ProofState.PROVEN_LOSS.value,
                completed=True,
                nodes=10,
                elapsed_seconds=0.01,
            ),
        )
        result = RootResult(
            move=move,
            score=MATE_SCORE - 4,
            principal_variation=(move,),
            ranked_moves=((move, MATE_SCORE - 4),),
            ranked_variations=((move, MATE_SCORE - 4, (move,)),),
        )

        revised = ai._quarantine_unproven_root_scores(board, result)

        self.assertLess(abs(revised.score), MATE_SCORE - 10_000)
        self.assertTrue(ai._root_mate_scores_quarantined)

    def test_quarantined_root_skips_selective_safety_recheck(
        self,
    ) -> None:
        ai = SearchAI(player=BLACK)
        first = (7, 7)
        second = (7, 8)
        result = RootResult(
            move=first,
            score=10_000,
            principal_variation=(first,),
            ranked_moves=((first, 10_000), (second, 9_500)),
        )
        ai._root_mate_scores_quarantined = True

        self.assertIsNone(
            ai._root_safety_trigger(result, [result, result])
        )


if __name__ == "__main__":
    unittest.main()
