import unittest
from dataclasses import replace
from unittest.mock import patch

from engine.ai import ProofCandidateAnalysis
from engine.board import BLACK, WHITE, Board
from engine.game import format_move, parse_move
from engine.proof_search import ProofState
from engine.search import MATE_SCORE, RootResult, SearchAI


def build_black_thirteen_position() -> Board:
    board = Board()
    coordinates = (
        "H8",
        "H9",
        "G7",
        "I9",
        "G9",
        "I7",
        "I8",
        "G8",
        "F7",
        "J8",
        "H10",
        "K9",
    )
    for index, coordinate in enumerate(coordinates):
        board.place(
            *parse_move(coordinate, board.size),
            BLACK if index % 2 == 0 else WHITE,
        )
    return board


class TestV0121RootCandidateSafety(unittest.TestCase):
    def test_attack_candidate_cannot_hide_mandatory_defenses(
        self,
    ) -> None:
        board = build_black_thirteen_position()
        ai = SearchAI(
            player=BLACK,
            max_depth=2,
            time_limit_seconds=None,
            diagnostics=True,
        )
        ai.config = replace(
            ai.config,
            vcf_max_attacker_moves=0,
            root_safety_enabled=False,
        )

        selected = ai.choose_move(board)

        self.assertEqual("L10", format_move(*selected))
        candidates = {
            candidate.move
            for candidate in ai.last_analysis.top_candidates
        }
        self.assertIn(parse_move("H6", board.size), candidates)
        self.assertIn(parse_move("L10", board.size), candidates)
        self.assertNotEqual(parse_move("E7", board.size), selected)
        self.assertIn("合并攻方强制点", ai.last_analysis.reason)
        self.assertIn("必防分支独立仲裁", ai.last_analysis.reason)
        self.assertTrue(ai.last_analysis.defense_vct_checked)
        self.assertEqual(
            "L10",
            format_move(*ai.last_analysis.defense_vct_best_move),
        )

    def test_all_near_mate_losses_expand_the_root_pool(self) -> None:
        board = Board()
        board.place(7, 7, BLACK)
        first = (7, 6)
        second = (6, 7)
        ai = SearchAI(
            player=WHITE,
            root_candidate_limit=6,
            time_limit_seconds=None,
        )
        result = RootResult(
            move=first,
            score=-MATE_SCORE + 3,
            principal_variation=(first,),
            ranked_moves=((first, -MATE_SCORE + 3),),
        )

        self.assertTrue(
            ai._all_root_candidates_near_forced_loss(
                result,
                [first],
            )
        )
        with patch.object(
            ai,
            "_ordered_moves",
            return_value=[first, second],
        ):
            expanded = ai._expand_near_loss_root_candidates(
                board,
                [first],
            )

        self.assertEqual([first, second], expanded)

    def test_expansion_registers_unprobed_moves_as_unknown(
        self,
    ) -> None:
        ai = SearchAI(player=BLACK, time_limit_seconds=None)
        probed = (7, 6)
        unprobed = (6, 7)
        ai._proof_candidates = (
            ProofCandidateAnalysis(
                move=probed,
                state=ProofState.UNKNOWN.value,
                completed=False,
                nodes=1,
                elapsed_seconds=0.01,
                threat_risk=0,
            ),
        )
        result = RootResult(
            move=unprobed,
            score=100,
            principal_variation=(unprobed,),
            ranked_moves=((unprobed, 100), (probed, 90)),
        )

        ai._register_expanded_candidates_as_unknown(
            [probed, unprobed]
        )

        states = {
            candidate.move: candidate.state
            for candidate in ai._proof_candidates
        }
        self.assertEqual(
            ProofState.UNKNOWN.value,
            states[unprobed],
        )
        self.assertIs(result, ai._apply_proof_tiebreak(result))


if __name__ == "__main__":
    unittest.main()
