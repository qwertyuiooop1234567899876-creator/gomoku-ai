import unittest
from unittest.mock import patch

import arena
import main
from engine.ai import (
    ProofCandidateAnalysis,
    RootSafetyCandidateAnalysis,
)
from engine.board import BLACK, Board
from engine.game import format_move
from engine.proof_search import ProofState
from engine.search import (
    HEURISTIC_SCORE_LIMIT,
    RootResult,
    RootSafetyProbeResult,
    SearchAI,
)
from engine.time_manager import TimeManager
from engine.version import ENGINE_VERSION
from tests.test_v090_real_regressions import (
    E10_YIXIN_POSITION,
    E9_YIXIN_POSITION,
    I4_POSITION,
    J10_YIXIN_POSITION,
    board_state,
    build_position,
    parse_move,
)


class TestV090RootProofArbitration(unittest.TestCase):
    def test_i4_is_rejected_without_promoting_unknown_to_proof(
        self,
    ) -> None:
        board = build_position(I4_POSITION)
        before = board_state(board)
        ai = SearchAI(
            player=BLACK,
            max_depth=2,
            time_limit_seconds=4.0,
            diagnostics=True,
        )

        selected = ai.choose_move(board)

        self.assertNotEqual("I4", format_move(*selected))
        self.assertEqual("J5", format_move(*selected))
        self.assertEqual(before, board_state(board))
        self.assertIsNotNone(ai.last_analysis)
        assert ai.last_analysis is not None
        by_coordinate = {
            format_move(*candidate.move): candidate
            for candidate in ai.last_analysis.proof_candidates
        }
        self.assertIn("I4", by_coordinate)
        self.assertIn("J5", by_coordinate)
        self.assertEqual(
            ProofState.UNKNOWN.value,
            by_coordinate["I4"].state,
        )
        self.assertGreater(
            by_coordinate["I4"].threat_risk,
            by_coordinate["J5"].threat_risk,
        )

        payload = ai.last_analysis.to_dict()
        self.assertTrue(payload["proof_checked"])
        self.assertGreater(payload["proof_tt_queries"], 0)
        self.assertGreater(
            payload["threat_candidate_batches"],
            0,
        )
        self.assertIn("threat_cache_hits", payload)
        self.assertEqual(
            {"I4", "J5"},
            {
                candidate["coordinate"]
                for candidate in payload["proof_candidates"]
            },
        )

    def test_candidate_risk_survives_exhausted_proof_slice(
        self,
    ) -> None:
        board = build_position(I4_POSITION)
        before = board_state(board)
        ai = SearchAI(player=BLACK)
        i4 = parse_move("I4", board.size)
        j5 = parse_move("J5", board.size)

        with patch.object(
            ai,
            "_proof_budget_seconds",
            return_value=0.000_001,
        ):
            ai._run_proof_arbitration(
                board,
                [i4, j5],
                search_own_win=False,
            )

        by_move = {
            candidate.move: candidate
            for candidate in ai._proof_candidates
        }
        self.assertEqual(before, board_state(board))
        self.assertEqual(ProofState.UNKNOWN.value, by_move[i4].state)
        self.assertEqual(ProofState.UNKNOWN.value, by_move[j5].state)
        self.assertIsInstance(by_move[i4].threat_risk, int)
        self.assertIsInstance(by_move[j5].threat_risk, int)
        self.assertGreater(
            by_move[i4].threat_risk,
            by_move[j5].threat_risk,
        )

    def test_strict_candidate_state_precedes_pvs_score(self) -> None:
        ai = SearchAI(player=BLACK)
        safe = (4, 4)
        unknown = (4, 5)
        losing = (4, 6)
        ai._proof_candidates = (
            ProofCandidateAnalysis(
                move=safe,
                state=ProofState.PROVEN_LOSS.value,
                completed=True,
                nodes=10,
                elapsed_seconds=0.01,
            ),
            ProofCandidateAnalysis(
                move=unknown,
                state=ProofState.UNKNOWN.value,
                completed=False,
                nodes=20,
                elapsed_seconds=0.02,
            ),
            ProofCandidateAnalysis(
                move=losing,
                state=ProofState.PROVEN_WIN.value,
                completed=True,
                nodes=30,
                elapsed_seconds=0.03,
            ),
        )
        pvs = RootResult(
            move=losing,
            score=50_000,
            principal_variation=(losing,),
            ranked_moves=(
                (losing, 50_000),
                (unknown, 40_000),
                (safe, 30_000),
            ),
            ranked_variations=(
                (losing, 50_000, (losing,)),
                (unknown, 40_000, (unknown,)),
                (safe, 30_000, (safe,)),
            ),
        )

        filtered = ai._filter_proven_losing_candidates(
            [losing, unknown, safe]
        )
        decided = ai._apply_proof_tiebreak(pvs)

        self.assertEqual([unknown, safe], filtered)
        self.assertEqual(safe, decided.move)
        self.assertEqual(30_000, decided.score)

    def test_unknown_risk_cannot_cross_real_e9_pvs_gap(self) -> None:
        board = build_position(E9_YIXIN_POSITION)
        before = board_state(board)
        ai = SearchAI(player=BLACK)
        h5 = parse_move("H5", board.size)
        e5 = parse_move("E5", board.size)
        f5 = parse_move("F5", board.size)
        j5 = parse_move("J5", board.size)
        e9 = parse_move("E9", board.size)
        ai._proof_candidates = tuple(
            ProofCandidateAnalysis(
                move=move,
                state=ProofState.UNKNOWN.value,
                completed=False,
                nodes=1,
                elapsed_seconds=0.01,
                threat_risk=risk,
            )
            for move, risk in (
                (h5, 3_810_404),
                (f5, 3_810_404),
                (j5, 3_810_404),
                (e9, 2_208_804),
            )
        )
        pvs = RootResult(
            move=h5,
            score=899_000,
            principal_variation=(h5,),
            ranked_moves=(
                (h5, 899_000),
                (e5, 898_800),
                (f5, 890_100),
                (j5, 890_100),
                (e9, 1_100),
            ),
            ranked_variations=(
                (h5, 899_000, (h5,)),
                (e5, 898_800, (e5,)),
                (f5, 890_100, (f5,)),
                (j5, 890_100, (j5,)),
                (e9, 1_100, (e9,)),
            ),
        )

        decided = ai._apply_proof_tiebreak(pvs)

        self.assertEqual(h5, decided.move)
        self.assertEqual(899_000, decided.score)
        self.assertNotEqual(e9, decided.move)
        self.assertEqual(before, board_state(board))

    def test_unknown_risk_still_breaks_close_pvs_scores(self) -> None:
        ai = SearchAI(player=BLACK)
        pvs_best = (4, 4)
        lower_risk = (4, 5)
        ai._proof_candidates = (
            ProofCandidateAnalysis(
                move=pvs_best,
                state=ProofState.UNKNOWN.value,
                completed=False,
                nodes=1,
                elapsed_seconds=0.01,
                threat_risk=4_000_000,
            ),
            ProofCandidateAnalysis(
                move=lower_risk,
                state=ProofState.UNKNOWN.value,
                completed=False,
                nodes=1,
                elapsed_seconds=0.01,
                threat_risk=2_000_000,
            ),
        )
        pvs = RootResult(
            move=pvs_best,
            score=100_000,
            principal_variation=(pvs_best,),
            ranked_moves=(
                (pvs_best, 100_000),
                (lower_risk, 90_000),
            ),
            ranked_variations=(
                (pvs_best, 100_000, (pvs_best,)),
                (lower_risk, 90_000, (lower_risk,)),
            ),
        )

        decided = ai._apply_proof_tiebreak(pvs)

        self.assertEqual(lower_risk, decided.move)
        self.assertEqual(90_000, decided.score)

    def test_unknown_risk_does_not_reorder_strict_safe_moves(
        self,
    ) -> None:
        ai = SearchAI(player=BLACK)
        pvs_best = (4, 4)
        lower_risk = (4, 5)
        ai._proof_candidates = (
            ProofCandidateAnalysis(
                move=pvs_best,
                state=ProofState.PROVEN_LOSS.value,
                completed=True,
                nodes=10,
                elapsed_seconds=0.01,
                threat_risk=4_000_000,
            ),
            ProofCandidateAnalysis(
                move=lower_risk,
                state=ProofState.PROVEN_LOSS.value,
                completed=True,
                nodes=10,
                elapsed_seconds=0.01,
                threat_risk=2_000_000,
            ),
        )
        pvs = RootResult(
            move=pvs_best,
            score=100_000,
            principal_variation=(pvs_best,),
            ranked_moves=(
                (pvs_best, 100_000),
                (lower_risk, 90_000),
            ),
            ranked_variations=(
                (pvs_best, 100_000, (pvs_best,)),
                (lower_risk, 90_000, (lower_risk,)),
            ),
        )

        decided = ai._apply_proof_tiebreak(pvs)

        self.assertEqual(pvs_best, decided.move)
        self.assertEqual(100_000, decided.score)


class TestV091RootDecisionSafety(unittest.TestCase):
    @staticmethod
    def _j10_near_tie(board: Board) -> RootResult:
        j10 = parse_move("J10", board.size)
        j4 = parse_move("J4", board.size)
        return RootResult(
            move=j10,
            score=889_900,
            principal_variation=(j10,),
            ranked_moves=(
                (j10, 889_900),
                (j4, 889_100),
            ),
            ranked_variations=(
                (j10, 889_900, (j10,)),
                (j4, 889_100, (j4,)),
            ),
        )

    def test_micro_gap_triggers_even_with_a_stable_main_leader(
        self,
    ) -> None:
        board = build_position(J10_YIXIN_POSITION)
        ai = SearchAI(player=BLACK)
        result = self._j10_near_tie(board)

        trigger = ai._root_safety_trigger(
            result,
            [result, result],
        )

        self.assertEqual("micro_pvs_gap", trigger)

    def test_stable_non_micro_leader_does_not_spend_reserve(
        self,
    ) -> None:
        ai = SearchAI(player=BLACK)
        first = (4, 4)
        second = (4, 5)
        result = RootResult(
            move=first,
            score=100_000,
            principal_variation=(first,),
            ranked_moves=(
                (first, 100_000),
                (second, 90_000),
            ),
            ranked_variations=(
                (first, 100_000, (first,)),
                (second, 90_000, (second,)),
            ),
        )

        self.assertIsNone(
            ai._root_safety_trigger(
                result,
                [result, result, result],
            )
        )

    def test_strict_safe_candidate_is_not_overridden(self) -> None:
        board = build_position(J10_YIXIN_POSITION)
        ai = SearchAI(player=BLACK)
        result = self._j10_near_tie(board)
        ai._proof_candidates = (
            ProofCandidateAnalysis(
                move=result.move,
                state=ProofState.PROVEN_LOSS.value,
                completed=True,
                nodes=10,
                elapsed_seconds=0.01,
            ),
        )

        self.assertIsNone(
            ai._root_safety_trigger(
                result,
                [result, result],
            )
        )

    def test_real_j10_near_tie_is_independently_rechecked_as_j4(
        self,
    ) -> None:
        board = build_position(J10_YIXIN_POSITION)
        before = board_state(board)
        ai = SearchAI(
            player=BLACK,
            max_depth=8,
            time_limit_seconds=60.0,
        )
        # Use the production 6-second safety budget.  The former 0.8-second
        # scaled budget made this real-position regression depend on CPU
        # speed: slower Windows hosts could finish only J10 -> J4, correctly
        # leaving the rank unstable before the confirming J4 layer.
        ai._time = TimeManager.start(60.0, soft_ratio=0.99)
        result = self._j10_near_tie(board)

        probe = ai._maybe_run_root_safety_probe(
            board,
            result,
            completed_depth=5,
            root_history=[result, result],
        )

        self.assertIsNotNone(probe)
        assert probe is not None
        self.assertTrue(probe.rank_stable)
        self.assertGreaterEqual(probe.completed_depth, 3)
        self.assertEqual("J4", format_move(*probe.best_move))
        self.assertEqual(before, board_state(board))

        decided = ai._apply_root_safety_probe(result, probe)

        self.assertEqual("J4", format_move(*decided.move))
        self.assertNotEqual("J10", format_move(*decided.move))
        self.assertEqual(889_100, decided.score)
        self.assertEqual(before, board_state(board))

    def test_real_e10_risk_override_is_rechecked_and_keeps_i6(
        self,
    ) -> None:
        board = build_position(E10_YIXIN_POSITION)
        before = board_state(board)
        ai = SearchAI(
            player=BLACK,
            max_depth=8,
            time_limit_seconds=60.0,
        )
        ai._time = TimeManager.start(60.0, soft_ratio=0.99)
        i6 = parse_move("I6", board.size)
        e10 = parse_move("E10", board.size)
        ai._proof_candidates = (
            ProofCandidateAnalysis(
                move=i6,
                state=ProofState.UNKNOWN.value,
                completed=False,
                nodes=269,
                elapsed_seconds=7.20,
                cutoff_reason="deadline",
                threat_risk=5_780_559,
            ),
            ProofCandidateAnalysis(
                move=e10,
                state=ProofState.UNKNOWN.value,
                completed=False,
                nodes=215,
                elapsed_seconds=7.18,
                cutoff_reason="deadline",
                threat_risk=5_329_709,
            ),
        )
        pvs = RootResult(
            move=i6,
            score=-990_100,
            principal_variation=(i6,),
            ranked_moves=(
                (i6, -990_100),
                (e10, -1_000_100),
            ),
            ranked_variations=(
                (i6, -990_100, (i6,)),
                (e10, -1_000_100, (e10,)),
            ),
        )
        risk_result = ai._apply_proof_tiebreak(pvs)

        self.assertEqual("E10", format_move(*risk_result.move))
        self.assertTrue(
            ai._is_unknown_risk_override(pvs, risk_result)
        )

        decided = ai._finalize_risk_override(
            board,
            pvs,
            risk_result,
            completed_depth=8,
            root_history=[pvs, risk_result],
        )

        probe = ai._root_safety_probe
        self.assertIsNotNone(probe)
        assert probe is not None
        self.assertEqual("threat_risk_override", probe.trigger)
        confirmation_ready = (
            probe.rank_stable
            and probe.completed_depth
            >= ai.config.root_safety_min_completed_depth
        )
        if confirmation_ready:
            self.assertEqual("I6", format_move(*probe.best_move))
        else:
            # A slower host may finish only I6 -> I6 within the production
            # six-second budget.  Repeated leadership alone is insufficient:
            # the minimum-depth gate must reject the heuristic override and
            # preserve the original PVS choice.
            self.assertFalse(ai._root_safety_applied)

        self.assertEqual("I6", format_move(*decided.move))
        self.assertNotEqual("E10", format_move(*decided.move))
        self.assertEqual(-990_100, decided.score)
        self.assertEqual(before, board_state(board))

    def test_unconfirmed_risk_override_falls_back_to_pvs(
        self,
    ) -> None:
        board = build_position(E10_YIXIN_POSITION)
        before = board_state(board)
        ai = SearchAI(
            player=BLACK,
            max_depth=8,
            time_limit_seconds=60.0,
        )
        i6 = parse_move("I6", board.size)
        e10 = parse_move("E10", board.size)
        ai._proof_candidates = (
            ProofCandidateAnalysis(
                move=i6,
                state=ProofState.UNKNOWN.value,
                completed=False,
                nodes=1,
                elapsed_seconds=0.01,
                threat_risk=5_780_559,
            ),
            ProofCandidateAnalysis(
                move=e10,
                state=ProofState.UNKNOWN.value,
                completed=False,
                nodes=1,
                elapsed_seconds=0.01,
                threat_risk=5_329_709,
            ),
        )
        pvs = RootResult(
            move=i6,
            score=-990_100,
            principal_variation=(i6,),
            ranked_moves=(
                (i6, -990_100),
                (e10, -1_000_100),
            ),
            ranked_variations=(
                (i6, -990_100, (i6,)),
                (e10, -1_000_100, (e10,)),
            ),
        )
        risk_result = ai._apply_proof_tiebreak(pvs)
        ai._time = TimeManager.start(0.001, soft_ratio=0.99)

        decided = ai._finalize_risk_override(
            board,
            pvs,
            risk_result,
            completed_depth=8,
            root_history=[pvs, risk_result],
        )

        probe = ai._root_safety_probe
        self.assertIsNotNone(probe)
        assert probe is not None
        self.assertFalse(probe.rank_stable)
        self.assertEqual(0, probe.completed_depth)
        self.assertEqual("I6", format_move(*decided.move))
        self.assertEqual(before, board_state(board))

    def test_stable_probe_can_still_approve_unknown_risk_override(
        self,
    ) -> None:
        board = Board()
        before = board_state(board)
        ai = SearchAI(player=BLACK)
        pvs_move = (7, 7)
        risk_move = (7, 8)
        pvs = RootResult(
            move=pvs_move,
            score=100_000,
            principal_variation=(pvs_move,),
            ranked_moves=(
                (pvs_move, 100_000),
                (risk_move, 90_000),
            ),
            ranked_variations=(
                (pvs_move, 100_000, (pvs_move,)),
                (risk_move, 90_000, (risk_move,)),
            ),
        )
        risk_result = RootResult(
            move=risk_move,
            score=90_000,
            principal_variation=(risk_move,),
            ranked_moves=(
                (risk_move, 90_000),
                (pvs_move, 100_000),
            ),
            ranked_variations=(
                (risk_move, 90_000, (risk_move,)),
                (pvs_move, 100_000, (pvs_move,)),
            ),
        )
        probe = RootSafetyProbeResult(
            trigger="threat_risk_override",
            pvs_gap=10_000,
            main_rank_stable=True,
            completed_depth=3,
            nodes=20,
            candidates=(
                RootSafetyCandidateAnalysis(
                    move=risk_move,
                    score=2_000,
                    principal_variation=(risk_move,),
                ),
                RootSafetyCandidateAnalysis(
                    move=pvs_move,
                    score=1_000,
                    principal_variation=(pvs_move,),
                ),
            ),
            leader_history=(risk_move, risk_move),
        )

        with patch.object(
            ai,
            "_maybe_run_risk_override_probe",
            return_value=probe,
        ):
            decided = ai._finalize_risk_override(
                board,
                pvs,
                risk_result,
                completed_depth=8,
                root_history=[pvs, risk_result],
            )

        self.assertEqual(risk_move, decided.move)
        self.assertTrue(ai._root_safety_applied)
        self.assertIs(probe, ai._root_safety_probe)
        self.assertEqual(before, board_state(board))

        shallow_probe = RootSafetyProbeResult(
            trigger=probe.trigger,
            pvs_gap=probe.pvs_gap,
            main_rank_stable=probe.main_rank_stable,
            completed_depth=2,
            nodes=probe.nodes,
            candidates=probe.candidates,
            leader_history=probe.leader_history,
        )
        self.assertTrue(shallow_probe.rank_stable)

        with patch.object(
            ai,
            "_maybe_run_risk_override_probe",
            return_value=shallow_probe,
        ):
            decided = ai._finalize_risk_override(
                board,
                pvs,
                risk_result,
                completed_depth=8,
                root_history=[pvs, risk_result],
            )

        self.assertEqual(pvs_move, decided.move)
        self.assertFalse(ai._root_safety_applied)
        self.assertIs(shallow_probe, ai._root_safety_probe)
        self.assertEqual(before, board_state(board))


class TestV090ScoreAndVersionSeparation(unittest.TestCase):
    def test_heuristic_score_cannot_enter_terminal_band(self) -> None:
        ai = SearchAI(player=BLACK)
        board = Board()

        with patch(
            "engine.search.evaluate_search_position",
            return_value=10_000_000_000,
        ):
            self.assertEqual(
                HEURISTIC_SCORE_LIMIT,
                ai._static_score(board, BLACK),
            )
        with patch(
            "engine.search.evaluate_search_position",
            return_value=-10_000_000_000,
        ):
            self.assertEqual(
                -HEURISTIC_SCORE_LIMIT,
                ai._static_score(board, BLACK),
            )

    def test_public_entry_points_share_v090_version(self) -> None:
        self.assertEqual("0.16.2", ENGINE_VERSION)
        self.assertEqual(ENGINE_VERSION, main.ENGINE_VERSION)
        self.assertEqual(ENGINE_VERSION, arena.ENGINE_VERSION)


if __name__ == "__main__":
    unittest.main()
