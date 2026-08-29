from __future__ import annotations

import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from engine import root_candidates
from engine.board import BLACK, Board
from engine.game import parse_move
from engine.proof_search import ProofResult, ProofState
from engine.search import SearchAI
from engine.search_types import RootResult, SearchConfig, SearchTimeout
from tools import native_search_baseline


FIXTURE = (
    Path(__file__).resolve().parent
    / "positions"
    / "v01617_selfplay_move14.json"
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


class TestV01618ActiveCounterattackFinalists(unittest.TestCase):
    def _build_real_finalist_case(
        self,
    ) -> tuple[
        dict[str, object],
        Board,
        SearchAI,
        RootResult,
        list[tuple[int, int]],
    ]:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        case = native_search_baseline.load_case(FIXTURE)
        board = native_search_baseline.build_board(case)
        moves = {
            coordinate: parse_move(coordinate, board.size)
            for coordinate in case.candidates
        }
        ranked = tuple(
            (moves[str(coordinate)], int(score))
            for coordinate, score in payload["root_review_ranked_moves"]
        )
        leader = moves[str(payload["root_result_move"])]
        result = RootResult(
            leader,
            dict(ranked)[leader],
            (leader,),
            ranked,
        )
        ai = SearchAI(
            case.player,
            max_depth=8,
            time_limit_seconds=None,
        )
        ai._begin_move_search()
        ai._root_candidate_sources = {
            moves[coordinate]: frozenset(
                root_candidates.CandidateSource(source)
                for source in sources
            )
            for coordinate, sources in payload[
                "root_candidate_sources"
            ].items()
        }
        ai._root_attack_priority = tuple(
            moves[coordinate]
            for coordinate in payload["root_attack_priority"]
        )
        ai._root_pressure_prevention = tuple(
            moves[coordinate]
            for coordinate in payload["root_pressure_prevention"]
        )
        return payload, board, ai, result, list(moves.values())

    def _run_real_finalist_flow(
        self,
        ai: SearchAI,
        board: Board,
        result: RootResult,
        candidates: list[tuple[int, int]],
        *,
        budget_seconds: float,
    ) -> object:
        with (
            patch.object(
                ai,
                "_dynamic_review_budget_seconds",
                return_value=budget_seconds,
            ),
            patch.object(
                ai,
                "_run_dynamic_pair_review",
                return_value=None,
            ) as pair_review,
        ):
            ai._maybe_run_dynamic_root_review(
                board,
                result,
                candidates,
                completed_depth=6,
            )
        return pair_review

    def _run_post_search_review_order(
        self,
        *,
        active_source: bool,
        dedicated_budget: float,
        dynamic_succeeds: bool,
    ) -> list[str]:
        board = Board()
        leader = (7, 7)
        challenger = (7, 8)
        candidates = [leader, challenger]
        result = RootResult(
            leader,
            100,
            (leader,),
            ((leader, 100), (challenger, 0)),
        )
        ai = SearchAI(
            BLACK,
            max_depth=4,
            time_limit_seconds=None,
        )
        ai._begin_move_search()
        ai._root_candidate_sources = {
            challenger: frozenset(
                (root_candidates.CandidateSource.ACTIVE_COUNTERATTACK,)
                if active_source
                else (root_candidates.CandidateSource.ORDINARY,)
            )
        }
        events: list[str] = []
        dynamic_probe = object() if dynamic_succeeds else None

        def search_root(
            _board: Board,
            _player: int,
            depth: int,
            _candidates: list[tuple[int, int]],
            **_kwargs: object,
        ) -> RootResult:
            if depth == 4:
                raise SearchTimeout
            return result

        def generic_review(*_args: object, **_kwargs: object) -> None:
            events.append("generic")
            return None

        def dynamic_review(*_args: object, **_kwargs: object) -> object:
            events.append("dynamic")
            return dynamic_probe

        with (
            patch.object(ai, "_search_root", side_effect=search_root),
            patch.object(
                ai,
                "_quarantine_unproven_root_scores",
                side_effect=lambda _board, current, **_kwargs: current,
            ),
            patch.object(ai, "_root_safety_trigger", return_value=None),
            patch.object(
                ai,
                "_active_counterattack_review_budget_seconds",
                return_value=dedicated_budget,
            ),
            patch.object(
                ai,
                "_maybe_run_root_safety_probe",
                side_effect=generic_review,
            ),
            patch.object(
                ai,
                "_maybe_run_dynamic_root_review",
                side_effect=dynamic_review,
            ),
            patch.object(
                ai,
                "_apply_and_record_root_safety",
                side_effect=lambda current, _probe: current,
            ),
        ):
            ai._run_iterative_root_search(
                board,
                candidates,
                fallback_move=leader,
                preserve_frontier_order=False,
                allow_near_loss_expansion=False,
                defense_probe=None,
            )
        return events

    def test_real_finalist_flow_reserves_screened_h10(self) -> None:
        payload, board, ai, result, candidates = (
            self._build_real_finalist_case()
        )
        before = native_search_baseline.board_state(board)

        pair_review = self._run_real_finalist_flow(
            ai,
            board,
            result,
            candidates,
            budget_seconds=6.0,
        )

        expected = tuple(
            parse_move(coordinate, board.size)
            for coordinate in payload["expected_screened_finalists"]
        )
        self.assertEqual(expected, ai._root_review_finalists)
        self.assertEqual(before, native_search_baseline.board_state(board))
        self.assertTrue(pair_review.call_args_list)
        first_call = pair_review.call_args_list[0]
        self.assertEqual(
            parse_move("H10", board.size),
            first_call.args[2],
        )
        self.assertTrue(first_call.kwargs["reject_mate_like"])
        self.assertEqual(
            5.0,
            first_call.kwargs["budget_seconds"],
        )

    def test_real_safe_pair_approves_h10_at_five_seconds(self) -> None:
        _payload, board, ai, result, candidates = (
            self._build_real_finalist_case()
        )
        with patch.object(
            ai,
            "_dynamic_review_budget_seconds",
            return_value=6.0,
        ):
            probe = ai._maybe_run_dynamic_root_review(
                board,
                result,
                candidates,
                completed_depth=6,
            )

        self.assertIsNotNone(probe)
        assert probe is not None
        self.assertEqual(parse_move("H10", board.size), probe.best_move)
        self.assertGreaterEqual(probe.completed_depth, 5)
        self.assertTrue(probe.rank_stable)
        self.assertEqual(
            "active_counterattack_safe_equal_window",
            probe.selection_basis,
        )

        revised = ai._apply_and_record_root_safety(result, probe)
        self.assertEqual(parse_move("H10", board.size), revised.move)
        self.assertEqual(revised.move, ai._root_review_confirmed_move)
        self.assertEqual(
            "active_counterattack_safe_equal_window",
            ai._root_review_confirmed_basis,
        )

        pressure = parse_move("I6", board.size)
        ai._root_pressure_prevention = (pressure,)
        with (
            patch("engine.search.ProofSearch", UnknownProofSearch),
            patch.object(
                ai,
                "_final_proof_budget_seconds",
                return_value=1.0,
            ),
        ):
            audited = ai._run_final_proof_audit(board, revised)
        self.assertEqual(revised.move, audited.move)
        self.assertEqual(
            "checked_unknown_review_confirmed",
            ai._final_proof_selection_basis,
        )

    def test_dedicated_pair_budget_shortfall_skips_h10_cleanly(
        self,
    ) -> None:
        _payload, board, ai, result, candidates = (
            self._build_real_finalist_case()
        )

        pair_review = self._run_real_finalist_flow(
            ai,
            board,
            result,
            candidates,
            budget_seconds=4.9,
        )

        h10 = parse_move("H10", board.size)
        self.assertTrue(pair_review.call_args_list)
        self.assertTrue(
            all(call.args[2] != h10 for call in pair_review.call_args_list)
        )
        self.assertEqual(
            tuple(
                parse_move(coordinate, board.size)
                for coordinate in (
                    "E7", "I6", "H7", "H10"
                )
            ),
            ai._root_review_finalists,
        )

    def test_active_screen_budget_shortfall_preserves_old_finalists(
        self,
    ) -> None:
        payload, board, ai, result, candidates = (
            self._build_real_finalist_case()
        )

        self._run_real_finalist_flow(
            ai,
            board,
            result,
            candidates,
            budget_seconds=0.49,
        )

        expected = tuple(
            parse_move(coordinate, board.size)
            for coordinate in payload["expected_unscreened_finalists"]
        )
        self.assertEqual(expected, ai._root_review_finalists)

    def test_partial_active_screen_is_discarded_and_restores_board(
        self,
    ) -> None:
        _payload, board, ai, _result, _candidates = (
            self._build_real_finalist_case()
        )
        active_moves = ai._root_attack_priority[:2]
        before = native_search_baseline.board_state(board)
        deadline = time.perf_counter() + 1.0

        with (
            patch.object(
                ai,
                "_selective_extension_reply_score",
                return_value=(100, None),
            ) as score,
            patch(
                "engine.search.time.perf_counter",
                side_effect=(deadline - 1.0, deadline - 0.4),
            ),
        ):
            representative = (
                ai._active_counterattack_finalist_representative(
                    board,
                    active_moves,
                    deadline=deadline,
                    minimum_remaining_seconds=0.5,
                )
            )

        self.assertIsNone(representative)
        self.assertEqual(1, score.call_count)
        self.assertEqual(before, native_search_baseline.board_state(board))

    def test_no_active_source_preserves_old_finalists(self) -> None:
        payload, board, ai, result, candidates = (
            self._build_real_finalist_case()
        )
        ai._root_attack_priority = ()
        ai._root_candidate_sources = {
            move: frozenset(
                source
                for source in sources
                if source
                is not root_candidates.CandidateSource.ACTIVE_COUNTERATTACK
            )
            for move, sources in ai._root_candidate_sources.items()
        }

        self._run_real_finalist_flow(
            ai,
            board,
            result,
            candidates,
            budget_seconds=5.0,
        )

        expected = tuple(
            parse_move(coordinate, board.size)
            for coordinate in payload["expected_unscreened_finalists"]
        )
        self.assertEqual(expected, ai._root_review_finalists)

    def test_active_representative_already_leader_is_not_duplicated(
        self,
    ) -> None:
        payload, board, ai, result, candidates = (
            self._build_real_finalist_case()
        )
        ai._root_attack_priority = (result.move,)
        ai._root_candidate_sources = {
            move: frozenset(
                source
                for source in sources
                if (
                    source
                    is not root_candidates.CandidateSource.ACTIVE_COUNTERATTACK
                    or move == result.move
                )
            )
            for move, sources in ai._root_candidate_sources.items()
        }

        self._run_real_finalist_flow(
            ai,
            board,
            result,
            candidates,
            budget_seconds=5.0,
        )

        expected = tuple(
            parse_move(coordinate, board.size)
            for coordinate in payload["expected_unscreened_finalists"]
        )
        self.assertEqual(expected, ai._root_review_finalists)
        self.assertEqual(
            len(ai._root_review_finalists),
            len(set(ai._root_review_finalists)),
        )

    def test_active_review_budget_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            SearchConfig(root_active_counterattack_review_seconds=0.0)
        with self.assertRaises(ValueError):
            SearchConfig(
                root_dynamic_review_max_seconds=4.0,
                root_active_counterattack_review_seconds=5.0,
            )

    def test_fully_funded_active_review_runs_before_generic_safety(
        self,
    ) -> None:
        self.assertEqual(
            ["dynamic", "generic"],
            self._run_post_search_review_order(
                active_source=True,
                dedicated_budget=5.25,
                dynamic_succeeds=False,
            ),
        )

    def test_successful_active_review_skips_weaker_generic_safety(
        self,
    ) -> None:
        self.assertEqual(
            ["dynamic"],
            self._run_post_search_review_order(
                active_source=True,
                dedicated_budget=5.25,
                dynamic_succeeds=True,
            ),
        )

    def test_no_active_source_preserves_generic_first_order(self) -> None:
        self.assertEqual(
            ["generic", "dynamic"],
            self._run_post_search_review_order(
                active_source=False,
                dedicated_budget=5.25,
                dynamic_succeeds=False,
            ),
        )

    def test_incomplete_active_budget_preserves_generic_first_order(
        self,
    ) -> None:
        self.assertEqual(
            ["generic", "dynamic"],
            self._run_post_search_review_order(
                active_source=True,
                dedicated_budget=0.0,
                dynamic_succeeds=False,
            ),
        )


if __name__ == "__main__":
    unittest.main()
