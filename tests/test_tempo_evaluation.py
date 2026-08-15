from __future__ import annotations

import unittest
from unittest.mock import patch

from engine.ai import DecisionAnalysis
from engine.board import BLACK, WHITE, Board
from engine.evaluator import (
    DEFAULT_SEARCH_EVALUATION_CONFIG,
    STATIC_SEARCH_EVALUATION_CONFIG,
    EvaluationConfig,
    evaluate_board,
    evaluate_search_features,
    evaluate_search_position,
)
from engine.search import SearchAI


class TestTempoEvaluationSemantics(unittest.TestCase):
    TEST_CONFIG = EvaluationConfig(
        profile_name="tempo-test",
        initiative_open_three_bonus=101,
        initiative_jump_three_bonus=37,
        initiative_cap=200,
    )

    @staticmethod
    def _open_three_board(player: int = BLACK) -> Board:
        board = Board()
        for column in (6, 7, 8):
            board.place(7, column, player)
        return board

    def test_perspective_and_side_to_move_are_independent(self) -> None:
        board = self._open_three_board()

        black_view = evaluate_search_features(
            board,
            perspective=BLACK,
            side_to_move=BLACK,
            config=self.TEST_CONFIG,
        )
        white_view = evaluate_search_features(
            board,
            perspective=WHITE,
            side_to_move=BLACK,
            config=self.TEST_CONFIG,
        )

        self.assertEqual(evaluate_board(board, BLACK), black_view.static_score)
        self.assertEqual(1, black_view.side_to_move_open_threes)
        self.assertEqual(101, black_view.initiative_bonus)
        self.assertEqual(-black_view.total_score, white_view.total_score)
        self.assertEqual(
            -black_view.initiative_adjustment,
            white_view.initiative_adjustment,
        )

    def test_initiative_only_rewards_the_side_that_can_move(self) -> None:
        board = self._open_three_board()

        black_to_move = evaluate_search_position(
            board,
            BLACK,
            BLACK,
            config=self.TEST_CONFIG,
        )
        white_to_move = evaluate_search_position(
            board,
            BLACK,
            WHITE,
            config=self.TEST_CONFIG,
        )

        self.assertEqual(evaluate_board(board, BLACK), white_to_move)
        self.assertEqual(black_to_move, white_to_move + 101)

    def test_jump_three_has_smaller_bounded_initiative(self) -> None:
        board = Board()
        for column in (5, 6, 8):
            board.place(7, column, BLACK)

        breakdown = evaluate_search_features(
            board,
            BLACK,
            BLACK,
            config=self.TEST_CONFIG,
        )

        self.assertEqual(0, breakdown.side_to_move_open_threes)
        self.assertEqual(1, breakdown.side_to_move_jump_threes)
        self.assertEqual(37, breakdown.initiative_bonus)

    def test_initiative_is_capped_across_multiple_lines(self) -> None:
        board = Board()
        for move in (
            (4, 4),
            (4, 5),
            (4, 6),
            (8, 10),
            (9, 10),
            (10, 10),
        ):
            board.place(*move, BLACK)
        config = EvaluationConfig(
            profile_name="cap-test",
            initiative_open_three_bonus=12_000,
            initiative_jump_three_bonus=8_000,
            initiative_cap=10_000,
        )

        breakdown = evaluate_search_features(
            board,
            BLACK,
            BLACK,
            config=config,
        )

        self.assertEqual(2, breakdown.side_to_move_open_threes)
        self.assertEqual(10_000, breakdown.initiative_bonus)

    def test_forcing_patterns_do_not_receive_tempo_bonus(self) -> None:
        for columns in ((5, 6, 7, 8), (5, 6, 7, 8, 9)):
            with self.subTest(stones=len(columns)):
                board = Board()
                for column in columns:
                    board.place(7, column, BLACK)

                breakdown = evaluate_search_features(board, BLACK, BLACK)

                self.assertTrue(
                    breakdown.initiative_suppressed_by_forcing
                )
                self.assertEqual(0, breakdown.initiative_bonus)
                self.assertEqual(breakdown.static_score, breakdown.total_score)

    def test_any_forcing_pattern_suppresses_unrelated_initiative(self) -> None:
        board = self._open_three_board(BLACK)
        for column in (4, 5, 6, 7):
            board.place(10, column, WHITE)

        breakdown = evaluate_search_features(
            board,
            BLACK,
            BLACK,
            config=self.TEST_CONFIG,
        )

        self.assertEqual(1, breakdown.side_to_move_open_threes)
        self.assertTrue(breakdown.initiative_suppressed_by_forcing)
        self.assertEqual(0, breakdown.initiative_bonus)
        self.assertEqual(breakdown.static_score, breakdown.total_score)

    def test_blocked_three_does_not_receive_initiative(self) -> None:
        board = Board()
        board.place(7, 5, WHITE)
        for column in (6, 7, 8):
            board.place(7, column, BLACK)
        board.place(7, 9, WHITE)

        breakdown = evaluate_search_features(
            board,
            BLACK,
            BLACK,
            config=self.TEST_CONFIG,
        )

        self.assertEqual(0, breakdown.side_to_move_open_threes)
        self.assertEqual(0, breakdown.initiative_bonus)
        self.assertEqual(breakdown.static_score, breakdown.total_score)

    def test_static_profile_exactly_preserves_legacy_score(self) -> None:
        board = self._open_three_board()

        for perspective in (BLACK, WHITE):
            for side_to_move in (BLACK, WHITE):
                with self.subTest(
                    perspective=perspective,
                    side_to_move=side_to_move,
                ):
                    self.assertEqual(
                        evaluate_board(board, perspective),
                        evaluate_search_position(
                            board,
                            perspective,
                            side_to_move,
                            config=STATIC_SEARCH_EVALUATION_CONFIG,
                        ),
                    )

    def test_config_rejects_negative_parameters(self) -> None:
        with self.assertRaises(ValueError):
            EvaluationConfig(initiative_cap=-1)

    def test_config_rejects_non_integer_parameters(self) -> None:
        with self.assertRaises(TypeError):
            EvaluationConfig(initiative_cap=1.5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            EvaluationConfig(initiative_cap=True)

    def test_config_rejects_cap_above_tactical_safety_limit(self) -> None:
        with self.assertRaises(ValueError):
            EvaluationConfig(initiative_cap=10_001)


class TestSearchEvaluationConfig(unittest.TestCase):
    def test_search_instances_keep_independent_evaluation_profiles(self) -> None:
        board = TestTempoEvaluationSemantics._open_three_board()
        custom_tempo = EvaluationConfig(
            profile_name="isolation-test",
            initiative_open_three_bonus=321,
            initiative_jump_three_bonus=0,
            initiative_cap=321,
        )
        tempo = SearchAI(
            player=BLACK,
            time_limit_seconds=None,
            evaluation_config=custom_tempo,
        )
        static = SearchAI(
            player=BLACK,
            time_limit_seconds=None,
            evaluation_config=STATIC_SEARCH_EVALUATION_CONFIG,
        )

        static_before = static._static_score(board, BLACK)
        self.assertEqual(static_before + 321, tempo._static_score(board, BLACK))
        self.assertEqual(static_before, static._static_score(board, BLACK))
        self.assertEqual("isolation-test", tempo.evaluation_config.profile_name)
        self.assertEqual("static-v1", static.evaluation_config.profile_name)

        with self.assertRaises(AttributeError):
            tempo.evaluation_config = STATIC_SEARCH_EVALUATION_CONFIG

    def test_nested_search_probes_inherit_the_same_config(self) -> None:
        class ProbeConstructed(RuntimeError):
            pass

        config = EvaluationConfig(
            profile_name="probe-test",
            initiative_open_three_bonus=123,
            initiative_jump_three_bonus=45,
            initiative_cap=200,
        )
        ai = SearchAI(
            player=BLACK,
            max_depth=2,
            time_limit_seconds=None,
            evaluation_config=config,
        )
        board = Board()
        candidates = [(7, 7), (7, 8)]

        def assert_propagated(call) -> None:
            captured: dict[str, object] = {}

            def stop_at_constructor(**kwargs):
                captured.update(kwargs)
                raise ProbeConstructed

            with patch(
                "engine.search.SearchAI",
                side_effect=stop_at_constructor,
            ):
                with self.assertRaises(ProbeConstructed):
                    call()
            self.assertIs(config, captured["evaluation_config"])

        assert_propagated(
            lambda: ai._run_root_safety_probe(
                board,
                candidates,
                trigger="test",
                pvs_gap=0,
                main_rank_stable=False,
                completed_depth=1,
                budget_seconds=1.0,
            )
        )
        with patch.object(
            ai,
            "_defense_vct_budget_seconds",
            return_value=1.0,
        ):
            assert_propagated(
                lambda: ai._run_defense_vct_probe(
                    board,
                    BLACK,
                    candidates,
                )
            )
        with patch.object(
            ai,
            "_mandatory_defense_budget_seconds",
            return_value=1.0,
        ):
            assert_propagated(
                lambda: ai._run_mandatory_defense_probe(
                    board,
                    BLACK,
                    candidates,
                )
            )

    def test_analysis_serializes_evaluation_profile(self) -> None:
        payload = DecisionAnalysis(
            selected_move=(7, 7),
            reason="test",
            candidate_count=1,
            evaluation_profile="tempo-v1",
            evaluation_parameters=(("initiative_cap", 10_000),),
        ).to_dict()

        self.assertEqual("tempo-v1", payload["evaluation_profile"])
        self.assertEqual(
            {"initiative_cap": 10_000},
            payload["evaluation_parameters"],
        )

    def test_search_analysis_records_its_actual_config(self) -> None:
        config = EvaluationConfig(
            profile_name="recorded-test",
            initiative_open_three_bonus=123,
            initiative_jump_three_bonus=45,
            initiative_cap=200,
        )
        ai = SearchAI(
            player=BLACK,
            time_limit_seconds=None,
            evaluation_config=config,
            diagnostics=True,
        )

        self.assertEqual((7, 7), ai.choose_move(Board()))
        self.assertIsNotNone(ai.last_analysis)
        payload = ai.last_analysis.to_dict()
        self.assertEqual("recorded-test", payload["evaluation_profile"])
        self.assertEqual(
            dict(config.parameter_items()),
            payload["evaluation_parameters"],
        )

    def test_root_heuristic_uses_opponent_as_side_to_move(self) -> None:
        board = Board()
        for column in (6, 8):
            board.place(7, column, BLACK)
        original_grid = [row[:] for row in board.grid]
        original_history = board.move_history[:]
        config = EvaluationConfig(
            profile_name="root-tempo-test",
            initiative_open_three_bonus=321,
            initiative_jump_three_bonus=0,
            initiative_cap=321,
        )
        tempo = SearchAI(
            player=BLACK,
            time_limit_seconds=None,
            evaluation_config=config,
        )
        move = (7, 7)

        board.place(*move, BLACK)
        try:
            expected = tempo._static_score(
                board,
                perspective=BLACK,
                side_to_move=WHITE,
            )
            wrong_tempo = tempo._static_score(
                board,
                perspective=BLACK,
                side_to_move=BLACK,
            )
        finally:
            board.undo()

        self.assertEqual(expected, tempo._heuristic_root_score(board, move))
        self.assertEqual(expected + 321, wrong_tempo)
        self.assertEqual(original_grid, board.grid)
        self.assertEqual(original_history, board.move_history)


if __name__ == "__main__":
    unittest.main()
