import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from arena import (
    _analysis_to_dict,
    create_ai,
    engine_display_name,
    play_game,
)
from engine.ai import RandomAI, ScoringAI, TacticalAI
from engine.arena_settings import AISelection
from engine.board import BLACK, WHITE, Board
from engine.evaluator import PositionEvaluation
from engine.search import SearchAI
from engine.yixin import YixinEngine


class TestArenaFactory(unittest.TestCase):
    def test_factory_creates_every_ai_stage(self) -> None:
        self.assertIsInstance(
            create_ai(AISelection("random"), BLACK),
            RandomAI,
        )
        self.assertIsInstance(
            create_ai(AISelection("tactical"), BLACK),
            TacticalAI,
        )
        self.assertIsInstance(
            create_ai(AISelection("scoring"), WHITE),
            ScoringAI,
        )
        self.assertIsInstance(
            create_ai(AISelection("search"), WHITE),
            SearchAI,
        )
        yixin = create_ai(AISelection("yixin"), WHITE)
        try:
            self.assertIsInstance(yixin, YixinEngine)
        finally:
            yixin.close()

    def test_search_sides_can_use_different_parameters(self) -> None:
        black = create_ai(
            AISelection("search", 5, 7.0),
            BLACK,
        )
        white = create_ai(
            AISelection("search", 2, 0.5),
            WHITE,
        )

        self.assertIsInstance(black, SearchAI)
        self.assertIsInstance(white, SearchAI)
        self.assertEqual(5, black.config.max_depth)
        self.assertEqual(7.0, black.config.time_limit_seconds)
        self.assertEqual(2, white.config.max_depth)
        self.assertEqual(0.5, white.config.time_limit_seconds)

    def test_display_name_contains_search_parameters(self) -> None:
        name = engine_display_name(
            AISelection("search", 4, 3.5)
        )

        self.assertEqual("SearchAI(d=4,t=3.5s)", name)

    def test_display_name_contains_yixin_parameters(self) -> None:
        name = engine_display_name(
            AISelection("yixin", 3, 10.0)
        )

        self.assertEqual("YiXin(t=10s,threads=2)", name)


class TestArenaAnalysisCompatibility(unittest.TestCase):
    def test_yixin_dict_analysis_is_accepted(self) -> None:
        analysis = {
            "engine_name": "yixin",
            "reason": "YiXin 外部核心协议搜索",
        }
        ai = type("FakeYixin", (), {"last_analysis": analysis})()

        payload = _analysis_to_dict(ai)

        self.assertEqual(analysis, payload)
        self.assertIsNot(analysis, payload)

    def test_search_analysis_object_is_still_serialized(self) -> None:
        class Analysis:
            def to_dict(self) -> dict[str, object]:
                return {
                    "engine_name": "search",
                    "reason": "PVS 搜索",
                }

        ai = type(
            "FakeSearch",
            (),
            {"last_analysis": Analysis()},
        )()

        self.assertEqual(
            {
                "engine_name": "search",
                "reason": "PVS 搜索",
            },
            _analysis_to_dict(ai),
        )


class TestArenaYixinEvaluationBar(unittest.TestCase):
    def test_watch_mode_uses_independent_yixin_evaluator(self) -> None:
        class RowAI:
            def __init__(self, player: int) -> None:
                self.player = player
                self.last_analysis = None

            def choose_move(self, board: Board) -> tuple[int, int]:
                row = 0 if self.player == BLACK else 1
                return next(
                    (row, column)
                    for column in range(board.size)
                    if board.is_empty(row, column)
                )

            def close(self) -> None:
                return None

        class FakeEvaluator:
            config = type(
                "Config",
                (),
                {"timeout_turn_seconds": 0.1},
            )()

            def __init__(self) -> None:
                self.calls: list[int] = []

            def evaluate_for_display(
                self,
                _board: Board,
                current_player: int,
            ) -> PositionEvaluation:
                self.calls.append(current_player)
                return PositionEvaluation(
                    source="YiXin",
                    score_white=-80,
                    raw_score=80,
                    depth=8,
                    selective_depth=16,
                    elapsed_seconds=0.1,
                )

        evaluator = FakeEvaluator()

        def make_ai(
            _selection: AISelection,
            player: int,
        ) -> RowAI:
            return RowAI(player)

        with (
            patch("arena.create_ai", side_effect=make_ai),
            patch(
                "arena.YixinPositionEvaluator.from_settings",
                return_value=evaluator,
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            result = play_game(
                black=AISelection("random"),
                white=AISelection("random"),
                watch=True,
                show_evaluation=True,
                save_record=False,
            )

        self.assertEqual(BLACK, result.winner)
        self.assertEqual(WHITE, evaluator.calls[0])
        self.assertIn("评价条：独立 YiXin", output.getvalue())
        self.assertIn("YiXin：黑棋 +80", output.getvalue())


if __name__ == "__main__":
    unittest.main()
