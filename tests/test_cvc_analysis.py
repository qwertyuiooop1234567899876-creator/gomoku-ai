import tempfile
import unittest
from pathlib import Path

from cvc_analysis import (
    analyze_cvc_payload,
    classify_move,
    render_analysis_text,
)
from engine.board import BLACK, WHITE
from engine.yixin import YixinConfig, YixinSearchReport


class FakeAnalyzer:
    def __init__(self, reports: list[YixinSearchReport]) -> None:
        self.reports = list(reports)
        self.player = BLACK
        self.last_report = None

    def choose_move(self, _board) -> tuple[int, int]:
        self.last_report = self.reports.pop(0)
        if self.last_report.move is None:
            raise RuntimeError("fake report needs a move")
        return self.last_report.move

    def close(self) -> None:
        return None


def report(
    move: tuple[int, int],
    evaluation: int,
    *bestline: str,
) -> YixinSearchReport:
    return YixinSearchReport(
        move=move,
        depth=10,
        selective_depth=20,
        evaluation=evaluation,
        elapsed_ms=100,
        nodes=50_000,
        bestline=list(bestline),
    )


class TestMoveClassification(unittest.TestCase):
    def test_move_entering_mate_band_is_direct_blunder(self) -> None:
        classification, loss = classify_move(
            matches_recommendation=False,
            before_white=50,
            after_white=10_000,
            player=BLACK,
        )

        self.assertEqual("直接败着", classification)
        self.assertEqual(9_950, loss)

    def test_white_loss_uses_reversed_white_delta(self) -> None:
        classification, loss = classify_move(
            matches_recommendation=False,
            before_white=100,
            after_white=-100,
            player=WHITE,
        )

        self.assertEqual("明显失误", classification)
        self.assertEqual(200, loss)


class TestCVCAnalysis(unittest.TestCase):
    def test_analysis_finds_first_evaluation_cliff(self) -> None:
        payload = {
            "mode": "CVC",
            "black": "SearchAI",
            "white": "YiXin",
            "result": "白棋 O 获胜",
            "moves": [
                {
                    "player": BLACK,
                    "row": 7,
                    "column": 7,
                    "coordinate": "H8",
                }
            ],
        }
        analyzer = FakeAnalyzer(
            [
                report((7, 6), 0, "G8"),
                report((6, 7), 10_000, "H7"),
            ]
        )

        result = analyze_cvc_payload(payload, analyzer)

        move = result["moves"][0]
        self.assertEqual("H8", move["actual_move"])
        self.assertEqual("G8", move["recommended_move"])
        self.assertEqual("直接败着", move["classification"])
        self.assertEqual(
            1,
            result["first_decisive_blunder"]["number"],
        )

    def test_recommendations_and_white_view_are_recorded(self) -> None:
        payload = {
            "mode": "CVC",
            "black": "SearchAI",
            "white": "YiXin",
            "result": "未结束",
            "moves": [
                {"player": BLACK, "row": 7, "column": 7},
                {"player": WHITE, "row": 6, "column": 7},
            ],
        }
        analyzer = FakeAnalyzer(
            [
                report((7, 7), 0, "H8"),
                report((6, 7), 100, "H7"),
                report((7, 6), -120, "G8"),
            ]
        )

        result = analyze_cvc_payload(payload, analyzer)

        self.assertTrue(result["moves"][0]["matches_recommendation"])
        self.assertTrue(result["moves"][1]["matches_recommendation"])
        self.assertEqual(
            100,
            result["moves"][0]["evaluation_after_white"],
        )
        self.assertEqual(
            120,
            result["moves"][1]["evaluation_after_white"],
        )

    def test_text_report_names_decisive_move(self) -> None:
        payload = {
            "mode": "CVC",
            "black": "SearchAI",
            "white": "YiXin",
            "result": "白棋 O 获胜",
            "moves": [
                {"player": BLACK, "row": 7, "column": 7}
            ],
        }
        result = analyze_cvc_payload(
            payload,
            FakeAnalyzer(
                [
                    report((7, 6), 0, "G8"),
                    report((6, 7), 10_000, "H7"),
                ]
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "game.json"
            text = render_analysis_text(
                result,
                source_path=source,
                config=YixinConfig(),
                executable_sha256="abc",
            )

        self.assertIn("首个断崖败着：第 1 手 H8", text)
        self.assertIn("YiXin 推荐 G8", text)


if __name__ == "__main__":
    unittest.main()
