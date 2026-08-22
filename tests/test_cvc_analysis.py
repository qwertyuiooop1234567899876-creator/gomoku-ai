import tempfile
import unittest
from pathlib import Path

from tools.cvc_analysis import (
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
    def test_real_selfplay_move_eight_uses_paired_children(self) -> None:
        coordinates = "H8 I7 H6 H9 H7 H4 G7 I9".split()
        payload = {
            "mode": "CVC",
            "black": "SearchAI",
            "white": "SearchAI",
            "result": "未结束",
            "moves": [
                {
                    "player": BLACK if index % 2 == 0 else WHITE,
                    "row": int(coordinate[1:]) - 1,
                    "column": ord(coordinate[0]) - ord("A"),
                }
                for index, coordinate in enumerate(coordinates)
            ],
        }
        analyzer = FakeAnalyzer(
            [
                # Before move 8 YiXin recommends F8.
                report((7, 5), -174, "F8", "F6"),
                # Counterfactual child after F8: white-view -239.
                report((8, 8), 239, "I9"),
                # Actual child after I9: black-to-move +10000,
                # normalized to white-view -10000.
                report((7, 5), 10_000, "F8"),
            ]
        )

        result = analyze_cvc_payload(
            payload,
            analyzer,
            first_move=8,
            last_move=8,
        )

        move = result["moves"][0]
        self.assertEqual("I9", move["actual_move"])
        self.assertEqual("F8", move["recommended_move"])
        self.assertEqual(-10_000, move["evaluation_after_white"])
        self.assertEqual(
            -239,
            move["evaluation_after_recommended_white"],
        )
        self.assertEqual("直接败着", move["classification"])
        self.assertEqual(9_761, move["loss_for_mover"])
        self.assertEqual(
            "paired_child_positions",
            move["comparison_basis"],
        )

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
                report((6, 7), 50, "H7"),
                report((6, 7), 10_000, "H7"),
            ]
        )

        result = analyze_cvc_payload(payload, analyzer)

        move = result["moves"][0]
        self.assertEqual("H8", move["actual_move"])
        self.assertEqual("G8", move["recommended_move"])
        self.assertEqual("直接败着", move["classification"])
        self.assertEqual(9_950, move["loss_for_mover"])
        self.assertEqual(
            50,
            move["evaluation_after_recommended_white"],
        )
        self.assertEqual(
            "paired_child_positions",
            move["comparison_basis"],
        )
        self.assertEqual(
            1,
            result["first_decisive_blunder"]["number"],
        )
        self.assertEqual(
            1,
            result["first_decisive_blunder_by_player"]["BLACK"][
                "number"
            ],
        )
        self.assertIsNone(
            result["first_decisive_blunder_by_player"]["WHITE"]
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
                    report((6, 7), 50, "H7"),
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
        self.assertIn("黑方首个断崖败着：第 1 手 H8", text)
        self.assertIn("YiXin 推荐 G8", text)
        self.assertIn("同根对照：实战着后 +10000；推荐着后 +50", text)

    def test_mismatched_return_still_uses_completed_position_score(
        self,
    ) -> None:
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
                report((8, 7), -50, "I8", "G7"),
                report((6, 7), 0, "H7"),
                report((6, 7), 10_000, "H7"),
                report((8, 6), 0, "F8"),
            ]
        )

        result = analyze_cvc_payload(payload, analyzer)
        move = result["moves"][0]

        self.assertEqual("H9", move["recommended_move"])
        self.assertEqual(
            "I8",
            move["completed_best_move_before"],
        )
        self.assertFalse(move["evaluation_aligned_before"])
        self.assertEqual(10_000, move["loss_for_mover"])
        self.assertEqual("直接败着", move["classification"])
        self.assertEqual(1, result["first_decisive_blunder"]["number"])
        self.assertEqual(1, result["largest_losses"][0]["number"])

        with tempfile.TemporaryDirectory() as directory:
            text = render_analysis_text(
                result,
                source_path=Path(directory) / "game.json",
                config=YixinConfig(),
                executable_sha256="abc",
            )
        self.assertIn("最终返回 H9；完成层首选 I8", text)
        self.assertIn("仍按完成层局面分数统计", text)

    def test_unaligned_completed_mate_score_remains_comparable(
        self,
    ) -> None:
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
                report((8, 7), -50, "I8"),
                report((6, 7), 0, "H7"),
                report((6, 7), 10_000, "H7"),
                report((8, 6), 0, "F8"),
            ]
        )

        result = analyze_cvc_payload(payload, analyzer)

        self.assertEqual("直接败着", result["moves"][0]["classification"])
        self.assertEqual(10_000, result["moves"][0]["loss_for_mover"])
        self.assertEqual(1, result["first_decisive_blunder"]["number"])
        self.assertEqual(1, result["largest_losses"][0]["number"])

    def test_recommended_move_is_not_blamed_for_adjacent_score_jump(
        self,
    ) -> None:
        payload = {
            "mode": "CVC",
            "black": "SearchAI",
            "white": "YiXin",
            "result": "未结束",
            "moves": [
                {"player": BLACK, "row": 7, "column": 7},
            ],
        }
        result = analyze_cvc_payload(
            payload,
            FakeAnalyzer(
                [
                    report((7, 7), 0, "H8"),
                    report((6, 7), 10_000, "H7"),
                ]
            ),
        )

        move = result["moves"][0]
        self.assertEqual("推荐一致", move["classification"])
        self.assertEqual(0, move["loss_for_mover"])
        self.assertIsNone(result["first_decisive_blunder"])


if __name__ == "__main__":
    unittest.main()
