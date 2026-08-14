from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cvc_workflow
from arena import GameResult
from engine.records import RecordPaths


def fake_result(json_path: Path) -> GameResult:
    return GameResult(
        winner=None,
        move_count=0,
        duration_seconds=0.0,
        record_paths=RecordPaths(
            txt=json_path.with_suffix(".txt"),
            json=json_path,
        ),
    )


class TestCVCWorkflow(unittest.TestCase):
    def test_default_flow_uses_expected_engines_and_limits(self) -> None:
        selfplay, yixin = cvc_workflow.default_stages()

        self.assertEqual(("search", "search"), (
            selfplay.black.engine_name,
            selfplay.white.engine_name,
        ))
        self.assertEqual((8, 60.0), (
            selfplay.black.max_depth,
            selfplay.black.time_limit_seconds,
        ))
        self.assertEqual(("search", "yixin"), (
            yixin.black.engine_name,
            yixin.white.engine_name,
        ))
        self.assertEqual(10.0, yixin.white.time_limit_seconds)

    def test_each_game_analyzes_its_exact_new_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = (root / "selfplay.json", root / "yixin.json")
            for record in records:
                record.write_text("{}", encoding="utf-8")

            with (
                patch(
                    "cvc_workflow.play_game",
                    side_effect=[fake_result(path) for path in records],
                ) as play,
                patch("cvc_workflow.analyze_record") as analyze,
            ):
                returned = cvc_workflow.run_workflow()

        self.assertEqual(tuple(path.resolve() for path in records), returned)
        self.assertEqual(2, play.call_count)
        self.assertEqual(
            [(path.resolve(),) for path in records],
            [call.args for call in analyze.call_args_list],
        )
        for call in play.call_args_list:
            self.assertTrue(call.kwargs["save_record"])
            self.assertTrue(call.kwargs["watch"])
            self.assertFalse(call.kwargs["show_evaluation"])

    def test_analysis_failure_stops_before_second_game(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "selfplay.json"
            record.write_text("{}", encoding="utf-8")
            with (
                patch(
                    "cvc_workflow.play_game",
                    return_value=fake_result(record),
                ) as play,
                patch(
                    "cvc_workflow.analyze_record",
                    side_effect=subprocess.CalledProcessError(2, "analysis"),
                ),
            ):
                exit_code = cvc_workflow.main()

        self.assertEqual(1, exit_code)
        self.assertEqual(1, play.call_count)


if __name__ == "__main__":
    unittest.main()
