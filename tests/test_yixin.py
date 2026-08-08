import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from engine.board import BLACK, WHITE, Board
from engine.yixin import (
    YixinConfig,
    YixinConfigurationError,
    YixinEngine,
    YixinPositionEvaluator,
    YixinSearchReport,
    load_yixin_config,
    render_yixin_evaluation_bar,
    save_yixin_config,
)


FAKE_ENGINE = """
import sys

log_path = sys.argv[1]
board = []
position = {}

def log(line):
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(line + "\\n")

def play():
    move = next(
        candidate
        for candidate in ((7, 7), (7, 6), (6, 7), (8, 7))
        if candidate not in position
    )
    position[move] = 1
    coordinate = f"[{chr(ord('A') + move[1])},{15 - move[0]}]"
    print(
        "MESSAGE DETAIL DEPTH:12-26 VAL:141 "
        "TIME:1250MS NODE:3M " + coordinate,
        flush=True,
    )
    print(
        "MESSAGE Speed: 2400 | Evaluation: 141",
        flush=True,
    )
    print(
        "MESSAGE Bestline: " + coordinate + " [H,9]",
        flush=True,
    )
    print(f"{move[0]},{move[1]}", flush=True)

for raw in sys.stdin:
    line = raw.strip()
    log(line)
    upper = line.upper()
    if upper.startswith("START "):
        print("MESSAGE fake yixin", flush=True)
        print("OK", flush=True)
    elif upper == "RESTART":
        board = []
        position = {}
        print("OK", flush=True)
    elif upper.startswith("TURN "):
        x, y = (
            int(value)
            for value in line.split(maxsplit=1)[1].split(",")
        )
        if (x, y) in position:
            print(
                f"ERROR opponents's move [{x},{y}]",
                flush=True,
            )
        else:
            position[(x, y)] = 2
            play()
    elif upper == "BOARD":
        board = []
    elif upper == "DONE":
        rejected = False
        for item in board:
            x, y, field = (int(value) for value in item.split(","))
            coordinate = (x, y)
            if coordinate in position:
                print(
                    f"ERROR duplicate move [{x},{y}]",
                    flush=True,
                )
                rejected = True
                break
            position[coordinate] = field
        if rejected:
            continue
        play()
    elif upper == "END":
        break
    elif "," in line and line.count(",") == 2:
        board.append(line)
"""


class TestYixinConfiguration(unittest.TestCase):
    def test_default_settings_match_confirmed_profile(self) -> None:
        config = YixinConfig()

        self.assertEqual(2, config.thread_num)
        self.assertEqual(6, config.thread_split_depth)
        self.assertEqual(21, config.hash_size)
        self.assertEqual(2, config.caution_factor)
        self.assertEqual(0, config.checkmate)
        self.assertFalse(config.pondering)
        self.assertEqual(2.0, config.evaluation_time_seconds)

    def test_round_trip_preserves_optional_limits(self) -> None:
        config = YixinConfig(
            executable_path="custom/engine.exe",
            launch_arguments=("--fake",),
            timeout_turn_seconds=3.5,
            max_depth=18,
            max_node=2_000_000,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yixin.json"
            save_yixin_config(config, path)
            loaded = load_yixin_config(path)

        self.assertEqual(config, loaded)

    def test_bad_hash_size_is_rejected(self) -> None:
        with self.assertRaises(YixinConfigurationError):
            YixinConfig(hash_size=0)


class TestYixinReportParser(unittest.TestCase):
    def test_detail_summary_and_bestline_are_parsed(self) -> None:
        report = YixinSearchReport()
        report.consume(
            "MESSAGE DETAIL DEPTH:13-26 VAL:-115 "
            "TIME:6352MS NODE:5M [I,9]"
        )
        report.consume(
            "MESSAGE Speed: 802 | Evaluation: -115"
        )
        report.consume(
            "MESSAGE Bestline: [I,9] [J,8] [K,7]"
        )

        self.assertEqual(13, report.depth)
        self.assertEqual(26, report.selective_depth)
        self.assertEqual(-115, report.evaluation)
        self.assertEqual(6352, report.elapsed_ms)
        self.assertEqual(5_000_000, report.nodes)
        self.assertEqual(802, report.speed)
        self.assertEqual(["G9", "H10", "I11"], report.bestline)

    def test_display_coordinates_align_with_numeric_protocol_move(
        self,
    ) -> None:
        report = YixinSearchReport(move=(8, 7), evaluation=10_000)
        report.consume("MESSAGE REALTIME BEST 7,8")
        report.consume("MESSAGE Bestline: [I,8] [G,7]")

        self.assertEqual("H9", report.coordinate)
        self.assertEqual("H9", report.realtime_coordinate)
        self.assertEqual(["H9", "I7"], report.bestline)
        self.assertEqual("H9", report.completed_best_coordinate)
        self.assertTrue(report.evaluation_aligned_with_move)

        analysis = report.to_analysis_dict(
            player=WHITE,
            requested_seconds=10.0,
        )
        self.assertEqual("H9", analysis["returned_coordinate"])
        self.assertEqual(
            "H9",
            analysis["completed_best_coordinate"],
        )
        self.assertEqual("H9", analysis["evaluation_coordinate"])
        self.assertTrue(
            analysis["evaluation_aligned_with_returned_move"]
        )
        self.assertEqual(
            "H9",
            analysis["top_candidates"][0]["coordinate"],
        )

    def test_returned_move_can_differ_from_completed_bestline(
        self,
    ) -> None:
        report = YixinSearchReport(move=(8, 7), evaluation=120)
        report.consume("MESSAGE Bestline: [H,7] [G,7]")

        self.assertEqual("H9", report.coordinate)
        self.assertEqual("I8", report.completed_best_coordinate)
        self.assertFalse(report.evaluation_aligned_with_move)

    def test_known_review_coordinates_use_yixin_rotation(self) -> None:
        report = YixinSearchReport()
        report.consume(
            "MESSAGE Bestline: [I,8] [G,8] [K,13] [O,1]"
        )

        self.assertEqual(
            ["H9", "H7", "C11", "O15"],
            report.bestline,
        )

    def test_white_perspective_is_normalized(self) -> None:
        report = YixinSearchReport(
            move=(7, 7),
            evaluation=120,
        )

        black = report.to_analysis_dict(
            player=BLACK,
            requested_seconds=10.0,
        )
        white = report.to_analysis_dict(
            player=WHITE,
            requested_seconds=10.0,
        )

        self.assertEqual(-120, black["evaluation_white"])
        self.assertEqual(120, white["evaluation_white"])


class TestYixinProtocolClient(unittest.TestCase):
    def _config(
        self,
        script_path: Path,
        log_path: Path,
    ) -> YixinConfig:
        return YixinConfig(
            executable_path=sys.executable,
            launch_arguments=(str(script_path), str(log_path)),
            timeout_turn_seconds=1.0,
            startup_timeout_seconds=1.0,
            response_grace_seconds=1.0,
        )

    def test_board_protocol_returns_legal_moves_and_diagnostics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script_path = root / "fake_yixin.py"
            log_path = root / "commands.txt"
            script_path.write_text(
                textwrap.dedent(FAKE_ENGINE),
                encoding="utf-8",
            )
            engine = YixinEngine(
                player=BLACK,
                config=self._config(script_path, log_path),
            )
            board = Board()
            try:
                self.assertEqual((7, 7), engine.choose_move(board))
                self.assertEqual([], board.move_history)
                self.assertEqual(12, engine.last_report.depth)
                self.assertEqual(["H8", "G8"], engine.last_report.bestline)

                board.place(7, 7, BLACK)
                board.place(6, 7, WHITE)
                self.assertEqual((7, 6), engine.choose_move(board))
            finally:
                engine.close()

            commands = log_path.read_text(encoding="utf-8").splitlines()

        self.assertIn("START 15", commands)
        self.assertIn("INFO thread_num 2", commands)
        self.assertIn("INFO checkmate 0", commands)
        self.assertEqual(1, commands.count("BOARD"))
        self.assertEqual(0, commands.count("RESTART"))
        self.assertIn("TURN 7,6", commands)
        self.assertIn("END", commands)

    def test_changed_board_restarts_before_full_resync(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script_path = root / "fake_yixin.py"
            log_path = root / "commands.txt"
            script_path.write_text(
                textwrap.dedent(FAKE_ENGINE),
                encoding="utf-8",
            )
            engine = YixinEngine(
                player=BLACK,
                config=self._config(script_path, log_path),
            )
            try:
                self.assertEqual((7, 7), engine.choose_move(Board()))

                changed = Board()
                changed.place(0, 0, BLACK)
                changed.place(0, 1, WHITE)
                self.assertEqual((7, 7), engine.choose_move(changed))
            finally:
                engine.close()

            commands = log_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(2, commands.count("BOARD"))
        self.assertEqual(1, commands.count("RESTART"))
        first_board = commands.index("BOARD")
        restart = commands.index("RESTART")
        second_board = commands.index("BOARD", first_board + 1)
        self.assertLess(first_board, restart)
        self.assertLess(restart, second_board)

    def test_white_yixin_uses_turn_after_first_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script_path = root / "fake_yixin.py"
            log_path = root / "commands.txt"
            script_path.write_text(
                textwrap.dedent(FAKE_ENGINE),
                encoding="utf-8",
            )
            engine = YixinEngine(
                player=WHITE,
                config=self._config(script_path, log_path),
            )
            board = Board()
            board.place(7, 7, BLACK)
            try:
                move = engine.choose_move(board)
                self.assertEqual((6, 7), move)
                board.place(*move, WHITE)
                board.place(7, 6, BLACK)
                self.assertEqual((7, 8), engine.choose_move(board))
            finally:
                engine.close()

            commands = log_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(1, commands.count("BOARD"))
        self.assertEqual(0, commands.count("RESTART"))
        self.assertEqual(1, commands.count("7,7,2"))
        self.assertIn("TURN 6,7", commands)

    def test_board_players_are_relative_to_white_yixin(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script_path = root / "fake_yixin.py"
            log_path = root / "commands.txt"
            script_path.write_text(
                textwrap.dedent(FAKE_ENGINE),
                encoding="utf-8",
            )
            engine = YixinEngine(
                player=WHITE,
                config=self._config(script_path, log_path),
            )
            board = Board()
            board.place(7, 7, BLACK)
            board.place(7, 6, WHITE)
            board.place(6, 7, BLACK)
            try:
                self.assertEqual((7, 8), engine.choose_move(board))
            finally:
                engine.close()

            commands = log_path.read_text(encoding="utf-8").splitlines()

        self.assertIn("7,7,2", commands)
        self.assertIn("6,7,1", commands)
        self.assertIn("7,6,2", commands)
        self.assertNotIn("7,7,1", commands)

    def test_missing_executable_fails_before_launch(self) -> None:
        engine = YixinEngine(
            player=WHITE,
            config=YixinConfig(
                executable_path="definitely-missing-engine.exe"
            ),
        )
        with self.assertRaises(YixinConfigurationError):
            engine.start()


class TestYixinPositionEvaluator(unittest.TestCase):
    def _config(
        self,
        script_path: Path,
        log_path: Path,
    ) -> YixinConfig:
        return YixinConfig(
            executable_path=sys.executable,
            launch_arguments=(str(script_path), str(log_path)),
            timeout_turn_seconds=10.0,
            evaluation_time_seconds=0.5,
            startup_timeout_seconds=1.0,
            response_grace_seconds=1.0,
        )

    def test_position_evaluation_is_white_normalized_and_cached(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script_path = root / "fake_yixin.py"
            log_path = root / "commands.txt"
            script_path.write_text(
                textwrap.dedent(FAKE_ENGINE),
                encoding="utf-8",
            )
            evaluator = YixinPositionEvaluator(
                config=self._config(script_path, log_path),
            )
            board = Board()
            board.place(0, 0, WHITE)
            grid_before = [row.copy() for row in board.grid]
            history_before = board.move_history.copy()

            first = evaluator.evaluate(board, BLACK)
            second = evaluator.evaluate(board, BLACK)
            commands = log_path.read_text(encoding="utf-8").splitlines()

        self.assertIs(first, second)
        self.assertEqual(-141, first.score_white)
        self.assertEqual(141, first.raw_score)
        self.assertEqual(12, first.depth)
        self.assertEqual(26, first.selective_depth)
        self.assertEqual(("H8", "G8"), first.bestline)
        self.assertEqual(grid_before, board.grid)
        self.assertEqual(history_before, board.move_history)
        self.assertEqual(1, commands.count("START 15"))
        self.assertIn("INFO timeout_turn 500", commands)

    def test_missing_core_is_rendered_without_static_fallback(self) -> None:
        evaluator = YixinPositionEvaluator(
            config=YixinConfig(
                executable_path="definitely-missing-engine.exe",
                evaluation_time_seconds=0.5,
            ),
        )
        board = Board()
        board.place(7, 7, BLACK)

        bar = render_yixin_evaluation_bar(
            evaluator,
            board,
            WHITE,
        )

        self.assertIn("YiXin评价不可用", bar)
        self.assertIn("黑 X   -- ", bar)


if __name__ == "__main__":
    unittest.main()
