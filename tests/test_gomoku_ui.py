from __future__ import annotations

import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

from engine.board import BLACK, WHITE, Board
from engine.game import parse_move
from gomoku_ui import (
    BoardGeometry,
    ClickConfirmation,
    GomokuApp,
    normalized_ai_selection,
)
from gomoku_web_ui import WebGameController
from gomoku_ui_common import clone_board


class _BlockingAI:
    def __init__(self, release: Event) -> None:
        self.release = release
        self.last_analysis = {"source": "ui-test"}

    def choose_move(self, board):
        self.release.wait(timeout=2)
        return parse_move("H8")

    def close(self) -> None:
        return None


class TestBoardGeometry(unittest.TestCase):
    def test_intersections_round_trip_through_canvas_coordinates(self) -> None:
        geometry = BoardGeometry(size=15, width=720, height=680)

        for coordinate in ("A1", "H8", "O15", "C12"):
            move = parse_move(coordinate)
            self.assertEqual(move, geometry.nearest_move(*geometry.point(*move)))

    def test_click_far_outside_board_is_rejected(self) -> None:
        geometry = BoardGeometry(size=15, width=720, height=680)

        self.assertIsNone(geometry.nearest_move(0, 0))
        self.assertIsNone(geometry.nearest_move(719, 679))


class TestClickConfirmation(unittest.TestCase):
    def test_same_intersection_requires_two_clicks(self) -> None:
        confirmation = ClickConfirmation()
        move = parse_move("H8")

        self.assertFalse(confirmation.register(move))
        self.assertEqual(move, confirmation.pending)
        self.assertTrue(confirmation.register(move))
        self.assertIsNone(confirmation.pending)

    def test_different_intersection_replaces_preview(self) -> None:
        confirmation = ClickConfirmation()
        first = parse_move("H8")
        second = parse_move("I8")

        self.assertFalse(confirmation.register(first))
        self.assertFalse(confirmation.register(second))
        self.assertEqual(second, confirmation.pending)


class TestUIAISelection(unittest.TestCase):
    def test_slider_values_are_snapped_to_supported_ranges(self) -> None:
        selection = normalized_ai_selection("search", 6.6, 3.24)

        self.assertEqual(7, selection.max_depth)
        self.assertEqual(3.0, selection.time_limit_seconds)

    def test_slider_values_are_clamped(self) -> None:
        selection = normalized_ai_selection("yixin", 99, 0.01)

        self.assertEqual(8, selection.max_depth)
        self.assertEqual(0.5, selection.time_limit_seconds)

    def test_search_board_snapshot_is_independent(self) -> None:
        board = Board()
        board.place(*parse_move("H8"), BLACK)

        snapshot = clone_board(board)
        snapshot.place(*parse_move("I8"), WHITE)

        self.assertEqual(1, len(board.move_history))
        self.assertEqual(2, len(snapshot.move_history))


class TestWebGameController(unittest.TestCase):
    def test_second_click_commits_human_move_and_starts_ai(self) -> None:
        release = Event()
        with patch("gomoku_web_ui.create_ai", return_value=_BlockingAI(release)):
            controller = WebGameController()
        try:
            first = controller.select(*parse_move("A1"))
            self.assertEqual([], first["moves"])
            self.assertEqual({"row": 0, "column": 0}, first["pending"])

            second = controller.select(*parse_move("A1"))
            self.assertEqual(1, len(second["moves"]))
            self.assertEqual(BLACK, second["moves"][0]["player"])
            self.assertTrue(second["ai_thinking"])
        finally:
            release.set()
            controller.close()

    def test_browser_asset_contains_requested_controls(self) -> None:
        html = (
            Path(__file__).parents[1] / "ui" / "gomoku.html"
        ).read_text(encoding="utf-8")

        self.assertIn('id="board"', html)
        self.assertIn('id="engine"', html)
        self.assertIn('id="depth"', html)
        self.assertIn('id="time"', html)
        self.assertIn('id="confirm"', html)

    def test_browser_backend_does_not_import_tk_desktop_ui(self) -> None:
        project_root = Path(__file__).parents[1]
        backend = (project_root / "gomoku_web_ui.py").read_text(encoding="utf-8")
        launcher = (project_root / "run_game_web.bat").read_text(encoding="utf-8")

        self.assertNotIn("import tkinter", backend)
        self.assertNotIn("from gomoku_ui import", backend)
        self.assertIn("gomoku_web_ui.py", launcher)


class TestTkDesktopWindow(unittest.TestCase):
    def test_window_builds_with_requested_controls(self) -> None:
        try:
            import tkinter as tk
        except ImportError as error:
            self.skipTest(f"Tk is unavailable: {error}")
            return
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display is unavailable: {error}")
            return
        root.withdraw()
        fake_ai = _BlockingAI(Event())
        app = None
        try:
            with patch("gomoku_ui.create_ai", return_value=fake_ai):
                app = GomokuApp(root)
            root.update_idletasks()

            self.assertEqual("readonly", str(app.engine_combo.cget("state")))
            self.assertIn("disabled", app.save_button.state())
            self.assertEqual("", app.progress.winfo_manager())
            self.assertEqual(15, app.board.size)
        finally:
            if app is not None:
                app._closed = True
                app._close_ai()
            root.destroy()


if __name__ == "__main__":
    unittest.main()
