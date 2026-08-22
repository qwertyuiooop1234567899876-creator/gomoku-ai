import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.settings import (
    SearchSettings,
    load_search_settings,
    save_search_settings,
)
from app.cli import choose_search_settings, create_computer


class TestSearchSettings(unittest.TestCase):
    def test_missing_file_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "settings.json"

            settings = load_search_settings(path)

            self.assertEqual(3, settings.max_depth)
            self.assertEqual(2.0, settings.time_limit_seconds)

    def test_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "settings.json"
            expected = SearchSettings(
                max_depth=4,
                time_limit_seconds=5.0,
            )

            save_search_settings(expected, path)
            actual = load_search_settings(path)

            self.assertEqual(expected, actual)

    def test_broken_file_falls_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "settings.json"
            path.write_text("{broken", encoding="utf-8")

            settings = load_search_settings(path)

            self.assertEqual(SearchSettings(), settings)

    def test_blank_input_keeps_saved_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "settings.json"
            save_search_settings(
                SearchSettings(
                    max_depth=4,
                    time_limit_seconds=3.5,
                ),
                path,
            )

            with patch("builtins.input", side_effect=["", ""]):
                with patch("builtins.print"):
                    selected = choose_search_settings(str(path))

            self.assertEqual(4, selected.max_depth)
            self.assertEqual(3.5, selected.time_limit_seconds)

    def test_new_values_are_saved_and_used_by_ai(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "settings.json"

            with patch("builtins.input", side_effect=["5", "6"]):
                with patch("builtins.print"):
                    selected = choose_search_settings(str(path))

            payload = json.loads(path.read_text(encoding="utf-8"))
            ai = create_computer(1, selected)

            self.assertEqual(5, payload["max_depth"])
            self.assertEqual(6.0, payload["time_limit_seconds"])
            self.assertEqual(5, ai.config.max_depth)
            self.assertEqual(6.0, ai.config.time_limit_seconds)

    @patch("builtins.input", side_effect=["0", "3", "61", "2.5"])
    @patch("builtins.print")
    def test_invalid_values_reprompt(
        self,
        _mock_print,
        _mock_input,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "settings.json"
            selected = choose_search_settings(str(path))

        self.assertEqual(3, selected.max_depth)
        self.assertEqual(2.5, selected.time_limit_seconds)


if __name__ == "__main__":
    unittest.main()
