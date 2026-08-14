import json
import tempfile
import unittest
from pathlib import Path

from engine.arena_settings import (
    AISelection,
    ArenaSettings,
    load_arena_settings,
    save_arena_settings,
)


class TestArenaSettings(unittest.TestCase):
    def test_round_trip_preserves_independent_search_settings(self) -> None:
        settings = ArenaSettings(
            black=AISelection("search", 5, 8.0),
            white=AISelection("search", 2, 0.5),
            watch=False,
            show_evaluation=True,
            delay_seconds=0.25,
            save_record=False,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arena_settings.json"
            save_arena_settings(settings, path)
            loaded = load_arena_settings(path)

        self.assertEqual(settings, loaded)
        self.assertEqual(5, loaded.black.max_depth)
        self.assertEqual(2, loaded.white.max_depth)
        self.assertEqual(8.0, loaded.black.time_limit_seconds)
        self.assertEqual(0.5, loaded.white.time_limit_seconds)

    def test_missing_file_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            loaded = load_arena_settings(path)

        self.assertEqual("search", loaded.black.engine_name)
        self.assertEqual("scoring", loaded.white.engine_name)

    def test_corrupt_file_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text("{not-json", encoding="utf-8")
            loaded = load_arena_settings(path)

        self.assertEqual(ArenaSettings(), loaded)

    def test_string_booleans_do_not_silently_enable_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-types.json"
            path.write_text(
                json.dumps({"show_evaluation": "false"}),
                encoding="utf-8",
            )
            loaded = load_arena_settings(path)

        self.assertEqual(ArenaSettings(), loaded)
        self.assertFalse(loaded.show_evaluation)

    def test_saved_file_is_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arena_settings.json"
            save_arena_settings(ArenaSettings(), path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertIn("black", payload)
        self.assertIn("white", payload)

    def test_switching_engine_keeps_search_parameters(self) -> None:
        selection = AISelection("search", 6, 12.0)
        changed = selection.with_engine("tactical")

        self.assertEqual("tactical", changed.engine_name)
        self.assertEqual(6, changed.max_depth)
        self.assertEqual(12.0, changed.time_limit_seconds)

    def test_yixin_is_a_valid_timed_engine(self) -> None:
        selection = AISelection("yixin", 6, 10.0)

        self.assertFalse(selection.uses_search)
        self.assertTrue(selection.uses_time_limit)


if __name__ == "__main__":
    unittest.main()
