import unittest

from arena import create_ai, engine_display_name
from engine.ai import RandomAI, ScoringAI, TacticalAI
from engine.arena_settings import AISelection
from engine.board import BLACK, WHITE
from engine.search import SearchAI


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


if __name__ == "__main__":
    unittest.main()
