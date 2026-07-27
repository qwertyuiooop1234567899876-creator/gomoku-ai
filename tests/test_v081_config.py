import unittest

from engine.board import BLACK, WHITE
from engine.search import SearchAI


class TestV081TranspositionConfiguration(unittest.TestCase):
    def test_default_transposition_capacity_is_fixed_at_100000(self) -> None:
        ai = SearchAI(player=BLACK)
        self.assertEqual(100_000, ai.config.transposition_max_entries)

    def test_search_ai_instances_keep_independent_tables(self) -> None:
        black_ai = SearchAI(player=BLACK)
        white_ai = SearchAI(player=WHITE)

        self.assertIsNot(
            black_ai._transposition_table,
            white_ai._transposition_table,
        )

        black_ai._transposition_table[123] = object()  # type: ignore[assignment]
        self.assertIn(123, black_ai._transposition_table)
        self.assertNotIn(123, white_ai._transposition_table)


if __name__ == "__main__":
    unittest.main()
