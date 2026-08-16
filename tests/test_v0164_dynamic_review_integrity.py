from __future__ import annotations

import unittest

from engine import root_review
from engine.search_types import RootResult, SearchConfig


class TestV0164DynamicReviewIntegrity(unittest.TestCase):
    def test_required_sources_cannot_restore_unsearched_move(self) -> None:
        leader = (1, 1)
        first = (1, 2)
        second = (1, 3)
        filtered_offensive = (9, 9)
        result = RootResult(
            leader,
            300,
            (leader,),
            (
                (leader, 300),
                (first, 200),
                (second, 100),
            ),
        )
        config = SearchConfig(
            root_dynamic_review_candidate_limit=4,
            root_dynamic_review_finalist_limit=4,
        )

        pool = root_review.review_pool(
            config,
            result,
            (),
            quiet_moves=(),
            offensive_moves=(filtered_offensive,),
        )

        searched = {move for move, _score in result.ranked_moves}
        self.assertTrue(set(pool).issubset(searched))
        self.assertNotIn(filtered_offensive, pool)


if __name__ == "__main__":
    unittest.main()
