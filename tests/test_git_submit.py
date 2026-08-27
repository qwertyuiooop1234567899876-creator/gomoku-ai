from __future__ import annotations

import unittest

from tools.git_submit import exclusion_reason, parse_changes


class TestGitSubmitHelpers(unittest.TestCase):
    def test_parse_changes_preserves_status_and_path(self) -> None:
        changes = parse_changes(" M engine/search.py\n?? tools/git_submit.py\n")

        self.assertEqual("M", changes[0].worktree_status)
        self.assertEqual("engine/search.py", changes[0].path)
        self.assertEqual("?", changes[1].index_status)
        self.assertEqual("tools/git_submit.py", changes[1].path)

    def test_records_and_transient_files_are_excluded(self) -> None:
        self.assertIsNotNone(exclusion_reason("records/game.json"))
        self.assertIsNotNone(exclusion_reason("native/bin/core.dll"))
        self.assertIsNotNone(exclusion_reason("search-benchmark-results.json"))
        self.assertIsNotNone(exclusion_reason("engine/__pycache__/search.pyc"))
        self.assertIsNone(exclusion_reason("engine/search.py"))
        self.assertIsNone(exclusion_reason("structure.txt"))


if __name__ == "__main__":
    unittest.main()
