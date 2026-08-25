from __future__ import annotations

import contextlib
import io
import unittest

from tools.search_benchmark import print_report


class SearchBenchmarkReportTests(unittest.TestCase):
    def test_report_separates_main_search_from_other_work(self) -> None:
        report = {
            "engine_version": "test",
            "evaluation_profile": "tempo-v1",
            "repeat": 1,
            "cases": [
                {
                    "name": "sample",
                    "median_elapsed_seconds": 0.5,
                    "median_nps": 20,
                    "median_main_search_elapsed_seconds": 0.2,
                    "median_other_elapsed_seconds": 0.3,
                    "median_main_search_nps": 50,
                    "runs": [
                        {
                            "elapsed_seconds": 0.5,
                            "selected_move": "H8",
                            "completed_depth": 2,
                            "nodes": 10,
                        }
                    ],
                }
            ],
        }

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_report(report)

        text = output.getvalue()
        self.assertIn("Whole", text)
        self.assertIn("Main", text)
        self.assertIn("Other", text)
        self.assertIn("PVSNodes", text)
        self.assertIn("MainNPS", text)
        self.assertIn("Whole=full move", text)
        self.assertIn("MainNPS=PVSNodes/Main", text)
        self.assertIn("not whole-move throughput", text)
        self.assertNotIn("Eff.NPS", text)


if __name__ == "__main__":
    unittest.main()
