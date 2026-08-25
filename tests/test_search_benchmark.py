from __future__ import annotations

import contextlib
import io
import unittest

from tools.search_benchmark import print_report


class SearchBenchmarkReportTests(unittest.TestCase):
    def test_report_labels_effective_nps_as_whole_move_metric(self) -> None:
        report = {
            "engine_version": "test",
            "evaluation_profile": "tempo-v1",
            "repeat": 1,
            "cases": [
                {
                    "name": "sample",
                    "median_elapsed_seconds": 0.5,
                    "median_nps": 20,
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
        self.assertIn("PVS Nodes", text)
        self.assertIn("Eff.NPS*", text)
        self.assertIn("PVS nodes / whole choose_move time", text)
        self.assertIn("not PVS-core NPS", text)
        self.assertNotIn(" Median ", text)


if __name__ == "__main__":
    unittest.main()
