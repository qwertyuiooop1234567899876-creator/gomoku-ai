from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import unittest

from engine.search import SearchAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestCompatibilityEntrypoints(unittest.TestCase):
    def test_root_modules_forward_to_package_implementations(self) -> None:
        module_pairs = (
            ("arena", "app.arena"),
            ("main", "app.cli"),
            ("gomoku_ui_common", "app.ui_common"),
            ("cvc_analysis", "tools.cvc_analysis"),
            ("cvc_workflow", "tools.cvc_workflow"),
            ("build_native", "tools.build_native"),
            ("manual_scenarios", "tools.manual_scenarios"),
            ("search_benchmark", "tools.search_benchmark"),
        )
        for compatibility_name, implementation_name in module_pairs:
            with self.subTest(module=compatibility_name):
                compatibility = importlib.import_module(compatibility_name)
                implementation = importlib.import_module(implementation_name)
                self.assertIs(compatibility, implementation)

    def test_package_paths_resolve_project_resources(self) -> None:
        web_ui = importlib.import_module("app.web_ui")
        build_native = importlib.import_module("tools.build_native")
        cvc_workflow = importlib.import_module("tools.cvc_workflow")

        self.assertEqual(PROJECT_ROOT / "ui" / "gomoku.html", web_ui.STATIC_FILE)
        self.assertEqual(PROJECT_ROOT, build_native.ROOT)
        self.assertEqual(PROJECT_ROOT, cvc_workflow.PROJECT_ROOT)
        self.assertEqual(
            PROJECT_ROOT / "cvc_analysis.py",
            cvc_workflow.ANALYSIS_PROGRAM,
        )


class TestSearchModuleBoundaries(unittest.TestCase):
    def test_diagnostics_serialization_is_a_small_delegate(self) -> None:
        source = inspect.getsource(SearchAI._save_search_analysis)

        self.assertLessEqual(len(source.splitlines()), 40)
        self.assertIn("build_search_analysis", source)
        self.assertNotIn("DecisionAnalysis(", source)


if __name__ == "__main__":
    unittest.main()
