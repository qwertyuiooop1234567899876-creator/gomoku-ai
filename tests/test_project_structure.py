from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import unittest

from engine.search import SearchAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestPackageEntrypoints(unittest.TestCase):
    def test_root_contains_no_python_compatibility_wrappers(self) -> None:
        self.assertEqual([], sorted(PROJECT_ROOT.glob("*.py")))

    def test_public_package_modules_import_directly(self) -> None:
        module_names = (
            "app.arena",
            "app.cli",
            "app.desktop_ui",
            "app.ui_common",
            "app.web_ui",
            "tools.build_native",
            "tools.cvc_analysis",
            "tools.cvc_workflow",
            "tools.manual_scenarios",
            "tools.native_benchmark",
            "tools.native_search_baseline",
            "tools.git_submit",
            "tools.search_benchmark",
            "tools.yixin_smoke_test",
        )
        for module_name in module_names:
            with self.subTest(module=module_name):
                self.assertEqual(
                    module_name,
                    importlib.import_module(module_name).__name__,
                )

    def test_batch_launchers_use_package_modules(self) -> None:
        launchers = {
            "run_game.bat": ("-m app.desktop_ui", "-m app.web_ui"),
            "run_game_web.bat": ("-m app.web_ui",),
            "run_arena.bat": ("-m app.arena",),
            "run_cvc_analysis.bat": ("-m tools.cvc_analysis",),
            "run_cvc_workflow.bat": ("-m tools.cvc_workflow",),
            "run_search_benchmark.bat": (
                "-m tools.search_benchmark",
            ),
            "run_git_submit.bat": ("-m tools.git_submit",),
            "build_native.bat": ("-m tools.build_native",),
            "run_yixin_smoke_test.bat": (
                "-m tools.yixin_smoke_test",
            ),
        }
        for filename, module_commands in launchers.items():
            with self.subTest(launcher=filename):
                content = (PROJECT_ROOT / filename).read_text(
                    encoding="utf-8"
                )
                for module_command in module_commands:
                    self.assertIn(module_command, content)

    def test_package_paths_resolve_project_resources(self) -> None:
        web_ui = importlib.import_module("app.web_ui")
        build_native = importlib.import_module("tools.build_native")
        cvc_workflow = importlib.import_module("tools.cvc_workflow")

        self.assertEqual(PROJECT_ROOT / "ui" / "gomoku.html", web_ui.STATIC_FILE)
        self.assertEqual(PROJECT_ROOT, build_native.ROOT)
        self.assertEqual(PROJECT_ROOT, cvc_workflow.PROJECT_ROOT)
        self.assertEqual("tools.cvc_analysis", cvc_workflow.ANALYSIS_MODULE)


class TestSearchModuleBoundaries(unittest.TestCase):
    def test_diagnostics_serialization_is_a_small_delegate(self) -> None:
        source = inspect.getsource(SearchAI._save_search_analysis)

        self.assertLessEqual(len(source.splitlines()), 40)
        self.assertIn("build_search_analysis", source)
        self.assertNotIn("DecisionAnalysis(", source)


if __name__ == "__main__":
    unittest.main()
