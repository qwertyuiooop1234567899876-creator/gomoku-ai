from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_native


class TestAtomicNativeBuild(unittest.TestCase):
    def test_failed_build_keeps_existing_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory)
            live = output_directory / "gomoku_native.dll"
            live.write_bytes(b"working-runtime")

            with (
                patch.object(build_native, "OUTPUT_DIRECTORY", output_directory),
                patch.object(build_native, "output_path", return_value=live),
                patch.object(
                    build_native,
                    "compiler_command",
                    return_value=["fake-compiler"],
                ),
                patch.object(
                    build_native.subprocess,
                    "run",
                    side_effect=subprocess.CalledProcessError(
                        1,
                        "fake-compiler",
                    ),
                ),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    build_native.build_native_runtime()

            self.assertEqual(b"working-runtime", live.read_bytes())
            self.assertFalse(
                (output_directory / "gomoku_native.staging.dll").exists()
            )

    def test_successful_build_atomically_replaces_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory)
            live = output_directory / "gomoku_native.dll"
            live.write_bytes(b"old-runtime")
            staged = output_directory / "gomoku_native.staging.dll"

            def fake_run(*_args, **_kwargs) -> None:
                staged.write_bytes(b"new-runtime")

            with (
                patch.object(build_native, "OUTPUT_DIRECTORY", output_directory),
                patch.object(build_native, "output_path", return_value=live),
                patch.object(
                    build_native,
                    "compiler_command",
                    return_value=["fake-compiler"],
                ),
                patch.object(
                    build_native.subprocess,
                    "run",
                    side_effect=fake_run,
                ),
            ):
                result = build_native.build_native_runtime()

            self.assertEqual(live, result)
            self.assertEqual(b"new-runtime", live.read_bytes())
            self.assertFalse(staged.exists())


if __name__ == "__main__":
    unittest.main()
