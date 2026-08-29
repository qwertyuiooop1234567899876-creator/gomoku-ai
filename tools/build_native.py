from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = ROOT / "native"
OUTPUT_DIRECTORY = ROOT / "native" / "bin"


def source_paths() -> tuple[Path, ...]:
    """Return every native translation unit in deterministic order."""
    sources = tuple(sorted(SOURCE_DIRECTORY.glob("*.cpp")))
    if not sources:
        raise RuntimeError(
            f"NativeCore 没有可编译的C++源文件：{SOURCE_DIRECTORY}"
        )
    return sources


def output_path() -> Path:
    if sys.platform == "win32":
        return OUTPUT_DIRECTORY / "gomoku_native.dll"
    if sys.platform == "darwin":
        return OUTPUT_DIRECTORY / "gomoku_native.dylib"
    return OUTPUT_DIRECTORY / "gomoku_native.so"


def compiler_command(*, output: Path | None = None) -> list[str] | str:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    selected_output = output or output_path()
    sources = source_paths()
    if sys.platform == "win32" and shutil.which("cl"):
        return [
            "cl",
            "/nologo",
            "/std:c++17",
            "/O2",
            "/EHsc",
            "/LD",
            f"/Fo:{OUTPUT_DIRECTORY}\\",
            *(str(source) for source in sources),
            "/link",
            f"/OUT:{selected_output}",
            f"/IMPLIB:{OUTPUT_DIRECTORY / 'gomoku_native.lib'}",
        ]

    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles(x86)")
        vswhere = (
            None
            if not program_files
            else Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        )
        if vswhere is not None and vswhere.is_file():
            query = subprocess.run(
                [
                    str(vswhere),
                    "-latest",
                    "-prerelease",
                    "-products", "*",
                    "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property", "installationPath",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            installation = query.stdout.strip()
            vcvars = Path(installation) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if installation and vcvars.is_file():
                quoted_sources = " ".join(
                    f'"{source}"' for source in sources
                )
                compile_text = (
                    f'call "{vcvars}" >nul && cl /nologo /std:c++17 /O2 /EHsc /LD '
                    f'/Fo:{OUTPUT_DIRECTORY}\\ {quoted_sources} /link '
                    f'/OUT:"{selected_output}" /IMPLIB:"{OUTPUT_DIRECTORY / "gomoku_native.lib"}"'
                )
                # ``/s`` rewrites the outer quote pair and breaks ``call``
                # when a prerelease Visual Studio path contains spaces.
                return compile_text

    compiler = shutil.which(os.environ.get("CXX", "")) if os.environ.get("CXX") else None
    compiler = compiler or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise RuntimeError(
            "未找到C++编译器。Windows请安装Visual Studio Build Tools的C++组件，"
            "或安装MinGW-w64并把g++加入PATH。"
        )

    command = [compiler, "-std=c++17", "-O3", "-DNDEBUG", "-shared"]
    if sys.platform != "win32":
        command.append("-fPIC")
    else:
        command.extend(["-static-libgcc", "-static-libstdc++"])
    command.extend(str(source) for source in sources)
    command.extend(["-o", str(selected_output)])
    return command


def build_native_runtime() -> Path:
    """Build beside the live library and replace it only after success."""
    output = output_path()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    staged_output = output.with_name(
        f"{output.stem}.staging{output.suffix}"
    )
    staged_output.unlink(missing_ok=True)
    command = compiler_command(output=staged_output)
    print(
        "NativeCore build:",
        command if isinstance(command, str) else " ".join(command),
    )
    try:
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            shell=isinstance(command, str),
        )
        if not staged_output.is_file():
            raise RuntimeError(
                f"编译器成功退出但没有生成原生库：{staged_output}"
            )
        staged_output.replace(output)
    finally:
        staged_output.unlink(missing_ok=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="编译Gomoku NativeCore")
    parser.add_argument("--clean", action="store_true", help="先删除旧原生库")
    args = parser.parse_args()

    if args.clean:
        print(
            "NativeCore clean build: live runtime is kept until "
            "replacement succeeds."
        )
    output = build_native_runtime()
    print(f"NativeCore ready: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
