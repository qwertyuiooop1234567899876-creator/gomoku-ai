from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "native" / "gomoku_native.cpp"
OUTPUT_DIRECTORY = ROOT / "native" / "bin"


def output_path() -> Path:
    if sys.platform == "win32":
        return OUTPUT_DIRECTORY / "gomoku_native.dll"
    if sys.platform == "darwin":
        return OUTPUT_DIRECTORY / "gomoku_native.dylib"
    return OUTPUT_DIRECTORY / "gomoku_native.so"


def compiler_command() -> list[str]:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output = output_path()
    if sys.platform == "win32" and shutil.which("cl"):
        return [
            "cl",
            "/nologo",
            "/std:c++17",
            "/O2",
            "/EHsc",
            "/LD",
            f"/Fo:{OUTPUT_DIRECTORY / 'gomoku_native.obj'}",
            str(SOURCE),
            "/link",
            f"/OUT:{output}",
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
                compile_text = (
                    f'call "{vcvars}" >nul && cl /nologo /std:c++17 /O2 /EHsc /LD '
                    f'/Fo:"{OUTPUT_DIRECTORY / "gomoku_native.obj"}" "{SOURCE}" /link '
                    f'/OUT:"{output}" /IMPLIB:"{OUTPUT_DIRECTORY / "gomoku_native.lib"}"'
                )
                return ["cmd", "/d", "/s", "/c", compile_text]

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
    command.extend([str(SOURCE), "-o", str(output)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="编译Gomoku NativeCore")
    parser.add_argument("--clean", action="store_true", help="先删除旧原生库")
    args = parser.parse_args()

    output = output_path()
    if args.clean and output.exists():
        output.unlink()
    command = compiler_command()
    print("NativeCore build:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"NativeCore ready: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
