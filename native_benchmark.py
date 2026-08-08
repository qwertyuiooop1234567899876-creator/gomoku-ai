from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import time

from engine.board import BLACK, WHITE, Board
from engine.game import parse_move
from engine.native_core import native_core_status
from engine.proof_search import ProofBudget, ProofSearch, ProofTable
from engine.threats import ThreatAnalyzer


CASES = (
    (
        "black15_L10",
        "H8 H9 G7 I9 G9 I7 I8 G8 I10 J8 H6 K7 H10 K9".split(),
        "L10",
    ),
    (
        "black51_H4",
        """H8 I7 G7 I9 I8 J8 H6 H10 G11 I5 H7 H5 J9 G6 F7 G5 F5 F6
        E7 D7 K10 L11 F8 E8 F9 E9 F10 F11 G8 I6 K7 K8 J7 E6 E10 D11
        G10 G9 K6 H9 K5 D10 D8 D12 D13 I4 I3 J4 D6 K4""".split(),
        "H4",
    ),
)


def run_worker(repeat: int) -> dict[str, object]:
    results: dict[str, object] = {
        "native": native_core_status(),
        "cases": {},
    }
    for name, prefix, candidate in CASES:
        samples: list[float] = []
        nodes: list[int] = []
        state = ""
        for _ in range(repeat):
            board = Board()
            for index, coordinate in enumerate(prefix):
                board.place(
                    *parse_move(coordinate, board.size),
                    BLACK if index % 2 == 0 else WHITE,
                )
            proof = ProofSearch(
                budget=ProofBudget(
                    max_nodes=20_000,
                    max_attacker_moves=10,
                    max_quiet_frontiers=16,
                    max_quiet_attacker_moves=1,
                    use_vcf_oracle=True,
                ),
                analyzer=ThreatAnalyzer(
                    candidate_limit=16,
                    frontier_scan_limit=24,
                ),
                table=ProofTable(),
                clock=time.perf_counter,
            ).search_after_move(
                board,
                move=parse_move(candidate, board.size),
                mover=BLACK,
                attacker=WHITE,
                side_to_move=WHITE,
            )
            samples.append(proof.elapsed_seconds)
            nodes.append(proof.nodes)
            state = proof.state.value
        results["cases"][name] = {
            "state": state,
            "median_seconds": median(samples),
            "median_nodes": int(median(nodes)),
        }
    return results


def child_result(*, native: bool, repeat: int) -> dict[str, object]:
    environment = os.environ.copy()
    if native:
        environment.pop("GOMOKU_NATIVE_DISABLE", None)
    else:
        environment["GOMOKU_NATIVE_DISABLE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", "--repeat", str(repeat)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description="V0.14.0 NativeCore Proof基准")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat必须大于0")
    if args.worker:
        print(json.dumps(run_worker(args.repeat), ensure_ascii=False))
        return 0

    python_result = child_result(native=False, repeat=args.repeat)
    native_result = child_result(native=True, repeat=args.repeat)
    if not native_result["native"]["available"]:
        print("NativeCore未加载，请先运行：python build_native.py --clean")
        return 2

    print(f"Gomoku NativeCore benchmark | repeat={args.repeat}")
    print(f"{'Case':<18} {'Python':>10} {'Native':>10} {'Speedup':>9} {'State':>12}")
    print("-" * 64)
    for name, _, _ in CASES:
        python_case = python_result["cases"][name]
        native_case = native_result["cases"][name]
        python_seconds = python_case["median_seconds"]
        native_seconds = native_case["median_seconds"]
        speedup = python_seconds / native_seconds
        print(
            f"{name:<18} {python_seconds:>9.3f}s {native_seconds:>9.3f}s "
            f"{speedup:>8.2f}x {native_case['state']:>12}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
