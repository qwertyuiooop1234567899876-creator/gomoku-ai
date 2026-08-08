from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from engine import evaluator
from engine.board import BLACK, WHITE, Board
from engine.game import format_move, parse_move
from engine.search import SearchAI
from engine.version import ENGINE_VERSION


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    coordinates: tuple[str, ...]
    player: int
    max_depth: int
    time_limit_seconds: float | None
    expected_move: str


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    elapsed_seconds: float
    selected_move: str
    completed_depth: int
    nodes: int
    nps: int
    proof_nodes: int
    vcf_nodes: int
    root_vcf_nodes: int
    defense_vct_nodes: int
    stop_reason: str | None


CASES = (
    BenchmarkCase(
        name="counterattack_j9",
        coordinates=(
            "H8", "H7", "I7", "G9", "J8",
            "I8", "K9", "H6", "J10",
        ),
        player=WHITE,
        max_depth=2,
        time_limit_seconds=None,
        expected_move="J9",
    ),
    BenchmarkCase(
        name="g9_multi_threat",
        coordinates=(
            "H8", "H7", "G7", "I9", "F8",
            "E8", "E9", "H6", "F10",
        ),
        player=WHITE,
        max_depth=2,
        time_limit_seconds=None,
        expected_move="G9",
    ),
    BenchmarkCase(
        name="h11_defense_vct",
        coordinates=(
            "H8", "I7", "I6", "H7", "G7", "F6",
            "J7", "K8", "J8", "J9", "I8", "I10",
        ),
        player=BLACK,
        max_depth=3,
        time_limit_seconds=None,
        expected_move="H11",
    ),
    BenchmarkCase(
        name="i4_proof_risk",
        coordinates=(
            "H8", "I7", "I6", "H7", "G7", "G6",
            "I8", "J8", "H6", "F8", "K3", "K6",
            "F6", "E5", "J4", "I5", "J7", "H5",
        ),
        player=BLACK,
        max_depth=2,
        time_limit_seconds=4.0,
        expected_move="J5",
    ),
)


def build_board(coordinates: tuple[str, ...]) -> Board:
    board = Board()
    player = BLACK
    for coordinate in coordinates:
        board.place(*parse_move(coordinate, board.size), player)
        player = WHITE if player == BLACK else BLACK
    return board


def clear_evaluator_cache() -> None:
    clear = getattr(evaluator._score_line, "cache_clear", None)
    if clear is not None:
        clear()


def run_once(case: BenchmarkCase) -> BenchmarkRun:
    clear_evaluator_cache()
    board = build_board(case.coordinates)
    before = (
        tuple(tuple(row) for row in board.grid),
        tuple(board.move_history),
        board.zobrist_hash,
        board.empty_count,
    )
    ai = SearchAI(
        player=case.player,
        max_depth=case.max_depth,
        time_limit_seconds=case.time_limit_seconds,
        diagnostics=True,
    )

    started_at = time.perf_counter()
    selected = ai.choose_move(board)
    elapsed = time.perf_counter() - started_at
    analysis = ai.last_analysis
    if analysis is None:
        raise RuntimeError(f"{case.name}: SearchAI 没有生成诊断数据。")

    after = (
        tuple(tuple(row) for row in board.grid),
        tuple(board.move_history),
        board.zobrist_hash,
        board.empty_count,
    )
    if after != before:
        raise RuntimeError(f"{case.name}: 搜索后棋盘状态发生变化。")

    selected_text = format_move(*selected)
    if selected_text != case.expected_move:
        raise RuntimeError(
            f"{case.name}: 预期 {case.expected_move}，实际 {selected_text}。"
        )

    return BenchmarkRun(
        elapsed_seconds=elapsed,
        selected_move=selected_text,
        completed_depth=analysis.search_depth,
        nodes=analysis.nodes,
        nps=analysis.nps,
        proof_nodes=analysis.proof_nodes,
        vcf_nodes=analysis.vcf_nodes,
        root_vcf_nodes=analysis.root_vcf_nodes,
        defense_vct_nodes=analysis.defense_vct_nodes,
        stop_reason=analysis.stop_reason,
    )


def summarize_case(
    case: BenchmarkCase,
    runs: list[BenchmarkRun],
) -> dict[str, object]:
    return {
        "name": case.name,
        "expected_move": case.expected_move,
        "max_depth": case.max_depth,
        "time_limit_seconds": case.time_limit_seconds,
        "median_elapsed_seconds": statistics.median(
            run.elapsed_seconds for run in runs
        ),
        "min_elapsed_seconds": min(run.elapsed_seconds for run in runs),
        "max_elapsed_seconds": max(run.elapsed_seconds for run in runs),
        "median_nps": int(statistics.median(run.nps for run in runs)),
        "runs": [asdict(run) for run in runs],
    }


def print_report(report: dict[str, object]) -> None:
    print(
        f"SearchAI benchmark | V{report['engine_version']} | "
        f"repeat={report['repeat']}"
    )
    print(
        f"{'Case':22} {'Move':>5} {'Median':>10} "
        f"{'Depth':>7} {'Nodes':>9} {'NPS':>9}"
    )
    print("-" * 68)
    for case in report["cases"]:
        runs = case["runs"]
        representative = min(
            runs,
            key=lambda run: abs(
                run["elapsed_seconds"]
                - case["median_elapsed_seconds"]
            ),
        )
        print(
            f"{case['name']:22} "
            f"{representative['selected_move']:>5} "
            f"{case['median_elapsed_seconds']:>9.3f}s "
            f"{representative['completed_depth']:>7} "
            f"{representative['nodes']:>9} "
            f"{case['median_nps']:>9}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行 SearchAI 固定局面性能回归。",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="每个局面的重复次数（默认 3）。",
    )
    parser.add_argument(
        "--case",
        choices=[case.name for case in CASES],
        help="只运行一个局面。",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="可选：把完整结果写入 JSON。",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat 必须大于 0。")

    selected_cases = [
        case
        for case in CASES
        if args.case is None or case.name == args.case
    ]
    summaries = []
    for case in selected_cases:
        runs = [run_once(case) for _ in range(args.repeat)]
        summaries.append(summarize_case(case, runs))

    report = {
        "engine_version": ENGINE_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "repeat": args.repeat,
        "cases": summaries,
    }
    print_report(report)
    if args.json is not None:
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON: {args.json}")


if __name__ == "__main__":
    main()
