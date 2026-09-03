from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

from engine.game import parse_move
from engine.native_core import native_core
from engine.threats import DefenseSet, ThreatAnalyzer
from tools.vct_reference import VCTReferenceCase, build_board, load_case


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = (
    ROOT / "tests" / "positions" / "v0175_reverse_move10_vct.json",
    ROOT / "tests" / "positions" / "v0175_selfplay_move24_vct.json",
    ROOT / "tests" / "positions" / "v0175_yixin_move21_vct.json",
)


@dataclass(frozen=True, slots=True)
class DefenseClassificationBenchmark:
    case_name: str
    coordinate: str
    continuation_count: int
    legal_reply_count: int
    equivalent: bool
    python_seconds: float
    native_seconds: float
    speedup: float


@dataclass(frozen=True, slots=True)
class DefenseClassificationRun:
    repeats: int
    minimum_speedup: float
    candidates: tuple[DefenseClassificationBenchmark, ...]
    nontrivial_minimum_speedup: float | None
    production_eligible: bool


def board_state(board) -> tuple[object, ...]:
    return (
        tuple(tuple(row) for row in board.grid),
        tuple(board.move_history),
        board.zobrist_hash,
        board.empty_count,
    )


def _native_signature(result) -> tuple[object, ...]:
    if result is None:
        raise RuntimeError("Native defense classification is unavailable")
    return result.signature


def benchmark_case(
    case: VCTReferenceCase,
    *,
    repeats: int,
) -> tuple[DefenseClassificationBenchmark, ...]:
    if repeats < 1:
        raise ValueError("repeats 必须大于 0")
    board = build_board(case)
    analyzer = ThreatAnalyzer(candidate_limit=24, frontier_scan_limit=48)
    initial_state = board_state(board)
    benchmarks: list[DefenseClassificationBenchmark] = []
    for coordinate in case.candidates:
        move = parse_move(coordinate, board.size)
        threat = analyzer.describe_move(board, move, case.player)
        if board_state(board) != initial_state:
            raise RuntimeError(f"Python threat analysis polluted {coordinate}")

        board.place(*move, case.player)
        placed_state = board_state(board)
        try:
            python_result = analyzer._classify_defenses(
                board,
                attacker=case.player,
                continuations=threat.continuations,
                counter_wins=threat.counter_wins,
                stop_requested=None,
            )
            native_result = native_core.classify_defenses(
                board,
                case.player,
                threat.continuations,
                threat.counter_wins,
            )
            if board_state(board) != placed_state:
                raise RuntimeError(
                    f"Defense classification polluted {coordinate}"
                )
            native_signature = _native_signature(native_result)
            if native_signature != python_result.signature:
                raise RuntimeError(
                    f"DefenseSet mismatch for {case.name}:{coordinate}"
                )

            started = time.perf_counter()
            for _ in range(repeats):
                current: DefenseSet = analyzer._classify_defenses(
                    board,
                    attacker=case.player,
                    continuations=threat.continuations,
                    counter_wins=threat.counter_wins,
                    stop_requested=None,
                )
                if current.signature != python_result.signature:
                    raise RuntimeError("Python classification is nondeterministic")
            python_seconds = (time.perf_counter() - started) / repeats

            started = time.perf_counter()
            for _ in range(repeats):
                current_native = native_core.classify_defenses(
                    board,
                    case.player,
                    threat.continuations,
                    threat.counter_wins,
                )
                if _native_signature(current_native) != python_result.signature:
                    raise RuntimeError("Native classification is nondeterministic")
            native_seconds = (time.perf_counter() - started) / repeats
            if board_state(board) != placed_state:
                raise RuntimeError(
                    f"Timed defense classification polluted {coordinate}"
                )
        finally:
            board.undo()
        if board_state(board) != initial_state:
            raise RuntimeError(f"Candidate replay polluted {coordinate}")
        benchmarks.append(
            DefenseClassificationBenchmark(
                case_name=case.name,
                coordinate=coordinate,
                continuation_count=len(threat.continuations),
                legal_reply_count=python_result.legal_reply_count,
                equivalent=True,
                python_seconds=python_seconds,
                native_seconds=native_seconds,
                speedup=(
                    float("inf")
                    if native_seconds == 0.0
                    else python_seconds / native_seconds
                ),
            )
        )
    return tuple(benchmarks)


def run_benchmark(
    fixtures: tuple[Path, ...] = DEFAULT_FIXTURES,
    *,
    repeats: int = 3,
    minimum_speedup: float = 3.0,
) -> DefenseClassificationRun:
    candidates = tuple(
        benchmark
        for fixture in fixtures
        for benchmark in benchmark_case(load_case(fixture), repeats=repeats)
    )
    nontrivial = tuple(
        item.speedup for item in candidates if item.continuation_count > 0
    )
    observed_minimum = min(nontrivial) if nontrivial else None
    return DefenseClassificationRun(
        repeats=repeats,
        minimum_speedup=minimum_speedup,
        candidates=candidates,
        nontrivial_minimum_speedup=observed_minimum,
        production_eligible=(
            observed_minimum is not None
            and observed_minimum >= minimum_speedup
        ),
    )


def print_run(run: DefenseClassificationRun) -> None:
    print("Native defense classification (read-only)")
    print("Case                         Move  Cont  Replies   Python    Native  Speedup")
    for item in run.candidates:
        print(
            f"{item.case_name:<28} {item.coordinate:>4} "
            f"{item.continuation_count:>5} {item.legal_reply_count:>8} "
            f"{item.python_seconds:>8.4f}s {item.native_seconds:>8.4f}s "
            f"{item.speedup:>7.2f}x"
        )
    observed = run.nontrivial_minimum_speedup
    print(
        "Nontrivial minimum: "
        + ("n/a" if observed is None else f"{observed:.2f}x")
    )
    print(
        f"Production threshold: {run.minimum_speedup:.2f}x; "
        f"eligible={run.production_eligible}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="只读 Native 批量防守分类等价性与组件基准。"
    )
    parser.add_argument("--fixture", action="append", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--minimum-speedup", type=float, default=3.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    fixtures = (
        DEFAULT_FIXTURES
        if args.fixture is None
        else tuple(args.fixture)
    )
    run = run_benchmark(
        fixtures,
        repeats=args.repeats,
        minimum_speedup=args.minimum_speedup,
    )
    print_run(run)
    if args.json is not None:
        args.json.write_text(
            json.dumps(asdict(run), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
