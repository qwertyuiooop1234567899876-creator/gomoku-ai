from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable

from engine.board import BLACK, WHITE, Board
from engine.game import format_move, parse_move
from engine.search import SearchAI
from engine.search_types import INFINITY, RootResult, TTEntry
from engine.version import ENGINE_VERSION


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "positions" / "v0168_yixin_move13.json"


@dataclass(frozen=True, slots=True)
class BaselineCase:
    name: str
    board_size: int
    player: int
    history: tuple[str, ...]
    candidates: tuple[str, ...]
    expected_hash: int


@dataclass(frozen=True, slots=True)
class SearchRun:
    mode: str
    requested_depth: int
    node_limit: int | None
    completed: bool
    completed_depth: int
    stop_reason: str
    selected_move: str | None
    score: int | None
    ranked_moves: tuple[tuple[str, int], ...]
    principal_variation: tuple[str, ...]
    nodes: int
    elapsed_seconds: float
    tt_entries: int
    tt_digest: str
    tt_bounds: dict[str, int]


def load_case(path: Path = DEFAULT_FIXTURE) -> BaselineCase:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "gomoku-search-baseline-v1":
        raise ValueError(f"不支持的基线格式：{payload.get('format')!r}")
    player = int(payload["player"])
    if player not in (BLACK, WHITE):
        raise ValueError("基线行棋方必须是 BLACK 或 WHITE。")
    return BaselineCase(
        name=str(payload["name"]),
        board_size=int(payload["board_size"]),
        player=player,
        history=tuple(map(str, payload["history"])),
        candidates=tuple(map(str, payload["candidates"])),
        expected_hash=int(payload["zobrist_hash"]),
    )


def build_board(case: BaselineCase) -> Board:
    board = Board(case.board_size)
    for index, coordinate in enumerate(case.history):
        board.place(
            *parse_move(coordinate, board.size),
            BLACK if index % 2 == 0 else WHITE,
        )
    if board.zobrist_hash != case.expected_hash:
        raise RuntimeError(
            f"基线哈希不一致：{board.zobrist_hash} != {case.expected_hash}"
        )
    return board


def board_state(board: Board) -> tuple[object, ...]:
    return (
        tuple(tuple(row) for row in board.grid),
        tuple(board.move_history),
        board.zobrist_hash,
        board.empty_count,
    )


def _tt_payload(table: dict[int, TTEntry]) -> list[dict[str, object]]:
    return [
        {
            "key": key,
            "depth": entry.depth,
            "extension_depth": entry.extension_depth,
            "score": entry.score,
            "bound": entry.bound.value,
            "best_move": entry.best_move,
            "principal_variation": entry.principal_variation,
            "generation": entry.generation,
        }
        for key, entry in sorted(table.items())
    ]


def tt_summary(table: dict[int, TTEntry]) -> tuple[int, str, dict[str, int]]:
    payload = _tt_payload(table)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    bounds: dict[str, int] = {}
    for item in payload:
        bound = str(item["bound"])
        bounds[bound] = bounds.get(bound, 0) + 1
    return len(payload), hashlib.sha256(encoded).hexdigest(), bounds


def _format_result(
    ai: SearchAI,
    *,
    mode: str,
    requested_depth: int,
    node_limit: int | None,
    completed: bool,
    completed_depth: int,
    stop_reason: str,
    result: RootResult | None,
    elapsed_seconds: float,
) -> SearchRun:
    count, digest, bounds = tt_summary(ai._transposition_table)
    return SearchRun(
        mode=mode,
        requested_depth=requested_depth,
        node_limit=node_limit,
        completed=completed,
        completed_depth=completed_depth,
        stop_reason=stop_reason,
        selected_move=(
            None if result is None else format_move(*result.move)
        ),
        score=None if result is None else result.score,
        ranked_moves=(
            ()
            if result is None
            else tuple(
                (format_move(*move), score)
                for move, score in result.ranked_moves
            )
        ),
        principal_variation=(
            ()
            if result is None
            else tuple(format_move(*move) for move in result.principal_variation)
        ),
        nodes=ai._counters.nodes,
        elapsed_seconds=elapsed_seconds,
        tt_entries=count,
        tt_digest=digest,
        tt_bounds=bounds,
    )


def run_full_window_candidate(
    case: BaselineCase,
    coordinate: str,
    depth: int,
) -> SearchRun:
    board = build_board(case)
    before = board_state(board)
    move = parse_move(coordinate, board.size)
    if not board.is_empty(*move):
        raise ValueError(f"候选点不是空位：{coordinate}")
    ai = SearchAI(
        case.player,
        max_depth=depth,
        time_limit_seconds=None,
    )
    ai._begin_move_search()
    started_at = time.perf_counter()
    result = ai._search_root(
        board,
        case.player,
        depth,
        [move],
        alpha=-INFINITY,
        beta=INFINITY,
    )
    elapsed = time.perf_counter() - started_at
    if board_state(board) != before:
        raise RuntimeError("单候选全窗口搜索污染了棋盘或有序历史。")
    return _format_result(
        ai,
        mode=f"full_window:{coordinate}",
        requested_depth=depth,
        node_limit=None,
        completed=True,
        completed_depth=depth,
        stop_reason="requested_depth_completed",
        result=result,
        elapsed_seconds=elapsed,
    )


def run_iterative_pair(
    case: BaselineCase,
    depth: int,
    *,
    node_limit: int | None,
) -> SearchRun:
    board = build_board(case)
    before = board_state(board)
    candidates = [
        parse_move(coordinate, board.size)
        for coordinate in case.candidates
    ]
    ai = SearchAI(
        case.player,
        max_depth=depth,
        time_limit_seconds=None,
        node_limit=node_limit,
    )
    ai._begin_move_search()
    started_at = time.perf_counter()
    outcome = ai._run_iterative_root_search(
        board,
        candidates,
        fallback_move=candidates[0],
        preserve_frontier_order=True,
        allow_near_loss_expansion=False,
        defense_probe=None,
    )
    elapsed = time.perf_counter() - started_at
    if board_state(board) != before:
        raise RuntimeError("双候选迭代搜索污染了棋盘或有序历史。")
    return _format_result(
        ai,
        mode="iterative_pair",
        requested_depth=depth,
        node_limit=node_limit,
        completed=outcome.search_completed,
        completed_depth=outcome.completed_depth,
        stop_reason=outcome.stop_reason,
        result=outcome.result,
        elapsed_seconds=elapsed,
    )


def run_production(
    case: BaselineCase,
    depth: int,
    *,
    time_limit_seconds: float | None,
    node_limit: int | None,
    warm_history: bool = False,
) -> SearchRun:
    ai = SearchAI(
        case.player,
        max_depth=depth,
        time_limit_seconds=time_limit_seconds,
        node_limit=node_limit,
        diagnostics=True,
    )
    if warm_history:
        board = Board(case.board_size)
        for index, coordinate in enumerate(case.history):
            player = BLACK if index % 2 == 0 else WHITE
            recorded = parse_move(coordinate, board.size)
            if player == case.player:
                selected = ai.choose_move(board)
                if selected != recorded:
                    raise RuntimeError(
                        "预热重放与记录分歧："
                        f"第 {index + 1} 手记录 {coordinate}，"
                        f"当前引擎 {format_move(*selected)}。"
                    )
            board.place(*recorded, player)
        if board.zobrist_hash != case.expected_hash:
            raise RuntimeError("预热重放后的棋盘哈希不一致。")
    else:
        board = build_board(case)
    before = board_state(board)
    started_at = time.perf_counter()
    ai.choose_move(board)
    elapsed = time.perf_counter() - started_at
    if board_state(board) != before:
        raise RuntimeError("生产管线搜索污染了棋盘或有序历史。")
    analysis = ai.last_analysis
    if analysis is None:
        raise RuntimeError("生产管线没有生成搜索诊断。")
    selected_score = next(
        (
            candidate.score
            for candidate in analysis.top_candidates
            if candidate.move == analysis.selected_move
        ),
        analysis.top_candidates[0].score if analysis.top_candidates else 0,
    )
    result = RootResult(
        move=analysis.selected_move,
        score=selected_score,
        principal_variation=analysis.principal_variation,
        ranked_moves=tuple(
            (item.move, item.score) for item in analysis.top_candidates
        ),
    )
    return _format_result(
        ai,
        mode="production_warm" if warm_history else "production_cold",
        requested_depth=depth,
        node_limit=node_limit,
        completed=analysis.search_completed,
        completed_depth=analysis.search_depth,
        stop_reason=analysis.stop_reason,
        result=result,
        elapsed_seconds=elapsed,
    )


def parse_depths(text: str) -> tuple[int, ...]:
    if "-" in text:
        first_text, last_text = text.split("-", 1)
        first = int(first_text)
        last = int(last_text)
        depths = tuple(range(first, last + 1))
    else:
        depths = tuple(int(item) for item in text.split(","))
    if not depths or any(depth < 1 for depth in depths):
        raise ValueError("搜索深度必须全部大于 0。")
    return depths


def print_runs(runs: Iterable[SearchRun]) -> None:
    print(
        f"{'Mode':20} {'Req':>3} {'Done':>4} {'Move':>5} "
        f"{'Score':>11} {'Nodes':>9} {'Seconds':>9} {'Stop'}"
    )
    print("-" * 92)
    for run in runs:
        print(
            f"{run.mode:20} {run.requested_depth:>3} "
            f"{run.completed_depth:>4} {(run.selected_move or '-'):>5} "
            f"{(run.score if run.score is not None else '-'):>11} "
            f"{run.nodes:>9} {run.elapsed_seconds:>8.3f}s "
            f"{run.stop_reason}"
        )
        if len(run.ranked_moves) > 1:
            print(
                " " * 7
                + "ranked="
                + ", ".join(
                    f"{move}:{score}"
                    for move, score in run.ranked_moves
                )
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="运行 Native 下沉前的第13手固定搜索基线。"
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--mode",
        choices=("full-window", "iterative", "production"),
        default="full-window",
    )
    parser.add_argument(
        "--depths",
        default="1-8",
        help="深度范围，例如 1-8 或 4,6,8。",
    )
    parser.add_argument("--node-limit", type=int)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument(
        "--warm-history",
        action="store_true",
        help="生产模式下逐手重放己方历史搜索，保留TT和排序状态。",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.node_limit is not None and args.node_limit < 1:
        parser.error("--node-limit 必须大于 0。")
    case = load_case(args.fixture)
    depths = parse_depths(args.depths)
    runs: list[SearchRun] = []
    for depth in depths:
        if args.mode == "full-window":
            runs.extend(
                run_full_window_candidate(case, coordinate, depth)
                for coordinate in case.candidates
            )
        elif args.mode == "iterative":
            runs.append(
                run_iterative_pair(
                    case,
                    depth,
                    node_limit=args.node_limit,
                )
            )
        else:
            runs.append(
                run_production(
                    case,
                    depth,
                    time_limit_seconds=args.time_limit,
                    node_limit=args.node_limit,
                    warm_history=args.warm_history,
                )
            )
    print(
        f"Native search baseline | engine={ENGINE_VERSION} | "
        f"case={case.name} | history={len(case.history)}"
    )
    print_runs(runs)
    if args.json is not None:
        report = {
            "engine_version": ENGINE_VERSION,
            "case": asdict(case),
            "runs": [asdict(run) for run in runs],
        }
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
