from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable

from engine.board import BLACK, WHITE, Board
from engine.game import format_move, parse_move
from engine.native_core import (
    STATUS_CUTOFF,
    STATUS_FOUND,
    native_core,
)
from engine.search import SearchAI
from engine.search_types import INFINITY, MATE_SCORE, RootResult, TTEntry
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
class CandidateLayerSummary:
    """A bounded representative of one internal PVS candidate layer."""

    ply: int
    remaining_depth: int
    extension_depth: int
    calls: int
    minimum_candidates: int
    maximum_candidates: int
    sample_moves: tuple[str, ...]
    sample_truncated: bool


@dataclass(frozen=True, slots=True)
class LeafTrace:
    """One opt-in heuristic category from the baseline-only tracer."""

    ply: int
    remaining_depth: int
    extension_depth: int
    score: int
    trace_category: str
    principal_variation: tuple[str, ...]


@dataclass(slots=True)
class _CandidateLayerAccumulator:
    ply: int
    remaining_depth: int
    extension_depth: int
    calls: int = 0
    minimum_candidates: int = 0
    maximum_candidates: int = 0
    sample_moves: tuple[str, ...] = ()
    sample_truncated: bool = False


class _TracingSearchAI(SearchAI):
    """Tool-only bounded trace without adding production search overhead."""

    def __init__(
        self,
        *args: object,
        candidate_trace_limit: int,
        candidate_sample_limit: int,
        leaf_trace_limit: int,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._baseline_candidate_trace_limit = candidate_trace_limit
        self._baseline_candidate_sample_limit = candidate_sample_limit
        self._baseline_leaf_trace_limit = leaf_trace_limit
        self._baseline_context: list[tuple[int, int, int]] = []
        self._baseline_candidate_layers: dict[
            tuple[int, int, int], _CandidateLayerAccumulator
        ] = {}
        self._baseline_candidate_trace_truncated = False
        self._baseline_leaf_trace: list[LeafTrace] = []
        self._baseline_leaf_trace_truncated = False

    def _negamax(
        self,
        board: Board,
        player: int,
        depth: int,
        alpha: int,
        beta: int,
        *,
        ply: int,
        extension_depth: int,
    ) -> tuple[int, tuple[tuple[int, int], ...]]:
        self._baseline_context.append((ply, depth, extension_depth))
        try:
            score, variation = super()._negamax(
                board,
                player,
                depth,
                alpha,
                beta,
                ply=ply,
                extension_depth=extension_depth,
            )
        finally:
            self._baseline_context.pop()
        if depth <= 0:
            self._record_leaf_trace(
                ply=ply,
                depth=depth,
                extension_depth=extension_depth,
                score=score,
                variation=variation,
            )
        return score, variation

    def _ordered_moves(
        self,
        board: Board,
        player: int,
        *,
        at_root: bool,
        ply: int,
        limit: int | None = None,
        tt_move: tuple[int, int] | None = None,
        use_search_heuristics: bool = True,
    ) -> list[tuple[int, int]]:
        moves = super()._ordered_moves(
            board,
            player,
            at_root=at_root,
            ply=ply,
            limit=limit,
            tt_move=tt_move,
            use_search_heuristics=use_search_heuristics,
        )
        if not at_root:
            self._record_candidate_layer(moves)
        return moves

    def _record_candidate_layer(
        self,
        moves: list[tuple[int, int]],
    ) -> None:
        if (
            self._baseline_candidate_trace_limit <= 0
            or not self._baseline_context
        ):
            return
        key = self._baseline_context[-1]
        accumulator = self._baseline_candidate_layers.get(key)
        if accumulator is None:
            if (
                len(self._baseline_candidate_layers)
                >= self._baseline_candidate_trace_limit
            ):
                self._baseline_candidate_trace_truncated = True
                return
            accumulator = _CandidateLayerAccumulator(
                ply=key[0],
                remaining_depth=key[1],
                extension_depth=key[2],
            )
            self._baseline_candidate_layers[key] = accumulator
        candidate_count = len(moves)
        accumulator.calls += 1
        if accumulator.calls == 1:
            accumulator.minimum_candidates = candidate_count
            accumulator.maximum_candidates = candidate_count
            accumulator.sample_moves = tuple(
                format_move(*move)
                for move in moves[: self._baseline_candidate_sample_limit]
            )
            accumulator.sample_truncated = (
                candidate_count > self._baseline_candidate_sample_limit
            )
            return
        accumulator.minimum_candidates = min(
            accumulator.minimum_candidates,
            candidate_count,
        )
        accumulator.maximum_candidates = max(
            accumulator.maximum_candidates,
            candidate_count,
        )

    def _record_leaf_trace(
        self,
        *,
        ply: int,
        depth: int,
        extension_depth: int,
        score: int,
        variation: tuple[tuple[int, int], ...],
    ) -> None:
        if self._baseline_leaf_trace_limit <= 0:
            return
        if len(self._baseline_leaf_trace) >= self._baseline_leaf_trace_limit:
            self._baseline_leaf_trace_truncated = True
            return
        trace_category = "no_forcing_static"
        if abs(score) >= MATE_SCORE - ply:
            trace_category = "terminal_or_forced"
        elif variation:
            trace_category = "forcing_extension"
        elif extension_depth <= 0:
            trace_category = "extension_limit_static"
        self._baseline_leaf_trace.append(
            LeafTrace(
                ply=ply,
                remaining_depth=depth,
                extension_depth=extension_depth,
                score=score,
                trace_category=trace_category,
                principal_variation=tuple(
                    format_move(*move) for move in variation
                ),
            )
        )

    def baseline_trace(
        self,
    ) -> tuple[
        tuple[CandidateLayerSummary, ...],
        bool,
        tuple[LeafTrace, ...],
        bool,
    ]:
        layers = tuple(
            CandidateLayerSummary(
                ply=item.ply,
                remaining_depth=item.remaining_depth,
                extension_depth=item.extension_depth,
                calls=item.calls,
                minimum_candidates=item.minimum_candidates,
                maximum_candidates=item.maximum_candidates,
                sample_moves=item.sample_moves,
                sample_truncated=item.sample_truncated,
            )
            for _key, item in sorted(self._baseline_candidate_layers.items())
        )
        return (
            layers,
            self._baseline_candidate_trace_truncated,
            tuple(self._baseline_leaf_trace),
            self._baseline_leaf_trace_truncated,
        )


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
    threat_extension_depth: int
    branch_candidate_limit: int
    extensions: int
    candidate_layers: tuple[CandidateLayerSummary, ...] = ()
    candidate_trace_truncated: bool = False
    leaf_trace: tuple[LeafTrace, ...] = ()
    leaf_trace_truncated: bool = False


@dataclass(frozen=True, slots=True)
class NativeReviewCandidateRun:
    """One independent Native full-window root-candidate search."""

    coordinate: str
    status: str
    completed_depth: int
    score: int | None
    principal_variation: tuple[str, ...]
    nodes: int
    elapsed_seconds: float
    tt_entries: int
    tt_digest: str
    input_digest: str


@dataclass(frozen=True, slots=True)
class NativeReviewLayer:
    """One complete or interrupted fixed-depth comparison layer."""

    requested_depth: int
    completed: bool
    leader: str | None
    candidates: tuple[NativeReviewCandidateRun, ...]
    nodes: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class NativeReviewRun:
    """Tool-only Native review ladder; never a production safety result."""

    mode: str
    requested_depths: tuple[int, ...]
    completed_depth: int
    stop_reason: str
    leader_history: tuple[str, ...]
    layers: tuple[NativeReviewLayer, ...]
    threat_extension_depth: int
    branch_candidate_limit: int
    node_limit_per_candidate: int | None


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
    trace_snapshot = getattr(ai, "baseline_trace", None)
    if callable(trace_snapshot):
        (
            candidate_layers,
            candidate_trace_truncated,
            leaf_trace,
            leaf_trace_truncated,
        ) = trace_snapshot()
    else:
        candidate_layers = ()
        candidate_trace_truncated = False
        leaf_trace = ()
        leaf_trace_truncated = False
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
        threat_extension_depth=ai.config.threat_extension_depth,
        branch_candidate_limit=ai.config.branch_candidate_limit,
        extensions=ai._counters.extensions,
        candidate_layers=candidate_layers,
        candidate_trace_truncated=candidate_trace_truncated,
        leaf_trace=leaf_trace,
        leaf_trace_truncated=leaf_trace_truncated,
    )


def run_full_window_candidate(
    case: BaselineCase,
    coordinate: str,
    depth: int,
    *,
    threat_extension_depth: int = 2,
    branch_candidate_limit: int = 8,
    candidate_trace_limit: int = 0,
    candidate_sample_limit: int = 8,
    leaf_trace_limit: int = 0,
    use_pvs: bool = True,
) -> SearchRun:
    board = build_board(case)
    before = board_state(board)
    move = parse_move(coordinate, board.size)
    if not board.is_empty(*move):
        raise ValueError(f"候选点不是空位：{coordinate}")
    if candidate_trace_limit > 0 or leaf_trace_limit > 0:
        ai = _TracingSearchAI(
            case.player,
            max_depth=depth,
            time_limit_seconds=None,
            threat_extension_depth=threat_extension_depth,
            branch_candidate_limit=branch_candidate_limit,
            candidate_trace_limit=candidate_trace_limit,
            candidate_sample_limit=candidate_sample_limit,
            leaf_trace_limit=leaf_trace_limit,
        )
    else:
        ai = SearchAI(
            case.player,
            max_depth=depth,
            time_limit_seconds=None,
            threat_extension_depth=threat_extension_depth,
            branch_candidate_limit=branch_candidate_limit,
        )
    ai.config = replace(ai.config, use_pvs=use_pvs)
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
    threat_extension_depth: int = 2,
    branch_candidate_limit: int = 8,
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
        threat_extension_depth=threat_extension_depth,
        branch_candidate_limit=branch_candidate_limit,
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


def run_native_pair(
    case: BaselineCase,
    depth: int,
    *,
    node_limit: int | None,
    threat_extension_depth: int,
    branch_candidate_limit: int,
) -> SearchRun:
    """Run the Phase-1 C++ core without wiring it into production policy."""
    board = build_board(case)
    before = board_state(board)
    candidates = tuple(
        parse_move(coordinate, board.size)
        for coordinate in case.candidates
    )
    started_at = time.perf_counter()
    probe = native_core.probe_main_search_contract(
        board,
        case.player,
        candidates,
        depth=depth,
        node_limit=node_limit,
        branch_candidate_limit=branch_candidate_limit,
        preselection_factor=3,
        candidate_radius=2,
        recent_move_count=4,
        threat_extension_depth=threat_extension_depth,
        use_pvs=True,
        use_transposition_table=True,
    )
    elapsed = time.perf_counter() - started_at
    if board_state(board) != before:
        raise RuntimeError("Native 主搜索污染了棋盘或有序历史。")
    if probe is None:
        raise RuntimeError(
            "Native 主搜索运行库不可用："
            f"{native_core.error or '缺少 gn_main_search_v1'}"
        )
    if probe.status not in (STATUS_FOUND, STATUS_CUTOFF):
        raise RuntimeError(f"Native 主搜索失败：status={probe.status}")

    indexed_scores = [
        (index, move, score)
        for index, (move, score) in enumerate(probe.root_scores)
    ]
    indexed_scores.sort(key=lambda item: (item[2], -item[0]), reverse=True)
    return SearchRun(
        mode="native_pair",
        requested_depth=depth,
        node_limit=node_limit,
        completed=probe.completed,
        completed_depth=probe.completed_depth,
        stop_reason=(
            "requested_depth_completed"
            if probe.completed
            else "node_limit"
        ),
        selected_move=(
            None
            if probe.best_move is None
            else format_move(*probe.best_move)
        ),
        score=None if probe.best_move is None else probe.score,
        ranked_moves=tuple(
            (format_move(*move), score)
            for _index, move, score in indexed_scores
        ),
        principal_variation=tuple(
            format_move(*move)
            for move in probe.principal_variation
        ),
        nodes=probe.nodes,
        elapsed_seconds=elapsed,
        tt_entries=probe.tt_entries,
        tt_digest=f"{probe.tt_digest:016x}",
        tt_bounds={},
        threat_extension_depth=threat_extension_depth,
        branch_candidate_limit=branch_candidate_limit,
        extensions=0,
    )


def run_native_full_window_review(
    case: BaselineCase,
    depths: tuple[int, ...],
    *,
    node_limit: int | None,
    threat_extension_depth: int,
    branch_candidate_limit: int,
    coordinates: tuple[str, ...] | None = None,
) -> NativeReviewRun:
    """Compare each candidate in an isolated one-candidate Native call.

    A node limit applies independently to every candidate call.  An interrupted
    candidate makes the whole depth incomplete, so partial scores never enter
    the leader history.
    """
    if not depths or any(depth < 1 for depth in depths):
        raise ValueError("Native复核深度必须全部大于 0。")
    selected_coordinates = case.candidates if coordinates is None else coordinates
    if not selected_coordinates:
        raise ValueError("Native复核至少需要一个候选。")
    unknown = [
        coordinate
        for coordinate in selected_coordinates
        if coordinate not in case.candidates
    ]
    if unknown:
        raise ValueError(
            "Native复核候选必须来自夹具：" + ", ".join(case.candidates)
        )

    board = build_board(case)
    before = board_state(board)
    moves = tuple(
        parse_move(coordinate, board.size)
        for coordinate in selected_coordinates
    )
    for coordinate, move in zip(selected_coordinates, moves):
        if not board.is_empty(*move):
            raise ValueError(f"候选点不是空位：{coordinate}")

    layers: list[NativeReviewLayer] = []
    leader_history: list[str] = []
    completed_depth = 0
    stop_reason = "requested_depths_completed"
    for depth in depths:
        candidate_runs: list[NativeReviewCandidateRun] = []
        layer_started_at = time.perf_counter()
        layer_completed = True
        for coordinate, move in zip(selected_coordinates, moves):
            started_at = time.perf_counter()
            probe = native_core.probe_main_search_contract(
                board,
                case.player,
                (move,),
                depth=depth,
                node_limit=node_limit,
                branch_candidate_limit=branch_candidate_limit,
                preselection_factor=3,
                candidate_radius=2,
                recent_move_count=4,
                threat_extension_depth=threat_extension_depth,
                use_pvs=False,
                use_transposition_table=True,
            )
            elapsed = time.perf_counter() - started_at
            if probe is None:
                raise RuntimeError(
                    "Native 主搜索运行库不可用："
                    f"{native_core.error or '缺少 gn_main_search_v1'}"
                )
            if probe.status not in (STATUS_FOUND, STATUS_CUTOFF):
                raise RuntimeError(
                    f"Native 独立复核失败：status={probe.status}"
                )
            if probe.status == STATUS_FOUND:
                if (
                    probe.completed_depth != depth
                    or probe.best_move != move
                    or len(probe.root_scores) != 1
                    or probe.root_scores[0][0] != move
                    or probe.root_scores[0][1] != probe.score
                    or not probe.principal_variation
                    or probe.principal_variation[0] != move
                ):
                    raise RuntimeError("Native 独立复核返回了不完整的固定深度结果。")
                status = "completed"
                score: int | None = probe.score
                variation = tuple(
                    format_move(*item) for item in probe.principal_variation
                )
            else:
                status = "node_limit"
                score = None
                variation = ()
                layer_completed = False
            candidate_runs.append(
                NativeReviewCandidateRun(
                    coordinate=coordinate,
                    status=status,
                    completed_depth=probe.completed_depth,
                    score=score,
                    principal_variation=variation,
                    nodes=probe.nodes,
                    elapsed_seconds=elapsed,
                    tt_entries=probe.tt_entries,
                    tt_digest=f"{probe.tt_digest:016x}",
                    input_digest=f"{probe.input_digest:016x}",
                )
            )
            if not layer_completed:
                break

        leader: str | None = None
        if layer_completed:
            leader = max(
                enumerate(candidate_runs),
                key=lambda item: (
                    item[1].score,
                    -item[0],
                ),
            )[1].coordinate
            leader_history.append(leader)
            completed_depth = depth
        else:
            stop_reason = "node_limit"
        layers.append(
            NativeReviewLayer(
                requested_depth=depth,
                completed=layer_completed,
                leader=leader,
                candidates=tuple(candidate_runs),
                nodes=sum(item.nodes for item in candidate_runs),
                elapsed_seconds=time.perf_counter() - layer_started_at,
            )
        )
        if not layer_completed:
            break

    if board_state(board) != before:
        raise RuntimeError("Native 独立复核污染了棋盘或有序历史。")
    return NativeReviewRun(
        mode="native_review",
        requested_depths=depths,
        completed_depth=completed_depth,
        stop_reason=stop_reason,
        leader_history=tuple(leader_history),
        layers=tuple(layers),
        threat_extension_depth=threat_extension_depth,
        branch_candidate_limit=branch_candidate_limit,
        node_limit_per_candidate=node_limit,
    )


def run_dynamic_pair(
    case: BaselineCase,
    depth: int,
    *,
    review_budget_seconds: float,
    quiet_frontier_extension: bool,
    threat_extension_depth: int,
    branch_candidate_limit: int,
) -> SearchRun:
    """Run the quiet-frontier experiment through dynamic pair review only."""
    board = build_board(case)
    before = board_state(board)
    candidates = [
        parse_move(coordinate, board.size)
        for coordinate in case.candidates
    ]
    if len(candidates) != 2:
        raise ValueError("dynamic-pair 基线需要恰好两个候选。")
    ai = SearchAI(
        case.player,
        max_depth=depth,
        time_limit_seconds=review_budget_seconds,
        branch_candidate_limit=branch_candidate_limit,
        threat_extension_depth=threat_extension_depth,
    )
    ai._begin_move_search()
    seed = RootResult(
        move=candidates[0],
        score=0,
        principal_variation=(candidates[0],),
        ranked_moves=((candidates[0], 0), (candidates[1], 0)),
    )
    started_at = time.perf_counter()
    probe = ai._run_dynamic_pair_review(
        board,
        seed,
        candidates[1],
        completed_depth=depth,
        budget_seconds=review_budget_seconds,
        target_depth_override=depth,
        branch_candidate_limit_override=branch_candidate_limit,
        quiet_frontier_extension_override=quiet_frontier_extension,
    )
    elapsed = time.perf_counter() - started_at
    if board_state(board) != before:
        raise RuntimeError("dynamic-pair 复核污染了棋盘或有序历史。")
    if probe is None:
        result = None
        completed = False
        completed_depth = 0
        stop_reason = "dynamic_pair_unavailable"
    else:
        ranked = tuple(
            (candidate.move, candidate.score)
            for candidate in probe.candidates
        )
        selected = probe.best_move or seed.move
        score = next(
            (score for move, score in ranked if move == selected),
            seed.score,
        )
        variation = next(
            (
                candidate.principal_variation
                for candidate in probe.candidates
                if candidate.move == selected
            ),
            (selected,),
        )
        result = RootResult(
            move=selected,
            score=score,
            principal_variation=variation,
            ranked_moves=ranked,
        )
        completed = probe.completed_depth >= depth
        completed_depth = probe.completed_depth
        stop_reason = "dynamic_pair_completed"
    run = _format_result(
        ai,
        mode=(
            "dynamic_pair:quiet"
            if quiet_frontier_extension
            else "dynamic_pair:normal"
        ),
        requested_depth=depth,
        node_limit=None,
        completed=completed,
        completed_depth=completed_depth,
        stop_reason=stop_reason,
        result=result,
        elapsed_seconds=elapsed,
    )
    actual_extension_depth = (
        max(
            6,
            threat_extension_depth + ai.config.root_safety_extension_bonus,
        )
        if quiet_frontier_extension
        else (
            threat_extension_depth
            + ai.config.root_safety_extension_bonus
        )
    )
    return replace(run, threat_extension_depth=actual_extension_depth)


def run_defense_vct_pair(
    case: BaselineCase,
    depth: int,
    *,
    time_limit_seconds: float,
) -> SearchRun:
    """Run the narrow Defense-VCT channel as a separate experiment mode."""
    board = build_board(case)
    before = board_state(board)
    candidates = [
        parse_move(coordinate, board.size)
        for coordinate in case.candidates
    ]
    ai = SearchAI(
        case.player,
        max_depth=depth,
        time_limit_seconds=time_limit_seconds,
    )
    ai.config = replace(ai.config, defense_vct_probe_depth=depth)
    ai._begin_move_search()
    started_at = time.perf_counter()
    probe = ai._run_defense_vct_probe(board, case.player, candidates)
    elapsed = time.perf_counter() - started_at
    if board_state(board) != before:
        raise RuntimeError("Defense-VCT 探针污染了棋盘或有序历史。")
    if probe is None:
        result = None
        completed = False
        completed_depth = 0
        stop_reason = "defense_vct_incomplete"
    else:
        best = probe.candidates[0]
        result = RootResult(
            move=best.move,
            score=best.score,
            principal_variation=best.principal_variation,
            ranked_moves=tuple(
                (candidate.move, candidate.score)
                for candidate in probe.candidates
            ),
        )
        completed = probe.completed_depth >= depth
        completed_depth = probe.completed_depth
        stop_reason = "defense_vct_completed"
    run = _format_result(
        ai,
        mode="defense_vct",
        requested_depth=depth,
        node_limit=None,
        completed=completed,
        completed_depth=completed_depth,
        stop_reason=stop_reason,
        result=result,
        elapsed_seconds=elapsed,
    )
    return replace(
        run,
        threat_extension_depth=ai.config.defense_vct_extension_depth,
        branch_candidate_limit=ai.config.defense_vct_branch_limit,
    )


def run_production(
    case: BaselineCase,
    depth: int,
    *,
    time_limit_seconds: float | None,
    node_limit: int | None,
    warm_history: bool = False,
    threat_extension_depth: int = 2,
    branch_candidate_limit: int = 8,
) -> SearchRun:
    ai = SearchAI(
        case.player,
        max_depth=depth,
        time_limit_seconds=time_limit_seconds,
        node_limit=node_limit,
        threat_extension_depth=threat_extension_depth,
        branch_candidate_limit=branch_candidate_limit,
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
        f"{'Mode':20} {'Req':>3} {'Done':>4} {'Ext':>3} {'Br':>3} "
        f"{'Move':>5} {'Score':>11} {'Nodes':>9} {'Exts':>5} "
        f"{'Seconds':>9} {'Stop'}"
    )
    print("-" * 112)
    for run in runs:
        print(
            f"{run.mode:20} {run.requested_depth:>3} "
            f"{run.completed_depth:>4} {run.threat_extension_depth:>3} "
            f"{run.branch_candidate_limit:>3} "
            f"{(run.selected_move or '-'):>5} "
            f"{(run.score if run.score is not None else '-'):>11} "
            f"{run.nodes:>9} {run.extensions:>5} "
            f"{run.elapsed_seconds:>8.3f}s "
            f"{run.stop_reason}"
        )
        print(
            " " * 7
            + "pv="
            + (" ".join(run.principal_variation) or "-")
            + " | tt="
            + f"{run.tt_entries}/{run.tt_digest[:16]}/{run.tt_bounds}"
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
        for layer in run.candidate_layers:
            sample = ",".join(layer.sample_moves) or "-"
            suffix = "…" if layer.sample_truncated else ""
            print(
                " " * 7
                + "layer="
                + (
                    f"ply{layer.ply}/d{layer.remaining_depth}/"
                    f"e{layer.extension_depth} "
                    f"calls={layer.calls} candidates="
                    f"{layer.minimum_candidates}-{layer.maximum_candidates} "
                    f"sample=[{sample}{suffix}]"
                )
            )
        if run.candidate_trace_truncated:
            print("       candidate_layers=truncated")
        for leaf in run.leaf_trace:
            variation = " ".join(leaf.principal_variation) or "-"
            print(
                " " * 7
                + (
                    f"leaf=ply{leaf.ply}/d{leaf.remaining_depth}/"
                    f"e{leaf.extension_depth} {leaf.trace_category} "
                    f"score={leaf.score} pv={variation}"
                )
            )
        if run.leaf_trace_truncated:
            print("       leaf_trace=truncated")


def print_native_review(run: NativeReviewRun) -> None:
    print(
        f"{'Depth':>5} {'Done':>4} {'Leader':>6} {'Move':>5} "
        f"{'Score':>11} {'Nodes':>9} {'Seconds':>9} {'Status'}"
    )
    print("-" * 82)
    for layer in run.layers:
        for index, candidate in enumerate(layer.candidates):
            print(
                f"{layer.requested_depth:>5} "
                f"{('yes' if layer.completed else 'no'):>4} "
                f"{((layer.leader or '-') if index == 0 else ''):>6} "
                f"{candidate.coordinate:>5} "
                f"{(candidate.score if candidate.score is not None else '-'):>11} "
                f"{candidate.nodes:>9} "
                f"{candidate.elapsed_seconds:>8.3f}s "
                f"{candidate.status}"
            )
            print(
                " " * 13
                + "pv="
                + (" ".join(candidate.principal_variation) or "-")
                + " | tt="
                + f"{candidate.tt_entries}/{candidate.tt_digest}"
            )
    print(
        "leader_history="
        + (" ".join(run.leader_history) or "-")
        + f" | stop={run.stop_reason}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="运行 Native 下沉前的第13手固定搜索基线。"
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--mode",
        choices=(
            "full-window",
            "iterative",
            "production",
            "dynamic-pair",
            "defense-vct",
            "native",
            "native-review",
        ),
        default="full-window",
    )
    parser.add_argument(
        "--depths",
        default="1-8",
        help="深度范围，例如 1-8 或 4,6,8。",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        help="仅 full-window：指定一个或多个夹具候选，可重复传入。",
    )
    parser.add_argument("--node-limit", type=int)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument(
        "--threat-extension-depth",
        type=int,
        default=2,
        help=(
            "主 PVS 的威胁延伸层数（full-window 默认 2；"
            "不会开启安静前沿）。"
        ),
    )
    parser.add_argument(
        "--branch-candidate-limit",
        type=int,
        default=8,
        help="主 PVS 的非根候选上限（默认 8）。",
    )
    parser.add_argument(
        "--candidate-trace-limit",
        type=int,
        default=0,
        help="显式启用 full-window 每层候选摘要的最大层数；0 禁用（默认）。",
    )
    parser.add_argument(
        "--candidate-sample-limit",
        type=int,
        default=8,
        help="每层候选摘要保留的有序样本数（默认 8）。",
    )
    parser.add_argument(
        "--leaf-trace-limit",
        type=int,
        default=0,
        help=(
            "显式启用 full-window 的有界叶面返回 trace；"
            "0 保持关闭（默认）。"
        ),
    )
    parser.add_argument(
        "--quiet-frontier",
        action="store_true",
        help=(
            "仅 dynamic-pair 模式：在该独立复核中开启安静前沿延伸。"
        ),
    )
    parser.add_argument(
        "--review-budget",
        type=float,
        default=10.0,
        help="dynamic-pair 的独立复核预算秒数（默认 10）。",
    )
    parser.add_argument(
        "--warm-history",
        action="store_true",
        help="生产模式下逐手重放己方历史搜索，保留TT和排序状态。",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.node_limit is not None and args.node_limit < 1:
        parser.error("--node-limit 必须大于 0。")
    if args.time_limit is not None and args.time_limit <= 0:
        parser.error("--time-limit 必须大于 0。")
    if args.threat_extension_depth < 0:
        parser.error("--threat-extension-depth 不能小于 0。")
    if args.branch_candidate_limit < 1:
        parser.error("--branch-candidate-limit 必须大于 0。")
    if args.candidate_trace_limit < 0:
        parser.error("--candidate-trace-limit 不能小于 0。")
    if args.candidate_sample_limit < 1:
        parser.error("--candidate-sample-limit 必须大于 0。")
    if args.leaf_trace_limit < 0:
        parser.error("--leaf-trace-limit 不能小于 0。")
    if args.review_budget <= 0:
        parser.error("--review-budget 必须大于 0。")
    if args.quiet_frontier and args.mode != "dynamic-pair":
        parser.error("--quiet-frontier 只能在 --mode dynamic-pair 使用。")
    if args.leaf_trace_limit and args.mode != "full-window":
        parser.error("--leaf-trace-limit 只能在 --mode full-window 使用。")
    if args.candidate is not None and args.mode not in {
        "full-window",
        "native-review",
    }:
        parser.error("--candidate 只能在 full-window/native-review 使用。")
    case = load_case(args.fixture)
    selected_coordinates = (
        case.candidates
        if args.candidate is None
        else tuple(args.candidate)
    )
    unknown_coordinates = [
        coordinate
        for coordinate in selected_coordinates
        if coordinate not in case.candidates
    ]
    if unknown_coordinates:
        parser.error(
            "--candidate 必须来自夹具候选："
            + ", ".join(case.candidates)
        )
    depths = parse_depths(args.depths)
    if args.mode == "native-review":
        review = run_native_full_window_review(
            case,
            depths,
            node_limit=args.node_limit,
            threat_extension_depth=args.threat_extension_depth,
            branch_candidate_limit=args.branch_candidate_limit,
            coordinates=selected_coordinates,
        )
        print(
            f"Native full-window review | engine={ENGINE_VERSION} | "
            f"case={case.name} | history={len(case.history)}"
        )
        print_native_review(review)
        if args.json is not None:
            report = {
                "engine_version": ENGINE_VERSION,
                "case": asdict(case),
                "review": asdict(review),
            }
            args.json.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"JSON: {args.json}")
        return 0
    runs: list[SearchRun] = []
    for depth in depths:
        if args.mode == "full-window":
            runs.extend(
                run_full_window_candidate(
                    case,
                    coordinate,
                    depth,
                    threat_extension_depth=args.threat_extension_depth,
                    branch_candidate_limit=args.branch_candidate_limit,
                    candidate_trace_limit=args.candidate_trace_limit,
                    candidate_sample_limit=args.candidate_sample_limit,
                    leaf_trace_limit=args.leaf_trace_limit,
                )
                for coordinate in selected_coordinates
            )
        elif args.mode == "iterative":
            runs.append(
                run_iterative_pair(
                    case,
                    depth,
                    node_limit=args.node_limit,
                    threat_extension_depth=args.threat_extension_depth,
                    branch_candidate_limit=args.branch_candidate_limit,
                )
            )
        elif args.mode == "production":
            runs.append(
                run_production(
                    case,
                    depth,
                    time_limit_seconds=args.time_limit,
                    node_limit=args.node_limit,
                    warm_history=args.warm_history,
                    threat_extension_depth=args.threat_extension_depth,
                    branch_candidate_limit=args.branch_candidate_limit,
                )
            )
        elif args.mode == "dynamic-pair":
            runs.append(
                run_dynamic_pair(
                    case,
                    depth,
                    review_budget_seconds=args.review_budget,
                    quiet_frontier_extension=args.quiet_frontier,
                    threat_extension_depth=args.threat_extension_depth,
                    branch_candidate_limit=args.branch_candidate_limit,
                )
            )
        elif args.mode == "defense-vct":
            runs.append(
                run_defense_vct_pair(
                    case,
                    depth,
                    time_limit_seconds=(
                        args.time_limit
                        if args.time_limit is not None
                        else args.review_budget
                    ),
                )
            )
        else:
            runs.append(
                run_native_pair(
                    case,
                    depth,
                    node_limit=args.node_limit,
                    threat_extension_depth=args.threat_extension_depth,
                    branch_candidate_limit=args.branch_candidate_limit,
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
