from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time

from engine.board import BLACK, WHITE, Board
from engine.evaluator import other_side
from engine.game import format_move, parse_move
from engine.proof_search import (
    ProofBudget,
    ProofKey,
    ProofSearch,
    ProofState,
    ProofTable,
    ProofTableStats,
    ProofTTEntry,
)
from engine.threats import DefenseSet, Threat, ThreatAnalyzer, ThreatAnalyzerStats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = (
    ROOT / "tests" / "positions" / "v0175_reverse_move10_vct.json"
)
DIAGNOSTIC_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class VCTReferenceCase:
    name: str
    board_size: int
    player: int
    history: tuple[str, ...]
    candidates: tuple[str, ...]
    expected_hash: int


@dataclass(frozen=True, slots=True)
class VCTCandidateResult:
    coordinate: str
    attacker_state: str
    completed: bool
    cutoff_reason: str | None
    best_coordinate: str | None
    principal_variation: tuple[str, ...]
    required_defenses: tuple[str, ...]
    nodes: int
    transposition_hits: int
    proof_table_stats: ProofTableStats
    distinct_query_keys: int
    repeated_queries: int
    query_hotspots: VCTQueryHotspotAnalysis
    searched_attacker_moves: int
    elapsed_seconds: float
    threat_candidate_batches: int
    threat_exact_descriptions: int
    threat_frontier_batches: int
    threat_frontier_descriptions: int
    threat_analyzer_stats: ThreatAnalyzerStats
    threat_audit: VCTThreatAudit
    proof_search_audit: VCTProofSearchAudit


@dataclass(frozen=True, slots=True)
class VCTReferenceRun:
    case_name: str
    player: int
    attacker: int
    candidates: tuple[VCTCandidateResult, ...]


@dataclass(frozen=True, slots=True)
class VCTQueryHotspotBucket:
    top_keys: int
    repeated_queries: int
    repeated_query_share: float


@dataclass(frozen=True, slots=True)
class VCTQueryHotspotAnalysis:
    repeated_key_count: int
    max_query_frequency: int
    buckets: tuple[VCTQueryHotspotBucket, ...]


@dataclass(frozen=True, slots=True)
class VCTCountBucket:
    label: str
    count: int


@dataclass(frozen=True, slots=True)
class VCTThreatAudit:
    defense_set_generations: int
    complete_defense_sets: int


@dataclass(frozen=True, slots=True)
class VCTProofSearchAudit:
    and_nodes: int
    and_nodes_with_defenses: int
    available_defenses: int
    examined_defenses: int
    first_proven_loss_defense_ordinals: tuple[VCTCountBucket, ...]
    replay_attempts: int
    replay_successes: int
    replay_failure_reasons: tuple[VCTCountBucket, ...]
    budget_exhausted_and_nodes: int
    unchecked_defenses_on_budget_exhaustion: int


@dataclass(frozen=True, slots=True)
class VCTReentryPass:
    pass_index: int
    result: VCTCandidateResult
    proof_table_delta: ProofTableStats
    previous_overlap_keys: int
    previous_overlap_ratio: float
    cumulative_overlap_keys: int
    cumulative_overlap_ratio: float


@dataclass(frozen=True, slots=True)
class VCTReentryComparison:
    case_name: str
    coordinate: str
    total_node_budget: int
    nodes_per_warm_pass: int
    cold_result: VCTCandidateResult
    warm_passes: tuple[VCTReentryPass, ...]


@dataclass(slots=True)
class _AndAuditFrame:
    history_count: int
    attacker: int
    defender: int
    examined_moves: list[tuple[int, int]] = field(default_factory=list)
    first_proven_loss_move: tuple[int, int] | None = None


class _AuditedThreatAnalyzer(ThreatAnalyzer):
    """Observe exact defense classification without changing its result."""

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self.defense_set_generations = 0
        self.complete_defense_sets = 0

    def _classify_defenses(  # type: ignore[no-untyped-def]
        self,
        *args,
        **kwargs,
    ) -> DefenseSet:
        self.defense_set_generations += 1
        result = super()._classify_defenses(*args, **kwargs)
        if result.analysis_completed and result.coverage_complete:
            self.complete_defense_sets += 1
        return result

    def audit(self) -> VCTThreatAudit:
        return VCTThreatAudit(
            defense_set_generations=self.defense_set_generations,
            complete_defense_sets=self.complete_defense_sets,
        )


class _AuditedProofSearch(ProofSearch):
    """Observe AND traversal and plan replay while preserving base semantics."""

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self._and_frames: list[_AndAuditFrame] = []
        self._and_nodes = 0
        self._and_nodes_with_defenses = 0
        self._available_defenses = 0
        self._examined_defenses = 0
        self._first_loss_ordinals: Counter[int] = Counter()
        self._replay_depth = 0
        self._replay_attempts = 0
        self._replay_successes = 0
        self._replay_failure_reasons: Counter[str] = Counter()
        self._budget_exhausted_and_nodes = 0
        self._unchecked_defenses = 0

    @staticmethod
    def _mark_examined(
        frame: _AndAuditFrame,
        move: tuple[int, int],
    ) -> None:
        if move not in frame.examined_moves:
            frame.examined_moves.append(move)

    def _direct_and_frame(
        self,
        board: Board,
        *,
        attacker: int,
        side_to_move: int,
        obligation: Threat | None,
    ) -> tuple[_AndAuditFrame, tuple[int, int]] | None:
        if not self._and_frames or obligation is not None:
            return None
        frame = self._and_frames[-1]
        if side_to_move != attacker or attacker != frame.attacker:
            return None
        if len(board.move_history) != frame.history_count + 1:
            return None
        row, column, player = board.move_history[-1]
        if player != frame.defender:
            return None
        return frame, (row, column)

    def _search_node(self, board: Board, **kwargs):  # type: ignore[no-untyped-def]
        direct = self._direct_and_frame(
            board,
            attacker=kwargs["attacker"],
            side_to_move=kwargs["side_to_move"],
            obligation=kwargs["obligation"],
        )
        if direct is not None:
            frame, move = direct
            self._mark_examined(frame, move)
        result = super()._search_node(board, **kwargs)
        if (
            direct is not None
            and result.state is ProofState.PROVEN_LOSS
            and direct[0].first_proven_loss_move is None
        ):
            direct[0].first_proven_loss_move = direct[1]
        return result

    def _search_and_node(self, board: Board, **kwargs):  # type: ignore[no-untyped-def]
        frame = _AndAuditFrame(
            history_count=len(board.move_history),
            attacker=kwargs["attacker"],
            defender=kwargs["defender"],
        )
        self._and_nodes += 1
        self._and_frames.append(frame)
        try:
            result = super()._search_and_node(board, **kwargs)
        finally:
            self._and_frames.pop()

        defenses = result.required_defenses
        if (
            result.state is ProofState.PROVEN_LOSS
            and result.best_move in defenses
            and result.best_move is not None
        ):
            self._mark_examined(frame, result.best_move)
            if frame.first_proven_loss_move is None:
                frame.first_proven_loss_move = result.best_move

        cutoff_reason = self._budget_cutoff_reason()
        loop_observed = bool(frame.examined_moves) or (
            cutoff_reason is not None and bool(defenses)
        )
        if loop_observed:
            self._and_nodes_with_defenses += 1
            self._available_defenses += len(defenses)
            self._examined_defenses += len(frame.examined_moves)
            if frame.first_proven_loss_move in defenses:
                ordinal = defenses.index(frame.first_proven_loss_move) + 1
                self._first_loss_ordinals[ordinal] += 1
            unchecked = max(0, len(defenses) - len(frame.examined_moves))
            if cutoff_reason is not None and unchecked:
                self._budget_exhausted_and_nodes += 1
                self._unchecked_defenses += unchecked
        return result

    def _replay_linear_plan(  # type: ignore[no-untyped-def]
        self,
        board: Board,
        **kwargs,
    ):
        top_level = self._replay_depth == 0
        if top_level:
            self._replay_attempts += 1
            direct = self._direct_and_frame(
                board,
                attacker=kwargs["attacker"],
                side_to_move=kwargs["attacker"],
                obligation=None,
            )
            if direct is not None:
                self._mark_examined(*direct)
            cutoff_reason = self._budget_cutoff_reason()
            if cutoff_reason is not None:
                failure_reason = cutoff_reason
            elif kwargs["remaining_attacker_moves"] <= 0:
                failure_reason = "attacker_depth_limit"
            elif not kwargs["plan"]:
                failure_reason = "empty_plan"
            else:
                failure_reason = "strict_revalidation_failed"
        else:
            failure_reason = "strict_revalidation_failed"

        self._replay_depth += 1
        try:
            result = super()._replay_linear_plan(board, **kwargs)
        finally:
            self._replay_depth -= 1
        if top_level:
            if result is None:
                self._replay_failure_reasons[failure_reason] += 1
            else:
                self._replay_successes += 1
        return result

    def audit(self) -> VCTProofSearchAudit:
        return VCTProofSearchAudit(
            and_nodes=self._and_nodes,
            and_nodes_with_defenses=self._and_nodes_with_defenses,
            available_defenses=self._available_defenses,
            examined_defenses=self._examined_defenses,
            first_proven_loss_defense_ordinals=tuple(
                VCTCountBucket(label=str(ordinal), count=count)
                for ordinal, count in sorted(self._first_loss_ordinals.items())
            ),
            replay_attempts=self._replay_attempts,
            replay_successes=self._replay_successes,
            replay_failure_reasons=tuple(
                VCTCountBucket(label=reason, count=count)
                for reason, count in sorted(
                    self._replay_failure_reasons.items()
                )
            ),
            budget_exhausted_and_nodes=self._budget_exhausted_and_nodes,
            unchecked_defenses_on_budget_exhaustion=(
                self._unchecked_defenses
            ),
        )


class _TrackingProofTable(ProofTable):
    """Proof table that observes query keys without changing table semantics."""

    def __init__(self) -> None:
        super().__init__()
        self._current_query_counts: Counter[ProofKey] = Counter()

    def begin_query_trace(self) -> None:
        self._current_query_counts.clear()

    @property
    def current_query_counts(self) -> Mapping[ProofKey, int]:
        return self._current_query_counts.copy()

    def get(self, key: ProofKey) -> ProofTTEntry | None:
        self._current_query_counts[key] += 1
        return super().get(key)


def analyze_query_hotspots(
    query_counts: Mapping[object, int],
) -> VCTQueryHotspotAnalysis:
    """Summarize how concentrated repeated ProofTable queries are."""
    repeated = sorted(
        (count - 1 for count in query_counts.values() if count > 1),
        reverse=True,
    )
    total_repeated = sum(repeated)
    buckets: list[VCTQueryHotspotBucket] = []
    for limit in (1, 10, 100):
        contribution = sum(repeated[:limit])
        buckets.append(
            VCTQueryHotspotBucket(
                top_keys=limit,
                repeated_queries=contribution,
                repeated_query_share=(
                    0.0
                    if total_repeated == 0
                    else contribution / total_repeated
                ),
            )
        )
    return VCTQueryHotspotAnalysis(
        repeated_key_count=len(repeated),
        max_query_frequency=(
            0 if not query_counts else max(query_counts.values())
        ),
        buckets=tuple(buckets),
    )


def load_case(path: Path = DEFAULT_FIXTURE) -> VCTReferenceCase:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "gomoku-search-baseline-v1":
        raise ValueError(f"不支持的基线格式：{payload.get('format')!r}")
    player = int(payload["player"])
    if player not in (BLACK, WHITE):
        raise ValueError("基线行棋方必须是 BLACK 或 WHITE。")
    history = tuple(map(str, payload["history"]))
    expected_player = BLACK if len(history) % 2 == 0 else WHITE
    if player != expected_player:
        raise ValueError("基线行棋方与有序历史奇偶不一致。")
    candidates = tuple(map(str, payload["candidates"]))
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("VCT 基线候选不能为空或重复。")
    return VCTReferenceCase(
        name=str(payload["name"]),
        board_size=int(payload["board_size"]),
        player=player,
        history=history,
        candidates=candidates,
        expected_hash=int(payload["zobrist_hash"]),
    )


def build_board(case: VCTReferenceCase) -> Board:
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


def _selected_coordinates(
    case: VCTReferenceCase,
    coordinates: tuple[str, ...] | None,
) -> tuple[str, ...]:
    selected = case.candidates if coordinates is None else coordinates
    if not selected:
        raise ValueError("VCT 参考验证至少需要一个候选。")
    unknown = tuple(move for move in selected if move not in case.candidates)
    if unknown:
        raise ValueError(
            "VCT 参考候选必须来自夹具：" + ", ".join(unknown)
        )
    return selected


def _run_candidate(
    case: VCTReferenceCase,
    board: Board,
    coordinate: str,
    *,
    table: _TrackingProofTable,
    analyzer: _AuditedThreatAnalyzer,
    seconds: float,
    max_nodes: int,
    max_attacker_moves: int,
    max_quiet_frontiers: int,
    max_quiet_attacker_moves: int,
    vcf_max_attacker_moves: int,
) -> tuple[VCTCandidateResult, frozenset[ProofKey], ProofTableStats]:
    before = board_state(board)
    move = parse_move(coordinate, board.size)
    if not board.is_empty(*move):
        raise ValueError(f"候选点不是空位：{coordinate}")

    table.begin_query_trace()
    table_before = table.stats()
    search = _AuditedProofSearch(
        budget=ProofBudget.from_now(
            seconds,
            max_nodes=max_nodes,
            max_attacker_moves=max_attacker_moves,
            max_quiet_frontiers=max_quiet_frontiers,
            max_quiet_attacker_moves=max_quiet_attacker_moves,
            vcf_max_attacker_moves=vcf_max_attacker_moves,
            use_vcf_oracle=True,
            clock=time.perf_counter,
        ),
        analyzer=analyzer,
        table=table,
        clock=time.perf_counter,
    )
    proof = search.search_after_move(
        board,
        move=move,
        mover=case.player,
        attacker=other_side(case.player),
        side_to_move=other_side(case.player),
    )
    if proof.completed == (proof.state is ProofState.UNKNOWN):
        raise RuntimeError("VCT 三态与完成标记不一致。")
    if board_state(board) != before:
        raise RuntimeError("VCT 参考验证污染了棋盘或有序历史。")

    table_stats = table.stats()
    table_delta = table_stats.delta(table_before)
    if proof.transposition_hits != table_delta.hits:
        raise RuntimeError("VCT 运行命中数与 ProofTable 增量不一致。")
    query_counts = table.current_query_counts
    query_keys = frozenset(query_counts)
    query_hotspots = analyze_query_hotspots(query_counts)
    repeated_queries = table_delta.queries - len(query_keys)
    if repeated_queries < 0:
        raise RuntimeError("VCT 唯一查询 key 数超过总查询数。")
    counted_repeats = sum(count - 1 for count in query_counts.values())
    if repeated_queries != counted_repeats:
        raise RuntimeError("VCT 热点摘要与重复查询总数不一致。")

    stats = analyzer.stats()
    return (
        VCTCandidateResult(
            coordinate=coordinate,
            attacker_state=proof.state.value,
            completed=proof.completed,
            cutoff_reason=proof.cutoff_reason,
            best_coordinate=(
                None
                if proof.best_move is None
                else format_move(*proof.best_move)
            ),
            principal_variation=tuple(
                format_move(*item) for item in proof.principal_variation
            ),
            required_defenses=tuple(
                format_move(*item) for item in proof.required_defenses
            ),
            nodes=proof.nodes,
            transposition_hits=proof.transposition_hits,
            proof_table_stats=table_stats,
            distinct_query_keys=len(query_keys),
            repeated_queries=repeated_queries,
            query_hotspots=query_hotspots,
            searched_attacker_moves=proof.searched_attacker_moves,
            elapsed_seconds=proof.elapsed_seconds,
            threat_candidate_batches=stats.candidate_batches,
            threat_exact_descriptions=stats.exact_descriptions,
            threat_frontier_batches=stats.frontier_batches,
            threat_frontier_descriptions=stats.frontier_descriptions,
            threat_analyzer_stats=stats,
            threat_audit=analyzer.audit(),
            proof_search_audit=search.audit(),
        ),
        query_keys,
        table_delta,
    )


def run_reference(
    case: VCTReferenceCase,
    *,
    coordinates: tuple[str, ...] | None = None,
    seconds_per_candidate: float = 10.0,
    max_nodes: int = 100_000,
    max_attacker_moves: int = 6,
    max_quiet_frontiers: int = 16,
    max_quiet_attacker_moves: int = 2,
    vcf_max_attacker_moves: int = 6,
    candidate_limit: int = 24,
    frontier_scan_limit: int | None = 48,
) -> VCTReferenceRun:
    """Run isolated conservative proofs relative to the opponent attacker."""
    selected = _selected_coordinates(case, coordinates)
    if seconds_per_candidate <= 0:
        raise ValueError("单候选秒数必须大于 0。")

    board = build_board(case)
    attacker = other_side(case.player)
    results: list[VCTCandidateResult] = []
    for coordinate in selected:
        analyzer = _AuditedThreatAnalyzer(
            candidate_limit=candidate_limit,
            frontier_scan_limit=frontier_scan_limit,
        )
        result, _, _ = _run_candidate(
            case,
            board,
            coordinate,
            table=_TrackingProofTable(),
            analyzer=analyzer,
            seconds=seconds_per_candidate,
            max_nodes=max_nodes,
            max_attacker_moves=max_attacker_moves,
            max_quiet_frontiers=max_quiet_frontiers,
            max_quiet_attacker_moves=max_quiet_attacker_moves,
            vcf_max_attacker_moves=vcf_max_attacker_moves,
        )
        results.append(result)

    return VCTReferenceRun(
        case_name=case.name,
        player=case.player,
        attacker=attacker,
        candidates=tuple(results),
    )


def _overlap_ratio(overlap: int, current_size: int) -> float:
    return 0.0 if current_size == 0 else overlap / current_size


def run_reentry_comparison(
    case: VCTReferenceCase,
    *,
    coordinate: str,
    total_nodes: int = 200_000,
    warm_passes: int = 4,
    seconds_per_warm_pass: float = 120.0,
    max_attacker_moves: int = 6,
    max_quiet_frontiers: int = 16,
    max_quiet_attacker_moves: int = 2,
    vcf_max_attacker_moves: int = 6,
    candidate_limit: int = 24,
    frontier_scan_limit: int | None = 48,
) -> VCTReentryComparison:
    """Compare one cold proof with equal-total-node warm re-entry passes."""
    _selected_coordinates(case, (coordinate,))
    if warm_passes < 2:
        raise ValueError("暖重入至少需要 2 个 pass。")
    if total_nodes < warm_passes or total_nodes % warm_passes != 0:
        raise ValueError("总节点数必须能被暖重入 pass 数整除。")
    if seconds_per_warm_pass <= 0:
        raise ValueError("单轮秒数必须大于 0。")

    board = build_board(case)
    analyzer = _AuditedThreatAnalyzer(
        candidate_limit=candidate_limit,
        frontier_scan_limit=frontier_scan_limit,
    )
    cold_result, _, _ = _run_candidate(
        case,
        board,
        coordinate,
        table=_TrackingProofTable(),
        analyzer=analyzer,
        seconds=seconds_per_warm_pass * warm_passes,
        max_nodes=total_nodes,
        max_attacker_moves=max_attacker_moves,
        max_quiet_frontiers=max_quiet_frontiers,
        max_quiet_attacker_moves=max_quiet_attacker_moves,
        vcf_max_attacker_moves=vcf_max_attacker_moves,
    )

    nodes_per_pass = total_nodes // warm_passes
    table = _TrackingProofTable()
    previous_keys: frozenset[ProofKey] = frozenset()
    cumulative_keys: frozenset[ProofKey] = frozenset()
    passes: list[VCTReentryPass] = []
    for pass_index in range(1, warm_passes + 1):
        analyzer = _AuditedThreatAnalyzer(
            candidate_limit=candidate_limit,
            frontier_scan_limit=frontier_scan_limit,
        )
        result, query_keys, table_delta = _run_candidate(
            case,
            board,
            coordinate,
            table=table,
            analyzer=analyzer,
            seconds=seconds_per_warm_pass,
            max_nodes=nodes_per_pass,
            max_attacker_moves=max_attacker_moves,
            max_quiet_frontiers=max_quiet_frontiers,
            max_quiet_attacker_moves=max_quiet_attacker_moves,
            vcf_max_attacker_moves=vcf_max_attacker_moves,
        )
        previous_overlap = len(query_keys & previous_keys)
        cumulative_overlap = len(query_keys & cumulative_keys)
        passes.append(
            VCTReentryPass(
                pass_index=pass_index,
                result=result,
                proof_table_delta=table_delta,
                previous_overlap_keys=previous_overlap,
                previous_overlap_ratio=_overlap_ratio(
                    previous_overlap,
                    len(query_keys),
                ),
                cumulative_overlap_keys=cumulative_overlap,
                cumulative_overlap_ratio=_overlap_ratio(
                    cumulative_overlap,
                    len(query_keys),
                ),
            )
        )
        previous_keys = query_keys
        cumulative_keys = frozenset(cumulative_keys | query_keys)

    return VCTReentryComparison(
        case_name=case.name,
        coordinate=coordinate,
        total_node_budget=total_nodes,
        nodes_per_warm_pass=nodes_per_pass,
        cold_result=cold_result,
        warm_passes=tuple(passes),
    )


def _player_name(player: int) -> str:
    return "BLACK" if player == BLACK else "WHITE"


def diagnostic_payload(
    output: VCTReferenceRun | VCTReentryComparison,
) -> dict[str, object]:
    """Build the opt-in diagnostic document without touching record schemas."""
    return {
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "run_type": (
            "reentry_comparison"
            if isinstance(output, VCTReentryComparison)
            else "reference"
        ),
        "run": asdict(output),
    }


def print_run(run: VCTReferenceRun) -> None:
    print(
        f"VCT reference | case={run.case_name} | "
        f"player={_player_name(run.player)} | "
        f"attacker={_player_name(run.attacker)}"
    )
    print(
        f"{'Candidate':<10} {'Attacker state':<15} {'Complete':<9} "
        f"{'Nodes':>10} {'TT hits':>9} {'Time':>9} {'Cutoff'}"
    )
    print("-" * 88)
    for candidate in run.candidates:
        print(
            f"{candidate.coordinate:<10} "
            f"{candidate.attacker_state:<15} "
            f"{str(candidate.completed):<9} "
            f"{candidate.nodes:>10,} "
            f"{candidate.transposition_hits:>9,} "
            f"{candidate.elapsed_seconds:>8.3f}s "
            f"{candidate.cutoff_reason or '-'}"
        )
        stats = candidate.proof_table_stats
        print(
            "  Proof TT: "
            f"queries={stats.queries:,} hits={stats.hits:,} "
            f"compatible={stats.compatible_hits:,} "
            f"stores={stats.stores:,} skipped={stats.skipped_stores:,} "
            f"evictions={stats.evictions:,} size={stats.size:,}"
        )
        print(
            "  Hints: "
            f"queries={stats.hint_queries:,} hits={stats.hint_hits:,} "
            f"stores={stats.hint_stores:,} "
            f"evictions={stats.hint_evictions:,} "
            f"size={stats.hint_size:,}"
        )
        threat_stats = candidate.threat_analyzer_stats
        print(
            "  Threat cache: "
            f"queries={threat_stats.cache_queries:,} "
            f"hits={threat_stats.cache_hits:,} "
            f"stores={threat_stats.cache_stores:,} "
            f"skips={threat_stats.cache_skips:,}"
        )
        print(
            "  Threat work: "
            f"candidate_batches={threat_stats.candidate_batches:,} "
            f"exact_descriptions={threat_stats.exact_descriptions:,} "
            f"frontier_batches={threat_stats.frontier_batches:,} "
            f"frontier_descriptions={threat_stats.frontier_descriptions:,}"
        )
        threat_audit = candidate.threat_audit
        print(
            "  Defense sets: "
            f"generated={threat_audit.defense_set_generations:,} "
            f"complete={threat_audit.complete_defense_sets:,}"
        )
        proof_audit = candidate.proof_search_audit
        first_losses = ",".join(
            f"{item.label}:{item.count}"
            for item in proof_audit.first_proven_loss_defense_ordinals
        ) or "-"
        replay_failures = ",".join(
            f"{item.label}:{item.count}"
            for item in proof_audit.replay_failure_reasons
        ) or "-"
        print(
            "  AND audit: "
            f"nodes={proof_audit.and_nodes:,} "
            f"with_defenses={proof_audit.and_nodes_with_defenses:,} "
            f"examined={proof_audit.examined_defenses:,}/"
            f"{proof_audit.available_defenses:,} "
            f"first_loss_ordinals={first_losses}"
        )
        print(
            "  Replay audit: "
            f"attempts={proof_audit.replay_attempts:,} "
            f"successes={proof_audit.replay_successes:,} "
            f"failures={replay_failures} "
            f"budget_nodes={proof_audit.budget_exhausted_and_nodes:,} "
            f"unchecked={proof_audit.unchecked_defenses_on_budget_exhaustion:,}"
        )
        print(
            "  Query keys: "
            f"distinct={candidate.distinct_query_keys:,} "
            f"repeated={candidate.repeated_queries:,}"
        )
        hotspots = candidate.query_hotspots
        shares = " ".join(
            f"top{bucket.top_keys}={bucket.repeated_query_share:.3%}"
            for bucket in hotspots.buckets
        )
        print(
            "  Hotspots: "
            f"repeated_keys={hotspots.repeated_key_count:,} "
            f"max_frequency={hotspots.max_query_frequency:,} "
            f"{shares}"
        )
        if candidate.principal_variation:
            print("  PV: " + " ".join(candidate.principal_variation))


def print_reentry_comparison(comparison: VCTReentryComparison) -> None:
    cold = comparison.cold_result
    print(
        f"VCT re-entry | case={comparison.case_name} | "
        f"candidate={comparison.coordinate} | "
        f"total_nodes={comparison.total_node_budget:,}"
    )
    print(
        "Cold: "
        f"state={cold.attacker_state} nodes={cold.nodes:,} "
        f"distinct={cold.distinct_query_keys:,} "
        f"repeated={cold.repeated_queries:,} "
        f"time={cold.elapsed_seconds:.3f}s "
        f"cutoff={cold.cutoff_reason or '-'}"
    )
    print(
        "Cold hotspots: "
        f"repeated_keys={cold.query_hotspots.repeated_key_count:,} "
        f"max_frequency={cold.query_hotspots.max_query_frequency:,} "
        + " ".join(
            f"top{bucket.top_keys}={bucket.repeated_query_share:.3%}"
            for bucket in cold.query_hotspots.buckets
        )
    )
    print(
        f"{'Pass':>4} {'State':<12} {'Nodes':>9} {'Unique':>9} "
        f"{'Repeat':>9} {'Prev':>8} {'Prior':>8} "
        f"{'TT':>6} {'Hint':>6} {'Time':>9} {'Cutoff'}"
    )
    print("-" * 112)
    for item in comparison.warm_passes:
        result = item.result
        delta = item.proof_table_delta
        print(
            f"{item.pass_index:>4} "
            f"{result.attacker_state:<12} "
            f"{result.nodes:>9,} "
            f"{result.distinct_query_keys:>9,} "
            f"{result.repeated_queries:>9,} "
            f"{item.previous_overlap_ratio:>7.1%} "
            f"{item.cumulative_overlap_ratio:>7.1%} "
            f"{delta.hits:>6,} "
            f"{delta.hint_hits:>6,} "
            f"{result.elapsed_seconds:>8.3f}s "
            f"{result.cutoff_reason or '-'}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="独立、保守的候选后 VCT 三态参考验证器。"
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--max-nodes", type=int, default=100_000)
    parser.add_argument("--max-attacker-moves", type=int, default=6)
    parser.add_argument("--max-quiet-frontiers", type=int, default=16)
    parser.add_argument("--max-quiet-attacker-moves", type=int, default=2)
    parser.add_argument("--vcf-max-attacker-moves", type=int, default=6)
    parser.add_argument("--candidate-limit", type=int, default=24)
    parser.add_argument("--frontier-scan-limit", type=int, default=48)
    parser.add_argument("--reentry-passes", type=int, default=0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.max_nodes < 0:
        parser.error("--max-nodes 不能小于 0")
    case = load_case(args.fixture)
    if args.reentry_passes:
        if args.candidate is None or len(args.candidate) != 1:
            parser.error("暖重入比较必须且只能指定一个 --candidate")
        output: VCTReferenceRun | VCTReentryComparison
        output = run_reentry_comparison(
            case,
            coordinate=args.candidate[0],
            total_nodes=args.max_nodes,
            warm_passes=args.reentry_passes,
            seconds_per_warm_pass=args.seconds,
            max_attacker_moves=args.max_attacker_moves,
            max_quiet_frontiers=args.max_quiet_frontiers,
            max_quiet_attacker_moves=args.max_quiet_attacker_moves,
            vcf_max_attacker_moves=args.vcf_max_attacker_moves,
            candidate_limit=args.candidate_limit,
            frontier_scan_limit=args.frontier_scan_limit,
        )
        print_reentry_comparison(output)
    else:
        output = run_reference(
            case,
            coordinates=(
                None if args.candidate is None else tuple(args.candidate)
            ),
            seconds_per_candidate=args.seconds,
            max_nodes=args.max_nodes,
            max_attacker_moves=args.max_attacker_moves,
            max_quiet_frontiers=args.max_quiet_frontiers,
            max_quiet_attacker_moves=args.max_quiet_attacker_moves,
            vcf_max_attacker_moves=args.vcf_max_attacker_moves,
            candidate_limit=args.candidate_limit,
            frontier_scan_limit=args.frontier_scan_limit,
        )
        print_run(output)
    if args.json is not None:
        args.json.write_text(
            json.dumps(
                diagnostic_payload(output),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
