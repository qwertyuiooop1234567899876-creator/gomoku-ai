from __future__ import annotations

import time
from dataclasses import dataclass, replace
from enum import Enum

from engine.ai import (
    CandidateAnalysis,
    DecisionAnalysis,
    DefenseCandidateAnalysis,
    Move,
    ScoringAI,
)
from engine.board import DIRECTIONS, EMPTY, WHITE, Board
from engine.evaluator import (
    ThreatProfile,
    analyze_move_threats,
    evaluate_board,
    evaluate_move,
    evaluate_player,
    find_winning_moves,
    other_side,
)
from engine.time_manager import TimeManager
from engine.zobrist import get_zobrist_table

MATE_SCORE = 1_000_000_000
INFINITY = MATE_SCORE * 2


class SearchTimeout(RuntimeError):
    """搜索超过硬时间预算。"""


class VCFTimeout(RuntimeError):
    """VCF 子搜索耗尽自己的时间份额。"""


class BoundType(str, Enum):
    EXACT = "exact"
    LOWER = "lower"
    UPPER = "upper"


class DefenseProof(str, Enum):
    """防守分支探针在有限搜索范围内给出的状态。"""

    SURVIVES_PROBE = "survives_probe"
    UNKNOWN = "unknown"
    FORCED_LOSS = "forced_loss"


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """V0.8.5 搜索参数。"""

    max_depth: int = 3
    time_limit_seconds: float | None = 2.0
    root_candidate_limit: int = 12
    branch_candidate_limit: int = 8
    preselection_factor: int = 3
    candidate_radius: int = 2
    root_candidate_radius: int = 3
    recent_move_count: int = 4
    threat_extension_depth: int = 2
    use_transposition_table: bool = True
    transposition_max_entries: int = 100_000
    use_pvs: bool = True
    use_aspiration: bool = True
    aspiration_window: int = 100_000
    soft_time_ratio: float = 0.88
    vcf_max_attacker_moves: int = 5
    vcf_time_fraction: float = 0.18
    frontier_reply_limit: int = 6
    defense_vct_max_candidates: int = 3
    defense_vct_probe_depth: int = 3
    defense_vct_extension_depth: int = 4
    defense_vct_branch_limit: int = 6
    defense_vct_time_fraction: float = 0.12
    defense_vct_max_seconds: float = 2.5
    defense_vct_score_margin: int = 20_000

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth 必须大于 0。")
        if (
            self.time_limit_seconds is not None
            and self.time_limit_seconds <= 0
        ):
            raise ValueError("time_limit_seconds 必须大于 0 或为 None。")
        if self.root_candidate_limit < 1:
            raise ValueError("root_candidate_limit 必须大于 0。")
        if self.branch_candidate_limit < 1:
            raise ValueError("branch_candidate_limit 必须大于 0。")
        if self.preselection_factor < 1:
            raise ValueError("preselection_factor 必须大于 0。")
        if self.candidate_radius < 1:
            raise ValueError("candidate_radius 必须大于 0。")
        if self.root_candidate_radius < self.candidate_radius:
            raise ValueError("root_candidate_radius 不能小于 candidate_radius。")
        if self.recent_move_count < 1:
            raise ValueError("recent_move_count 必须大于 0。")
        if self.threat_extension_depth < 0:
            raise ValueError("threat_extension_depth 不能小于 0。")
        if self.transposition_max_entries < 1_000:
            raise ValueError("transposition_max_entries 不能小于 1000。")
        if self.aspiration_window < 1:
            raise ValueError("aspiration_window 必须大于 0。")
        if not 0.5 <= self.soft_time_ratio < 1.0:
            raise ValueError("soft_time_ratio 必须在 0.5～1.0 之间。")
        if self.vcf_max_attacker_moves < 0:
            raise ValueError("vcf_max_attacker_moves 不能小于 0。")
        if not 0 < self.vcf_time_fraction <= 1:
            raise ValueError("vcf_time_fraction 必须在 0～1 之间。")
        if self.frontier_reply_limit < 2:
            raise ValueError("frontier_reply_limit 不能小于 2。")
        if self.defense_vct_max_candidates < 2:
            raise ValueError("defense_vct_max_candidates 不能小于 2。")
        if self.defense_vct_probe_depth < 1:
            raise ValueError("defense_vct_probe_depth 必须大于 0。")
        if self.defense_vct_extension_depth < 1:
            raise ValueError("defense_vct_extension_depth 必须大于 0。")
        if self.defense_vct_branch_limit < 2:
            raise ValueError("defense_vct_branch_limit 不能小于 2。")
        if not 0 < self.defense_vct_time_fraction <= 1:
            raise ValueError("defense_vct_time_fraction 必须在 0～1 之间。")
        if self.defense_vct_max_seconds <= 0:
            raise ValueError("defense_vct_max_seconds 必须大于 0。")
        if self.defense_vct_score_margin < 0:
            raise ValueError("defense_vct_score_margin 不能小于 0。")


@dataclass(slots=True)
class SearchCounters:
    nodes: int = 0
    cutoffs: int = 0
    transposition_hits: int = 0
    transposition_cutoffs: int = 0
    killer_hits: int = 0
    history_hits: int = 0
    extensions: int = 0
    pvs_researches: int = 0
    aspiration_researches: int = 0
    vcf_nodes: int = 0
    defense_vct_nodes: int = 0


@dataclass(frozen=True, slots=True)
class TTEntry:
    depth: int
    extension_depth: int
    score: int
    bound: BoundType
    best_move: Move | None
    principal_variation: tuple[Move, ...]
    generation: int


@dataclass(frozen=True, slots=True)
class RootResult:
    move: Move
    score: int
    principal_variation: tuple[Move, ...]
    ranked_moves: tuple[tuple[Move, int], ...]
    ranked_variations: tuple[
        tuple[Move, int, tuple[Move, ...]], ...
    ] = ()


@dataclass(frozen=True, slots=True)
class DefenseProbeResult:
    """一次窄分支防守 VCT 探测结果。"""

    completed_depth: int
    nodes: int
    candidates: tuple[DefenseCandidateAnalysis, ...]

    @property
    def best_move(self) -> Move | None:
        return self.candidates[0].move if self.candidates else None


class SearchAI(ScoringAI):
    """
    V0.8.5 搜索 AI。

    保留每个 SearchAI 独立的 100,000 条置换表。多重威胁前沿检测
    只负责把 G9 一类危险启动点提升到根节点候选前列，不再凭静态
    评分直接落子；最终选择重新交给 PVS/迭代加深，避免多个疑似
    前沿点分数接近时过早返回。V0.8.4 进一步禁止把普通 PVS 中的
    “接近将杀分”当作严格证明：除立即五连、唯一封堵、多胜点败势
    和独立 VCF 快速通道外，搜索必须继续到请求深度或时间边界。
    V0.8.5 在对手强威胁只有 2～3 个封堵候选时增加窄分支
    defense-VCT 探针：用更深的威胁延伸比较不同封堵端点，并在
    普通 PVS 分数接近时作为战术优先级裁决，处理 L7/H11 一类
    “都能挡住眼前棋形，但其中一边会放出长 VCT”的局面。
    """

    def __init__(
        self,
        player: int = WHITE,
        *,
        max_depth: int = 3,
        time_limit_seconds: float | None = 2.0,
        root_candidate_limit: int = 12,
        branch_candidate_limit: int = 8,
        threat_extension_depth: int = 2,
        diagnostics: bool = False,
        top_n: int = 5,
    ) -> None:
        super().__init__(
            player=player,
            diagnostics=diagnostics,
            top_n=top_n,
        )
        self.config = SearchConfig(
            max_depth=max_depth,
            time_limit_seconds=time_limit_seconds,
            root_candidate_limit=root_candidate_limit,
            branch_candidate_limit=branch_candidate_limit,
            threat_extension_depth=threat_extension_depth,
        )
        self._time = TimeManager.start(None)
        self._counters = SearchCounters()
        self._transposition_table: dict[int, TTEntry] = {}
        self._generation = 0
        self._killer_moves: dict[int, list[Move]] = {}
        self._history_scores: dict[tuple[int, int, int], int] = {}
        self._interrupted_depth = 0
        self._vcf_deadline: float | None = None
        self._threat_cache: dict[
            tuple[int, int, int, int],
            ThreatProfile,
        ] = {}
        self._defense_probe: DefenseProbeResult | None = None

    def choose_move(self, board: Board) -> Move:
        """按硬战术、VCF、限时迭代加深的顺序选择落点。"""
        self.last_analysis = None
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            raise ValueError("棋盘已满，电脑无法落子。")

        self._time = TimeManager.start(
            self.config.time_limit_seconds,
            soft_ratio=self.config.soft_time_ratio,
        )
        self._counters = SearchCounters()
        self._generation += 1
        self._killer_moves.clear()
        self._interrupted_depth = 0
        self._threat_cache.clear()
        self._defense_probe = None
        self._decay_history()
        self._prune_transposition_table()

        # 先准备一个绝不需要完整评价的合法回退点。
        fallback_move = self._quick_fallback(board, legal_moves, self.player)

        own_wins = self._timed_winning_moves(
            board,
            self.player,
            legal_moves,
        )
        if own_wins:
            selected = own_wins[0]
            self._save_search_analysis(
                selected_move=selected,
                reason="立即五连",
                candidate_count=len(legal_moves),
                ranked_moves=[(move, MATE_SCORE) for move in own_wins],
                completed_depth=0,
                principal_variation=(selected,),
                search_completed=True,
                stop_reason="immediate_win",
            )
            return selected

        opponent_wins = self._timed_winning_moves(
            board,
            self.opponent,
            legal_moves,
        )
        if len(opponent_wins) == 1:
            selected = opponent_wins[0]
            self._save_search_analysis(
                selected_move=selected,
                reason="封堵唯一胜点",
                candidate_count=len(legal_moves),
                ranked_moves=[(selected, MATE_SCORE // 2)],
                completed_depth=0,
                principal_variation=(selected,),
                search_completed=True,
                stop_reason="unique_block",
            )
            return selected

        if len(opponent_wins) >= 2:
            selected = self._quick_fallback(
                board,
                opponent_wins,
                self.player,
            )
            ranked = [
                (move, self._quick_order_score(board, move, self.player))
                for move in opponent_wins
            ]
            ranked.sort(key=lambda item: item[1], reverse=True)
            self._save_search_analysis(
                selected_move=selected,
                reason="对手多胜点：局面已属强制败势",
                candidate_count=len(legal_moves),
                ranked_moves=ranked,
                completed_depth=0,
                principal_variation=(selected,),
                search_completed=True,
                stop_reason="multiple_loss_points",
            )
            return selected

        if not board.move_history:
            center = board.size // 2
            selected = (center, center)
            self._save_search_analysis(
                selected_move=selected,
                reason="空棋盘选择天元",
                candidate_count=len(legal_moves),
                ranked_moves=[(selected, 0)],
                completed_depth=0,
                principal_variation=(selected,),
                search_completed=True,
                stop_reason="opening",
            )
            return selected

        # VCF 只领取小部分预算；超时后直接回到主搜索，不影响合法着法。
        if (
            self.config.vcf_max_attacker_moves > 0
            and self._stone_count(board, self.player) >= 3
        ):
            try:
                vcf_line = self._find_vcf(board, self.player)
            except VCFTimeout:
                vcf_line = None
            if vcf_line:
                selected = vcf_line[0]
                self._save_search_analysis(
                    selected_move=selected,
                    reason=f"VCF 强制胜势（{len(vcf_line)} 手变化）",
                    candidate_count=1,
                    ranked_moves=[
                        (selected, MATE_SCORE - len(vcf_line))
                    ],
                    completed_depth=0,
                    principal_variation=vcf_line,
                    search_completed=True,
                    vcf_found=True,
                    vcf_depth=len(vcf_line),
                    stop_reason="vcf_proven",
                )
                return selected

        preserve_frontier_order = False
        defense_probe: DefenseProbeResult | None = None

        try:
            root_pool = self._root_profile_pool(board, legal_moves)
            own_profiles = self._profile_moves_timed(
                board,
                root_pool,
                self.player,
            )
            opponent_profiles = self._profile_moves_timed(
                board,
                root_pool,
                self.opponent,
            )

            own_forcing_moves = [
                move
                for move, profile in own_profiles.items()
                if profile.forced_win
            ]
            opponent_forcing_moves = [
                move
                for move, profile in opponent_profiles.items()
                if profile.forced_win
            ]
            opponent_frontiers = self._multi_threat_frontiers(
                board,
                root_pool,
                self.opponent,
                profiles=opponent_profiles,
            )
            if opponent_frontiers:
                maximum_frontier_size = max(
                    len(replies)
                    for replies in opponent_frontiers.values()
                )
                opponent_frontier_moves = [
                    move
                    for move, replies in opponent_frontiers.items()
                    if len(replies) == maximum_frontier_size
                ]
            else:
                opponent_frontier_moves = []

            if own_forcing_moves:
                search_candidates = self._order_specific_moves(
                    board,
                    own_forcing_moves,
                    self.player,
                    ply=0,
                    tt_move=self._tt_best_move(board, self.player),
                    full_evaluation=True,
                )
                tactical_reason = "搜索自身强制威胁的最佳变化"
            elif opponent_forcing_moves:
                search_candidates = self._order_specific_moves(
                    board,
                    opponent_forcing_moves,
                    self.player,
                    ply=0,
                    tt_move=self._tt_best_move(board, self.player),
                    full_evaluation=True,
                )
                if 2 <= len(search_candidates) <= (
                    self.config.defense_vct_max_candidates
                ):
                    defense_probe = self._run_defense_vct_probe(
                        board,
                        self.player,
                        search_candidates,
                    )
                    if defense_probe is not None:
                        self._defense_probe = defense_probe
                        probe_order = [
                            candidate.move
                            for candidate in defense_probe.candidates
                        ]
                        search_candidates = [
                            *probe_order,
                            *(
                                move
                                for move in search_candidates
                                if move not in probe_order
                            ),
                        ]
                        tactical_reason = (
                            "防守分支 VCT 探针与 PVS 联合验证"
                        )
                    else:
                        tactical_reason = "搜索对手强制威胁的最佳防守"
                else:
                    tactical_reason = "搜索对手强制威胁的最佳防守"
            elif opponent_frontier_moves:
                # V0.8.2 在这里直接按静态分返回，导致多个前沿点分数
                # 接近时（例如 K9/J7/H6/J10）没有真正验证对手最佳应手。
                # V0.8.3 将前沿检测降级为“根节点排序提示”：危险点
                # 必须进入候选集并优先搜索，但最终着法由 PVS 决定。
                frontier_candidates = self._order_specific_moves(
                    board,
                    opponent_frontier_moves,
                    self.player,
                    ply=0,
                    tt_move=self._tt_best_move(board, self.player),
                    full_evaluation=True,
                )
                search_candidates = frontier_candidates[
                    : self.config.frontier_reply_limit
                ]
                preserve_frontier_order = True
                tactical_reason = "多重威胁启动点候选的 PVS 防守"
            else:
                search_candidates = self._ordered_moves(
                    board,
                    self.player,
                    at_root=True,
                    ply=0,
                )
                tactical_reason = "PVS 搜索最佳变化"
        except SearchTimeout:
            self._interrupted_depth = 0
            self._save_search_analysis(
                selected_move=fallback_move,
                reason="硬时间到达，使用快速候选回退",
                candidate_count=len(legal_moves),
                ranked_moves=[
                    (
                        fallback_move,
                        self._quick_order_score(
                            board,
                            fallback_move,
                            self.player,
                        ),
                    )
                ],
                completed_depth=0,
                principal_variation=(fallback_move,),
                search_completed=False,
                stop_reason="hard_deadline_fallback",
            )
            return fallback_move

        if not search_candidates:
            search_candidates = [fallback_move]

        fallback_move = search_candidates[0]
        fallback_score = self._quick_order_score(
            board,
            fallback_move,
            self.player,
        )
        best_result = RootResult(
            move=fallback_move,
            score=fallback_score,
            principal_variation=(fallback_move,),
            ranked_moves=((fallback_move, fallback_score),),
        )
        completed_depth = 0
        search_completed = True
        stop_reason = "requested_depth_completed"

        for depth in range(1, self.config.max_depth + 1):
            if depth > 1 and self._time.soft_expired():
                self._interrupted_depth = depth
                search_completed = False
                stop_reason = "soft_deadline"
                break

            try:
                if (
                    self.config.use_aspiration
                    and completed_depth > 0
                    and abs(best_result.score) < MATE_SCORE - 10_000
                ):
                    window = max(
                        self.config.aspiration_window,
                        abs(best_result.score) // 4 + 10_000,
                    )
                    low = best_result.score - window
                    high = best_result.score + window
                    result = self._search_root(
                        board,
                        self.player,
                        depth,
                        search_candidates,
                        alpha=low,
                        beta=high,
                    )
                    if result.score <= low or result.score >= high:
                        self._counters.aspiration_researches += 1
                        result = self._search_root(
                            board,
                            self.player,
                            depth,
                            search_candidates,
                            alpha=-INFINITY,
                            beta=INFINITY,
                        )
                else:
                    result = self._search_root(
                        board,
                        self.player,
                        depth,
                        search_candidates,
                        alpha=-INFINITY,
                        beta=INFINITY,
                    )
            except SearchTimeout:
                self._interrupted_depth = depth
                search_completed = False
                stop_reason = "hard_deadline"
                break

            if defense_probe is not None:
                result = self._apply_defense_probe_tiebreak(
                    result,
                    defense_probe,
                )

            best_result = result
            completed_depth = depth

            # 普通搜索下一层首先验证上一层最佳着。多重威胁前沿
            # 保留静态危险度顺序，避免浅层误判把真正关键点永久后移。
            if not preserve_frontier_order:
                search_candidates = self._promote_move(
                    search_candidates,
                    result.move,
                )

            # V0.8.3 会在分数接近 MATE_SCORE 时提前停止。但普通
            # PVS 使用的是选择性候选和威胁延伸，这类分数可能只是
            # 强棋型或受限搜索中的“疑似必杀”，并非严格证明。
            # 因此这里不再按分数退出；只有 choose_move 前面的明确
            # 战术通道和独立 VCF 能提前返回。

        if completed_depth < self.config.max_depth:
            search_completed = False
            if self._interrupted_depth == 0:
                self._interrupted_depth = completed_depth + 1
                stop_reason = "search_incomplete"

        reason = (
            f"{tactical_reason}（完成深度 {completed_depth}）"
            if completed_depth > 0
            else f"{tactical_reason}（时间不足，使用快速回退）"
        )
        self._save_search_analysis(
            selected_move=best_result.move,
            reason=reason,
            candidate_count=len(search_candidates),
            ranked_moves=list(best_result.ranked_moves),
            completed_depth=completed_depth,
            principal_variation=best_result.principal_variation,
            search_completed=search_completed,
            own_profiles=own_profiles,
            opponent_profiles=opponent_profiles,
            stop_reason=stop_reason,
        )
        return best_result.move

    def _run_defense_vct_probe(
        self,
        board: Board,
        player: int,
        candidates: list[Move],
    ) -> DefenseProbeResult | None:
        """用窄分支、更深威胁延伸比较少量防守候选。

        这不是完整 VCT 证明器。它只在根节点防守候选很少时运行，
        并把更多预算放到威胁延伸上，用来识别“眼前都能挡，但某个
        封堵端点会放出更长威胁链”的差异。结果只作为普通 PVS
        分数接近时的战术裁决，不替代完整搜索。
        """
        budget = self._defense_vct_budget_seconds()
        if budget == 0:
            return None

        probe = SearchAI(
            player=player,
            max_depth=self.config.defense_vct_probe_depth,
            time_limit_seconds=budget,
            root_candidate_limit=max(2, len(candidates)),
            branch_candidate_limit=self.config.defense_vct_branch_limit,
            threat_extension_depth=self.config.defense_vct_extension_depth,
            diagnostics=False,
        )
        probe.config = replace(
            probe.config,
            use_aspiration=False,
            use_pvs=True,
        )
        probe._time = TimeManager.start(
            budget,
            soft_ratio=0.96,
        )
        probe._generation += 1
        probe._counters = SearchCounters()

        ordered = list(candidates)
        best: RootResult | None = None
        completed_depth = 0

        for depth in range(1, self.config.defense_vct_probe_depth + 1):
            if depth > 1 and probe._time.soft_expired():
                break
            try:
                result = probe._search_root(
                    board,
                    player,
                    depth,
                    ordered,
                    alpha=-INFINITY,
                    beta=INFINITY,
                )
            except SearchTimeout:
                break
            best = result
            completed_depth = depth
            ordered = probe._promote_move(ordered, result.move)

        self._counters.defense_vct_nodes += probe._counters.nodes
        if (
            best is None
            or completed_depth < self.config.defense_vct_probe_depth
        ):
            return None

        variations = {
            move: (score, pv)
            for move, score, pv in best.ranked_variations
        }
        analyses: list[DefenseCandidateAnalysis] = []
        for move in candidates:
            score, pv = variations.get(
                move,
                (
                    dict(best.ranked_moves).get(move, -INFINITY),
                    (move,),
                ),
            )
            if score <= -MATE_SCORE + 10_000:
                status = DefenseProof.FORCED_LOSS
            elif completed_depth <= 0:
                status = DefenseProof.UNKNOWN
            else:
                status = DefenseProof.SURVIVES_PROBE
            analyses.append(
                DefenseCandidateAnalysis(
                    move=move,
                    score=score,
                    status=status.value,
                    principal_variation=pv,
                )
            )

        status_priority = {
            DefenseProof.SURVIVES_PROBE.value: 2,
            DefenseProof.UNKNOWN.value: 1,
            DefenseProof.FORCED_LOSS.value: 0,
        }
        original_priority = {
            move: len(candidates) - index
            for index, move in enumerate(candidates)
        }
        analyses.sort(
            key=lambda item: (
                status_priority.get(item.status, 0),
                item.score,
                original_priority.get(item.move, 0),
            ),
            reverse=True,
        )
        return DefenseProbeResult(
            completed_depth=completed_depth,
            nodes=probe._counters.nodes,
            candidates=tuple(analyses),
        )

    def _defense_vct_budget_seconds(self) -> float | None:
        total = self.config.time_limit_seconds
        if total is None:
            return None

        remaining = self._time.remaining_seconds
        if remaining is None:
            return None

        budget = min(
            self.config.defense_vct_max_seconds,
            max(0.15, total * self.config.defense_vct_time_fraction),
        )
        budget = min(budget, max(0.0, remaining - 0.05))
        return budget if budget >= 0.05 else 0

    def _apply_defense_probe_tiebreak(
        self,
        result: RootResult,
        probe: DefenseProbeResult,
    ) -> RootResult:
        """PVS 分数接近时，优先采用更深威胁探针的防守排序。"""
        root_scores = dict(result.ranked_moves)
        available = [
            candidate
            for candidate in probe.candidates
            if candidate.move in root_scores
        ]
        if not available:
            return result

        status_priority = {
            DefenseProof.SURVIVES_PROBE.value: 2,
            DefenseProof.UNKNOWN.value: 1,
            DefenseProof.FORCED_LOSS.value: 0,
        }
        best_status = max(
            status_priority.get(candidate.status, 0)
            for candidate in available
        )
        status_filtered = [
            candidate
            for candidate in available
            if status_priority.get(candidate.status, 0) == best_status
        ]
        best_root_score = max(
            root_scores[candidate.move]
            for candidate in status_filtered
        )
        eligible = [
            candidate
            for candidate in status_filtered
            if root_scores[candidate.move]
            >= best_root_score - self.config.defense_vct_score_margin
        ]
        chosen = eligible[0]
        if chosen.move == result.move:
            return result

        variation_map = {
            move: pv
            for move, _, pv in result.ranked_variations
        }
        chosen_pv = variation_map.get(
            chosen.move,
            chosen.principal_variation or (chosen.move,),
        )
        reordered = (
            (chosen.move, root_scores[chosen.move]),
            *(
                item
                for item in result.ranked_moves
                if item[0] != chosen.move
            ),
        )
        reordered_variations = (
            (chosen.move, root_scores[chosen.move], chosen_pv),
            *(
                item
                for item in result.ranked_variations
                if item[0] != chosen.move
            ),
        )
        return RootResult(
            move=chosen.move,
            score=root_scores[chosen.move],
            principal_variation=chosen_pv,
            ranked_moves=reordered,
            ranked_variations=reordered_variations,
        )

    def _search_root(
        self,
        board: Board,
        player: int,
        depth: int,
        candidates: list[Move],
        *,
        alpha: int,
        beta: int,
    ) -> RootResult:
        alpha_original = alpha
        ranked: list[tuple[Move, int, tuple[Move, ...]]] = []
        tt_move = self._tt_best_move(board, player)
        moves = self._promote_move(candidates, tt_move)

        for index, move in enumerate(moves):
            self._check_timeout()
            board.place(move[0], move[1], player)
            try:
                if board.check_win(move[0], move[1]):
                    score = MATE_SCORE
                    child_pv: tuple[Move, ...] = ()
                elif self.config.use_pvs and index > 0:
                    child_score, child_pv = self._negamax(
                        board,
                        other_side(player),
                        depth - 1,
                        -alpha - 1,
                        -alpha,
                        ply=1,
                        extension_depth=self.config.threat_extension_depth,
                    )
                    score = -child_score
                    if alpha < score < beta:
                        self._counters.pvs_researches += 1
                        child_score, child_pv = self._negamax(
                            board,
                            other_side(player),
                            depth - 1,
                            -beta,
                            -alpha,
                            ply=1,
                            extension_depth=self.config.threat_extension_depth,
                        )
                        score = -child_score
                else:
                    child_score, child_pv = self._negamax(
                        board,
                        other_side(player),
                        depth - 1,
                        -beta,
                        -alpha,
                        ply=1,
                        extension_depth=self.config.threat_extension_depth,
                    )
                    score = -child_score
            finally:
                board.undo()

            pv = (move, *child_pv)
            ranked.append((move, score, pv))
            alpha = max(alpha, score)

            if alpha >= beta:
                self._counters.cutoffs += 1
                break

        if not ranked:
            raise SearchTimeout

        candidate_priority = {
            move: len(candidates) - index
            for index, move in enumerate(candidates)
        }
        ranked.sort(
            key=lambda item: (
                item[1],
                candidate_priority.get(item[0], 0),
            ),
            reverse=True,
        )
        best_move, best_score, best_pv = ranked[0]

        # Aspiration fail-low 时保留一个明确分数；外层会全窗口重搜。
        if best_score <= alpha_original:
            best_score = min(best_score, alpha_original)

        return RootResult(
            move=best_move,
            score=best_score,
            principal_variation=best_pv,
            ranked_moves=tuple(
                (move, score)
                for move, score, _ in ranked
            ),
            ranked_variations=tuple(ranked),
        )

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
    ) -> tuple[int, tuple[Move, ...]]:
        self._counters.nodes += 1
        self._check_timeout()

        if board.is_full():
            return 0, ()

        alpha_original = alpha
        beta_original = beta
        key = self._position_key(board, player)
        tt_move: Move | None = None

        if self.config.use_transposition_table:
            entry = self._transposition_table.get(key)
            if entry is not None:
                tt_move = entry.best_move
                if (
                    entry.depth >= depth
                    and entry.extension_depth >= extension_depth
                ):
                    self._counters.transposition_hits += 1
                    if entry.bound == BoundType.EXACT:
                        return entry.score, entry.principal_variation
                    if entry.bound == BoundType.LOWER:
                        alpha = max(alpha, entry.score)
                    elif entry.bound == BoundType.UPPER:
                        beta = min(beta, entry.score)
                    if alpha >= beta:
                        self._counters.transposition_cutoffs += 1
                        return entry.score, entry.principal_variation

        if depth <= 0:
            score, pv = self._threat_extension(
                board,
                player,
                alpha,
                beta,
                ply=ply,
                extension_depth=extension_depth,
            )
            self._store_tt(
                key,
                depth,
                extension_depth,
                score,
                alpha_original,
                beta_original,
                pv,
                pv[0] if pv else None,
            )
            return score, pv

        moves = self._ordered_moves(
            board,
            player,
            at_root=False,
            ply=ply,
            tt_move=tt_move,
        )
        if not moves:
            return self._static_score(board, player), ()

        best_score = -INFINITY
        best_move: Move | None = None
        best_pv: tuple[Move, ...] = ()

        for index, move in enumerate(moves):
            board.place(move[0], move[1], player)
            try:
                if board.check_win(move[0], move[1]):
                    score = MATE_SCORE - ply
                    child_pv: tuple[Move, ...] = ()
                elif self.config.use_pvs and index > 0:
                    child_score, child_pv = self._negamax(
                        board,
                        other_side(player),
                        depth - 1,
                        -alpha - 1,
                        -alpha,
                        ply=ply + 1,
                        extension_depth=extension_depth,
                    )
                    score = -child_score
                    if alpha < score < beta:
                        self._counters.pvs_researches += 1
                        child_score, child_pv = self._negamax(
                            board,
                            other_side(player),
                            depth - 1,
                            -beta,
                            -alpha,
                            ply=ply + 1,
                            extension_depth=extension_depth,
                        )
                        score = -child_score
                else:
                    child_score, child_pv = self._negamax(
                        board,
                        other_side(player),
                        depth - 1,
                        -beta,
                        -alpha,
                        ply=ply + 1,
                        extension_depth=extension_depth,
                    )
                    score = -child_score
            finally:
                board.undo()

            if score > best_score:
                best_score = score
                best_move = move
                best_pv = (move, *child_pv)

            alpha = max(alpha, score)
            if alpha >= beta:
                self._counters.cutoffs += 1
                self._record_cutoff(
                    move,
                    player,
                    depth,
                    ply,
                )
                break

        self._store_tt(
            key,
            depth,
            extension_depth,
            best_score,
            alpha_original,
            beta_original,
            best_pv,
            best_move,
        )
        return best_score, best_pv

    def _threat_extension(
        self,
        board: Board,
        player: int,
        alpha: int,
        beta: int,
        *,
        ply: int,
        extension_depth: int,
    ) -> tuple[int, tuple[Move, ...]]:
        """共享 VCF 威胁生成器的叶子延伸。"""
        self._check_timeout()
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return 0, ()

        own_wins = self._timed_winning_moves(
            board,
            player,
            legal_moves,
        )
        if own_wins:
            return MATE_SCORE - ply, (own_wins[0],)

        opponent = other_side(player)
        opponent_wins = self._timed_winning_moves(
            board,
            opponent,
            legal_moves,
        )
        if len(opponent_wins) >= 2:
            return -MATE_SCORE + ply, ()

        if extension_depth <= 0:
            return self._static_score(board, player), ()

        if len(opponent_wins) == 1:
            forcing_moves = opponent_wins
        else:
            forcing_moves = self._forcing_attack_candidates(
                board,
                player,
                vcf_only=False,
                limit=4,
            )

        if not forcing_moves:
            return self._static_score(board, player), ()

        self._counters.extensions += 1
        best_score = -INFINITY
        best_pv: tuple[Move, ...] = ()

        for move in forcing_moves:
            self._check_timeout()
            board.place(move[0], move[1], player)
            try:
                if board.check_win(move[0], move[1]):
                    score = MATE_SCORE - ply
                    child_pv: tuple[Move, ...] = ()
                else:
                    child_score, child_pv = self._threat_extension(
                        board,
                        opponent,
                        -beta,
                        -alpha,
                        ply=ply + 1,
                        extension_depth=extension_depth - 1,
                    )
                    score = -child_score
            finally:
                board.undo()

            if score > best_score:
                best_score = score
                best_pv = (move, *child_pv)

            alpha = max(alpha, score)
            if alpha >= beta:
                self._counters.cutoffs += 1
                break

        return best_score, best_pv

    def _find_vcf(
        self,
        board: Board,
        attacker: int,
    ) -> tuple[Move, ...] | None:
        self._vcf_deadline = self._time.sub_deadline(
            self.config.vcf_time_fraction,
            minimum_seconds=0.02,
            maximum_seconds=0.8,
        )
        visited: set[tuple[int, int]] = set()
        return self._vcf_search(
            board,
            attacker,
            self.config.vcf_max_attacker_moves,
            visited,
        )

    def _vcf_search(
        self,
        board: Board,
        attacker: int,
        remaining_attacker_moves: int,
        visited: set[tuple[int, int]],
    ) -> tuple[Move, ...] | None:
        self._check_vcf_timeout()
        self._counters.vcf_nodes += 1

        key = (
            self._position_key(board, attacker),
            remaining_attacker_moves,
        )
        if key in visited:
            return None
        visited.add(key)

        legal_moves = board.get_legal_moves()
        immediate = find_winning_moves(board, attacker, legal_moves)
        if immediate:
            return (immediate[0],)
        if remaining_attacker_moves <= 0:
            return None

        defender = other_side(attacker)
        candidates = self._forcing_attack_candidates(
            board,
            attacker,
            vcf_only=True,
            limit=10,
            vcf_mode=True,
        )

        for move in candidates:
            self._check_vcf_timeout()
            board.place(move[0], move[1], attacker)
            try:
                if board.check_win(move[0], move[1]):
                    return (move,)

                attack_wins = find_winning_moves(board, attacker)
                if len(attack_wins) >= 2:
                    return (move,)
                if len(attack_wins) != 1:
                    continue

                # 对手若能直接获胜，就不必服从我们的冲四。
                defender_wins = find_winning_moves(board, defender)
                if defender_wins:
                    continue

                forced_block = attack_wins[0]
                board.place(
                    forced_block[0],
                    forced_block[1],
                    defender,
                )
                try:
                    if board.check_win(*forced_block):
                        continue
                    child = self._vcf_search(
                        board,
                        attacker,
                        remaining_attacker_moves - 1,
                        visited,
                    )
                finally:
                    board.undo()

                if child:
                    return (move, forced_block, *child)
            finally:
                board.undo()

        return None

    def _forcing_attack_candidates(
        self,
        board: Board,
        player: int,
        *,
        vcf_only: bool,
        limit: int,
        vcf_mode: bool = False,
    ) -> list[Move]:
        legal_moves = board.get_legal_moves()
        raw = self._raw_candidates(
            board,
            legal_moves,
            at_root=False,
        )
        shortlist = sorted(
            raw,
            key=lambda move: self._quick_order_score(
                board,
                move,
                player,
            ),
            reverse=True,
        )[: max(limit * 3, 16)]

        forcing: list[tuple[Move, ThreatProfile, int]] = []
        for move in shortlist:
            if vcf_mode:
                self._check_vcf_timeout()
            else:
                self._check_timeout()
            profile = self._analyze_cached(
                board,
                move,
                player,
            )
            is_vcf = (
                profile.immediate_win
                or profile.open_four_directions >= 1
                or profile.four_directions >= 1
                or profile.double_four
                or profile.four_three
            )
            is_extension = (
                is_vcf
                or profile.double_three
                or profile.open_three_directions >= 1
            )
            if (vcf_only and is_vcf) or (not vcf_only and is_extension):
                forcing.append(
                    (
                        move,
                        profile,
                        self._quick_order_score(board, move, player),
                    )
                )

        forcing.sort(
            key=lambda item: (
                item[1].tactical_rank,
                len(item[1].winning_moves),
                item[2],
            ),
            reverse=True,
        )
        return [move for move, _, _ in forcing[:limit]]

    def _multi_threat_frontiers(
        self,
        board: Board,
        candidates: list[Move],
        attacker: int,
        *,
        profiles: dict[Move, ThreatProfile] | None = None,
    ) -> dict[Move, tuple[Move, ...]]:
        """寻找一手后会产生至少两个独立强制点的启动着。

        只检查本身已经形成活三/冲四信号的候选，避免对所有普通点
        做二次展开。返回值记录启动点及其下一层强制点，供根节点
        防守筛选和自动回归测试使用。
        """
        profiles = profiles or {}
        frontiers: dict[Move, tuple[Move, ...]] = {}

        ordered = sorted(
            candidates,
            key=lambda move: self._quick_order_score(
                board,
                move,
                attacker,
            ),
            reverse=True,
        )

        for move in ordered:
            self._check_timeout()
            profile = profiles.get(move)
            if profile is None:
                profile = self._analyze_cached(board, move, attacker)

            # 已经属于当前强制威胁的点由原有逻辑处理。这里只寻找
            # “普通/活三启动 -> 下一手多重强制点”的前沿。
            if profile.forced_win:
                continue
            if (
                profile.open_three_directions < 1
                and profile.four_directions < 1
            ):
                continue

            board.place(move[0], move[1], attacker)
            try:
                reply_raw = self._raw_candidates(
                    board,
                    board.get_legal_moves(),
                    at_root=False,
                )
                reply_limit = max(
                    self.config.frontier_reply_limit * 5,
                    32,
                )
                replies = sorted(
                    reply_raw,
                    key=lambda reply: self._quick_order_score(
                        board,
                        reply,
                        attacker,
                    ),
                    reverse=True,
                )[:reply_limit]
                strong_replies: list[tuple[Move, ThreatProfile]] = []
                for reply in replies:
                    self._check_timeout()
                    reply_profile = self._analyze_cached(
                        board,
                        reply,
                        attacker,
                    )
                    if reply_profile.forced_win:
                        strong_replies.append((reply, reply_profile))
                strong_replies.sort(
                    key=lambda item: (
                        item[1].tactical_rank,
                        len(item[1].winning_moves),
                        self._quick_order_score(
                            board,
                            item[0],
                            attacker,
                        ),
                    ),
                    reverse=True,
                )
                if len(strong_replies) >= 2:
                    frontiers[move] = tuple(
                        reply
                        for reply, _ in strong_replies[
                            : self.config.frontier_reply_limit
                        ]
                    )
            finally:
                board.undo()

        return frontiers

    def _root_profile_pool(
        self,
        board: Board,
        legal_moves: list[Move],
    ) -> list[Move]:
        raw = self._raw_candidates(
            board,
            legal_moves,
            at_root=True,
        )
        limit = max(
            self.config.root_candidate_limit * 2,
            20,
        )
        return sorted(
            raw,
            key=lambda move: self._quick_order_score(
                board,
                move,
                self.player,
            ),
            reverse=True,
        )[:limit]

    def _profile_moves_timed(
        self,
        board: Board,
        candidates: list[Move],
        player: int,
    ) -> dict[Move, ThreatProfile]:
        profiles: dict[Move, ThreatProfile] = {}
        for move in candidates:
            self._check_timeout()
            profiles[move] = self._analyze_cached(
                board,
                move,
                player,
            )
        return profiles

    def _ordered_moves(
        self,
        board: Board,
        player: int,
        *,
        at_root: bool,
        ply: int,
        limit: int | None = None,
        tt_move: Move | None = None,
    ) -> list[Move]:
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return []

        own_wins = self._timed_winning_moves(board, player, legal_moves)
        if own_wins:
            return self._promote_move(own_wins, tt_move)

        opponent = other_side(player)
        opponent_wins = self._timed_winning_moves(
            board,
            opponent,
            legal_moves,
        )
        if len(opponent_wins) == 1:
            return opponent_wins
        if len(opponent_wins) >= 2:
            return self._order_specific_moves(
                board,
                opponent_wins,
                player,
                ply=ply,
                tt_move=tt_move,
                full_evaluation=(at_root or ply <= 1),
            )

        raw_candidates = self._raw_candidates(
            board,
            legal_moves,
            at_root=at_root,
        )
        if not raw_candidates:
            raw_candidates = legal_moves

        desired_limit = (
            self.config.root_candidate_limit
            if at_root
            else self.config.branch_candidate_limit
        )
        if limit is not None:
            desired_limit = limit

        # 根节点也先做廉价预筛，避免对几十到上百个点反复完整评价。
        preselection_limit = max(
            desired_limit,
            desired_limit * self.config.preselection_factor,
        )
        raw_candidates = sorted(
            raw_candidates,
            key=lambda move: (
                move == tt_move,
                self._killer_priority(move, ply),
                self._history_score(player, move),
                self._quick_order_score(board, move, player),
            ),
            reverse=True,
        )[:preselection_limit]

        if tt_move is not None and board.is_empty(*tt_move):
            if tt_move not in raw_candidates:
                raw_candidates.append(tt_move)

        ranked = self._order_specific_moves(
            board,
            raw_candidates,
            player,
            ply=ply,
            tt_move=tt_move,
            full_evaluation=(at_root or ply <= 1),
        )
        return ranked[:desired_limit]

    def _raw_candidates(
        self,
        board: Board,
        legal_moves: list[Move],
        *,
        at_root: bool,
    ) -> list[Move]:
        # 不维护全局增量候选集合；每个节点从当前棋盘安全重建局部集合。
        candidate_set: set[Move] = set()

        if at_root:
            for row, column, _ in board.move_history:
                self._add_neighborhood(
                    board,
                    candidate_set,
                    row,
                    column,
                    self.config.root_candidate_radius,
                )
        else:
            recent_moves = board.move_history[
                -self.config.recent_move_count:
            ]
            for row, column, _ in recent_moves:
                self._add_neighborhood(
                    board,
                    candidate_set,
                    row,
                    column,
                    self.config.candidate_radius,
                )
                self._add_directional_candidates(
                    board,
                    candidate_set,
                    row,
                    column,
                )

            for row, column, _ in board.move_history:
                self._add_neighborhood(
                    board,
                    candidate_set,
                    row,
                    column,
                    1,
                )

        if not candidate_set:
            return legal_moves
        return sorted(candidate_set)

    @staticmethod
    def _add_neighborhood(
        board: Board,
        target: set[Move],
        row: int,
        column: int,
        radius: int,
    ) -> None:
        for row_step in range(-radius, radius + 1):
            for column_step in range(-radius, radius + 1):
                candidate = (row + row_step, column + column_step)
                if board.is_empty(*candidate):
                    target.add(candidate)

    @staticmethod
    def _add_directional_candidates(
        board: Board,
        target: set[Move],
        row: int,
        column: int,
    ) -> None:
        """补充最近着法四条线路上的连接点，降低局部半径漏点。"""
        for row_step, column_step in DIRECTIONS:
            for sign in (-1, 1):
                for distance in range(1, 5):
                    candidate = (
                        row + sign * distance * row_step,
                        column + sign * distance * column_step,
                    )
                    if not board.is_inside(*candidate):
                        break
                    if board.is_empty(*candidate):
                        target.add(candidate)

    def _order_specific_moves(
        self,
        board: Board,
        moves: list[Move],
        player: int,
        *,
        ply: int,
        tt_move: Move | None,
        full_evaluation: bool,
    ) -> list[Move]:
        center = (board.size - 1) / 2
        scored: list[tuple[Move, int, int, int, int]] = []
        opponent = other_side(player)
        own_before = (
            evaluate_player(board, player)
            if full_evaluation
            else None
        )
        opponent_before = (
            evaluate_player(board, opponent)
            if full_evaluation
            else None
        )

        for move in dict.fromkeys(moves):
            self._check_timeout()
            killer_priority = self._killer_priority(move, ply)
            history_score = self._history_score(player, move)
            if killer_priority:
                self._counters.killer_hits += 1
            if history_score:
                self._counters.history_hits += 1
            quick_score = self._quick_order_score(board, move, player)
            move_score = (
                evaluate_move(
                    board,
                    move[0],
                    move[1],
                    player,
                    own_before=own_before,
                    opponent_before=opponent_before,
                    own_profile=self._analyze_cached(
                        board,
                        move,
                        player,
                    ),
                    opponent_profile=self._analyze_cached(
                        board,
                        move,
                        opponent,
                    ),
                )
                if full_evaluation
                else quick_score
            )
            scored.append(
                (
                    move,
                    move_score,
                    killer_priority,
                    history_score,
                    quick_score,
                )
            )

        scored.sort(
            key=lambda item: (
                item[0] == tt_move,
                item[2],
                item[3],
                item[1],
                item[4],
                -(
                    (item[0][0] - center) ** 2
                    + (item[0][1] - center) ** 2
                ),
                -item[0][0],
                -item[0][1],
            ),
            reverse=True,
        )
        return [move for move, *_ in scored]

    @staticmethod
    def _stone_count(board: Board, player: int) -> int:
        return sum(
            1
            for _, _, stone in board.move_history
            if stone == player
        )

    def _quick_fallback(
        self,
        board: Board,
        moves: list[Move],
        player: int,
    ) -> Move:
        center = (board.size - 1) / 2
        return max(
            moves,
            key=lambda move: (
                self._quick_order_score(board, move, player),
                -(
                    (move[0] - center) ** 2
                    + (move[1] - center) ** 2
                ),
                -move[0],
                -move[1],
            ),
        )

    def _quick_order_score(
        self,
        board: Board,
        move: Move,
        player: int,
    ) -> int:
        row, column = move
        opponent = other_side(player)
        score = 0
        distance_weights = (0, 24, 8, 3, 1)

        for row_step, column_step in DIRECTIONS:
            for sign in (-1, 1):
                for distance in range(1, 5):
                    neighbor_row = row + sign * distance * row_step
                    neighbor_column = (
                        column + sign * distance * column_step
                    )
                    if not board.is_inside(neighbor_row, neighbor_column):
                        break
                    cell = board.grid[neighbor_row][neighbor_column]
                    weight = distance_weights[distance]
                    if cell == player:
                        score += weight * 3
                    elif cell == opponent:
                        score += weight * 4
                    elif cell == EMPTY:
                        score += weight

        center = (board.size - 1) / 2
        score -= int(
            (row - center) ** 2
            + (column - center) ** 2
        )
        return score

    def _analyze_cached(
        self,
        board: Board,
        move: Move,
        player: int,
    ) -> ThreatProfile:
        key = (
            board.zobrist_hash,
            move[0],
            move[1],
            player,
        )
        cached = self._threat_cache.get(key)
        if cached is not None:
            return cached

        profile = analyze_move_threats(
            board,
            move[0],
            move[1],
            player,
        )
        self._threat_cache[key] = profile
        return profile

    def _static_score(self, board: Board, player: int) -> int:
        score = evaluate_board(board, player)
        return max(
            -MATE_SCORE + 10_000,
            min(MATE_SCORE - 10_000, score),
        )

    def _position_key(self, board: Board, player: int) -> int:
        return (
            board.zobrist_hash
            ^ get_zobrist_table(board.size).side_key(player)
        )

    def _tt_best_move(self, board: Board, player: int) -> Move | None:
        entry = self._transposition_table.get(
            self._position_key(board, player)
        )
        if entry is None or entry.best_move is None:
            return None
        return entry.best_move if board.is_empty(*entry.best_move) else None

    def _store_tt(
        self,
        key: int,
        depth: int,
        extension_depth: int,
        score: int,
        alpha_original: int,
        beta_original: int,
        principal_variation: tuple[Move, ...],
        best_move: Move | None,
    ) -> None:
        if not self.config.use_transposition_table:
            return

        if score <= alpha_original:
            bound = BoundType.UPPER
        elif score >= beta_original:
            bound = BoundType.LOWER
        else:
            bound = BoundType.EXACT

        previous = self._transposition_table.get(key)
        if previous is not None and previous.depth > depth:
            return

        self._transposition_table[key] = TTEntry(
            depth=depth,
            extension_depth=extension_depth,
            score=score,
            bound=bound,
            best_move=best_move,
            principal_variation=principal_variation,
            generation=self._generation,
        )

    def _record_cutoff(
        self,
        move: Move,
        player: int,
        depth: int,
        ply: int,
    ) -> None:
        killers = self._killer_moves.setdefault(ply, [])
        if move not in killers:
            killers.insert(0, move)
            del killers[2:]

        key = (player, move[0], move[1])
        self._history_scores[key] = min(
            1_000_000,
            self._history_scores.get(key, 0) + depth * depth * 16,
        )

    def _killer_priority(self, move: Move, ply: int) -> int:
        killers = self._killer_moves.get(ply, [])
        if move not in killers:
            return 0
        return 2 - killers.index(move)

    def _history_score(self, player: int, move: Move) -> int:
        return self._history_scores.get(
            (player, move[0], move[1]),
            0,
        )

    def _decay_history(self) -> None:
        if not self._history_scores:
            return
        self._history_scores = {
            key: value // 2
            for key, value in self._history_scores.items()
            if value // 2 > 0
        }

    def _prune_transposition_table(self) -> None:
        maximum = self.config.transposition_max_entries
        if len(self._transposition_table) <= maximum:
            return

        minimum_generation = max(0, self._generation - 2)
        self._transposition_table = {
            key: entry
            for key, entry in self._transposition_table.items()
            if entry.generation >= minimum_generation
        }
        if len(self._transposition_table) > maximum:
            self._transposition_table.clear()

    @staticmethod
    def _promote_move(
        moves: list[Move],
        preferred: Move | None,
    ) -> list[Move]:
        if preferred is None or preferred not in moves:
            return list(moves)
        return [preferred, *(move for move in moves if move != preferred)]

    def _timed_winning_moves(
        self,
        board: Board,
        player: int,
        candidates: list[Move],
    ) -> list[Move]:
        winning: list[Move] = []
        for row, column in candidates:
            self._check_timeout()
            board.place(row, column, player)
            try:
                if board.check_win(row, column):
                    winning.append((row, column))
            finally:
                board.undo()
        return winning

    def _check_timeout(self) -> None:
        if self._time.hard_expired():
            raise SearchTimeout

    def _check_vcf_timeout(self) -> None:
        now = time.perf_counter()
        if self._time.hard_deadline is not None:
            if now >= self._time.hard_deadline:
                raise SearchTimeout
        if self._vcf_deadline is not None and now >= self._vcf_deadline:
            raise VCFTimeout

    def _save_search_analysis(
        self,
        *,
        selected_move: Move,
        reason: str,
        candidate_count: int,
        ranked_moves: list[tuple[Move, int]],
        completed_depth: int,
        principal_variation: tuple[Move, ...],
        search_completed: bool,
        own_profiles: dict[Move, ThreatProfile] | None = None,
        opponent_profiles: dict[Move, ThreatProfile] | None = None,
        vcf_found: bool = False,
        vcf_depth: int = 0,
        stop_reason: str = "unspecified",
    ) -> None:
        elapsed = self._time.elapsed_seconds
        nps = int(self._counters.nodes / elapsed) if elapsed > 0 else 0
        top_candidates: list[CandidateAnalysis] = []

        if self.diagnostics:
            own_profiles = own_profiles or {}
            opponent_profiles = opponent_profiles or {}
            for move, score in ranked_moves[: self.top_n]:
                top_candidates.append(
                    CandidateAnalysis(
                        move=move,
                        score=score,
                        own_threat=(
                            own_profiles[move].label
                            if move in own_profiles
                            else "普通"
                        ),
                        opponent_threat=(
                            opponent_profiles[move].label
                            if move in opponent_profiles
                            else "普通"
                        ),
                    )
                )

        soft_limit = (
            None
            if self.config.time_limit_seconds is None
            else (
                self.config.time_limit_seconds
                * self.config.soft_time_ratio
            )
        )
        time_used_ratio = (
            None
            if self.config.time_limit_seconds is None
            else min(
                1.0,
                elapsed / self.config.time_limit_seconds,
            )
        )
        self.last_analysis = DecisionAnalysis(
            selected_move=selected_move,
            reason=reason,
            candidate_count=candidate_count,
            top_candidates=tuple(top_candidates),
            search_depth=completed_depth,
            requested_depth=self.config.max_depth,
            interrupted_depth=self._interrupted_depth,
            nodes=self._counters.nodes,
            nps=nps,
            cutoffs=self._counters.cutoffs,
            transposition_hits=self._counters.transposition_hits,
            transposition_cutoffs=self._counters.transposition_cutoffs,
            transposition_size=len(self._transposition_table),
            killer_hits=self._counters.killer_hits,
            history_hits=self._counters.history_hits,
            extensions=self._counters.extensions,
            pvs_researches=self._counters.pvs_researches,
            aspiration_researches=self._counters.aspiration_researches,
            vcf_found=vcf_found,
            vcf_depth=vcf_depth,
            vcf_nodes=self._counters.vcf_nodes,
            elapsed_seconds=elapsed,
            soft_time_limit_seconds=soft_limit,
            hard_time_limit_seconds=self.config.time_limit_seconds,
            principal_variation=principal_variation,
            search_completed=search_completed,
            stop_reason=stop_reason,
            time_used_ratio=time_used_ratio,
            defense_vct_checked=self._defense_probe is not None,
            defense_vct_depth=(
                0
                if self._defense_probe is None
                else self._defense_probe.completed_depth
            ),
            defense_vct_nodes=(
                0
                if self._defense_probe is None
                else self._defense_probe.nodes
            ),
            defense_vct_best_move=(
                None
                if self._defense_probe is None
                else self._defense_probe.best_move
            ),
            defense_vct_candidates=(
                ()
                if self._defense_probe is None
                else self._defense_probe.candidates
            ),
        )
