from __future__ import annotations

import time
from dataclasses import replace

from engine.ai import (
    CandidateAnalysis,
    DecisionAnalysis,
    DefenseCandidateAnalysis,
    Move,
    ProofCandidateAnalysis,
    RootSafetyCandidateAnalysis,
    ScoringAI,
)
from engine.board import DIRECTIONS, EMPTY, WHITE, Board
from engine.evaluator import (
    ThreatProfile,
    evaluate_board,
    evaluate_move,
    find_winning_moves,
    is_winning_move,
    other_side,
)
from engine.time_manager import TimeManager
from engine.proof_search import (
    ProofBudget,
    ProofResult,
    ProofSearch,
    ProofState,
    ProofTable,
    ProofTableStats,
)
from engine import root_candidates, root_policy, root_safety
from engine.vcf import VCFSearch
from engine.threats import ThreatAnalyzer, ThreatAnalyzerStats
from engine.zobrist import get_zobrist_table
from engine.search_types import (
    BoundType,
    DefenseProbeResult,
    DefenseProof,
    HEURISTIC_SCORE_LIMIT,
    INFINITY,
    IterativeSearchOutcome,
    MATE_SCORE,
    RootResult,
    RootSafetyProbeResult,
    RootVCFScanResult,
    SearchConfig,
    SearchCounters,
    SearchTimeout,
    TTEntry,
    VCFTimeout,
)


class SearchAI(ScoringAI):
    """
    V0.12.5 搜索 AI。

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
    V0.12.0 保持相同搜索树和候选顺序，将一步胜点、四方向威胁
    画像和候选棋型增益改为不修改棋盘的局部线计算。
    V0.12.1 在根节点同时保留己方强攻点和对手必防点，并在普通
    或仅己方强攻候选都落入近必败带时扩大候选集重搜，避免启发式
    强威胁把唯一有生存机会的防守着在搜索开始前排除。
    V0.12.2 为多个必防端点增加独立仲裁，并隔离选择性 PVS 中
    没有严格 Proof 支持的 Mate 分。
    V0.12.3 只重构职责边界：根候选、根策略、根安全、VCF 与共享
    搜索类型独立成模块，搜索参数和选着规则保持不变。
    V0.12.4 在根搜索前用共享预算逐项检查落子后的对手 VCF，
    淘汰已证明会立即进入连续冲四败势的候选；同时把已发现 VCF
    的攻击落点并入根候选，并在高密度威胁前沿保留安静活三反击。
    V0.12.5 补全防守候选来源，并让 VCF 拦截证据覆盖整条交替
    线路，同时继续遵守固定根候选上限。
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
        self._threat_exact_cache: dict = {}
        self._threat_candidate_cache: dict = {}
        self._defense_probe: DefenseProbeResult | None = None
        self._proof_table = ProofTable()
        self._proof_table_start_stats = ProofTableStats()
        self._proof_analyzer = self._new_threat_analyzer()
        self._proof_root_result: ProofResult | None = None
        self._proof_candidates: tuple[ProofCandidateAnalysis, ...] = ()
        self._root_safety_probe: RootSafetyProbeResult | None = None
        self._root_safety_applied = False
        self._root_vcf_scan: RootVCFScanResult | None = None
        self._root_mate_scores_quarantined = False
        self._root_heuristic_score_cache: dict[
            tuple[int, Move],
            int,
        ] = {}
        self._search_phase_deadline: float | None = None
        self._search_phase_timeout_hit = False

    def choose_move(self, board: Board) -> Move:
        """按硬战术、VCF、限时迭代加深的顺序选择落点。"""
        self.last_analysis = None
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            raise ValueError("棋盘已满，电脑无法落子。")

        self._begin_move_search()

        # 先准备一个绝不需要完整评价的合法回退点。
        fallback_move = self._quick_fallback(
            board,
            legal_moves,
            self.player,
        )
        tactical_move = self._try_tactical_shortcut(
            board,
            legal_moves,
        )
        if tactical_move is not None:
            return tactical_move

        try:
            plan = self._prepare_root_candidate_plan(
                board,
                legal_moves,
            )
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

        search_candidates = plan.moves
        own_profiles = plan.own_profiles
        opponent_profiles = plan.opponent_profiles
        own_forcing_moves = plan.own_forcing_moves
        preserve_frontier_order = plan.preserve_frontier_order
        allow_near_loss_expansion = (
            plan.allow_near_loss_expansion
        )
        defense_probe = plan.defense_probe
        tactical_reason = plan.reason

        root_vcf_scan = self._run_root_opponent_vcf_scan(
            board,
            search_candidates,
        )
        if root_vcf_scan is not None:
            self._root_vcf_scan = root_vcf_scan
            search_candidates = self._filter_root_vcf_candidates(
                list(root_vcf_scan.candidates)
            )

        proof_win = self._run_proof_arbitration(
            board,
            search_candidates,
            search_own_win=bool(own_forcing_moves),
        )
        if (
            proof_win is not None
            and proof_win.state is ProofState.PROVEN_WIN
            and proof_win.best_move is not None
            and board.is_empty(*proof_win.best_move)
        ):
            selected = proof_win.best_move
            self._save_search_analysis(
                selected_move=selected,
                reason="AND/OR 威胁空间搜索严格证明胜势",
                candidate_count=len(search_candidates),
                ranked_moves=[(selected, MATE_SCORE)],
                completed_depth=0,
                principal_variation=proof_win.principal_variation,
                search_completed=True,
                stop_reason="proof_proven_win",
            )
            return selected

        outcome = self._run_iterative_root_search(
            board,
            search_candidates,
            fallback_move=fallback_move,
            preserve_frontier_order=preserve_frontier_order,
            allow_near_loss_expansion=allow_near_loss_expansion,
            defense_probe=defense_probe,
        )
        best_result = outcome.result
        search_candidates = outcome.candidates
        completed_depth = outcome.completed_depth
        search_completed = outcome.search_completed
        stop_reason = outcome.stop_reason
        root_candidates_expanded = (
            outcome.root_candidates_expanded
        )

        reason = (
            f"{tactical_reason}（完成深度 {completed_depth}）"
            if completed_depth > 0
            else f"{tactical_reason}（时间不足，使用快速回退）"
        )
        if root_candidates_expanded:
            reason += "；近必败候选已自动扩展"
        reason = self._append_root_vcf_reason(reason)
        if self._root_mate_scores_quarantined:
            reason += "；未证明 Mate 分已降级为启发式分"
        if self._root_safety_probe is not None:
            if self._root_safety_applied:
                if (
                    self._root_safety_probe.trigger
                    == "threat_risk_override"
                ):
                    reason += "；独立复核批准风险指标改选"
                else:
                    reason += "；决胜节点复核改选近分候选"
            else:
                if (
                    self._root_safety_probe.trigger
                    == "threat_risk_override"
                ):
                    reason += "；风险改选未获确认，保留 PVS 首选"
                else:
                    reason += "；决胜节点复核保持原候选"
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

    def _append_root_vcf_reason(self, reason: str) -> str:
        scan = self._root_vcf_scan
        if scan is None:
            return reason
        if any(
            move not in scan.original_candidates
            for move in scan.candidates
        ):
            reason += "；已合并对手 VCF 拦截点"
        if any(
            candidate.status
            == root_safety.RootCandidateSafety.PROVEN_LOSS.value
            for candidate in scan.analyses
        ):
            reason += "；已淘汰对手 VCF 可证败着"
        if not scan.complete:
            reason += "；对手 VCF 生存检查部分候选未知"
        return reason

    def _begin_move_search(self) -> None:
        """Reset per-move state while preserving long-lived tables."""
        self._time = TimeManager.start(
            self.config.time_limit_seconds,
            soft_ratio=self.config.soft_time_ratio,
        )
        self._counters = SearchCounters()
        self._generation += 1
        self._killer_moves.clear()
        self._interrupted_depth = 0
        self._threat_cache.clear()
        self._threat_exact_cache.clear()
        self._threat_candidate_cache.clear()
        self._defense_probe = None
        self._proof_root_result = None
        self._proof_candidates = ()
        self._root_safety_probe = None
        self._root_safety_applied = False
        self._root_vcf_scan = None
        self._root_mate_scores_quarantined = False
        self._root_heuristic_score_cache.clear()
        self._search_phase_deadline = None
        self._search_phase_timeout_hit = False
        self._proof_table_start_stats = self._proof_table.stats()
        self._proof_analyzer = self._new_threat_analyzer()
        self._decay_history()
        self._prune_transposition_table()

    def _try_tactical_shortcut(
        self,
        board: Board,
        legal_moves: list[Move],
    ) -> Move | None:
        """Handle only strict immediate, opening, and VCF exits."""
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

        return None

    def _run_iterative_root_search(
        self,
        board: Board,
        search_candidates: list[Move],
        *,
        fallback_move: Move,
        preserve_frontier_order: bool,
        allow_near_loss_expansion: bool,
        defense_probe: DefenseProbeResult | None,
    ) -> IterativeSearchOutcome:
        """Run iterative PVS and return data needed by final reporting."""
        search_candidates = self._filter_proven_losing_candidates(
            search_candidates
        )
        search_candidates = self._filter_root_vcf_candidates(
            search_candidates
        )
        if not search_candidates:
            search_candidates = [fallback_move]
        self._register_expanded_candidates_as_unknown(
            search_candidates
        )

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
        final_pvs_result: RootResult | None = None
        final_risk_result: RootResult | None = None
        completed_depth = 0
        search_completed = True
        stop_reason = "requested_depth_completed"
        root_history: list[RootResult] = []
        root_candidates_expanded = False

        for depth in range(1, self.config.max_depth + 1):
            if depth > 1 and self._time.soft_expired():
                self._interrupted_depth = depth
                search_completed = False
                stop_reason = "soft_deadline"
                break

            self._search_phase_deadline = None
            self._search_phase_timeout_hit = False
            if completed_depth > 0 and self._root_safety_trigger(
                best_result,
                root_history,
            ) is not None:
                reserve = self._root_safety_budget_seconds()
                if (
                    reserve > 0
                    and self._time.hard_deadline is not None
                ):
                    self._search_phase_deadline = (
                        self._time.hard_deadline - reserve
                    )
                    if (
                        time.perf_counter()
                        >= self._search_phase_deadline
                    ):
                        self._interrupted_depth = depth
                        search_completed = False
                        stop_reason = "root_safety_reserve"
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

                if (
                    not root_candidates_expanded
                    and allow_near_loss_expansion
                    and self._all_root_candidates_near_forced_loss(
                        result,
                        search_candidates,
                    )
                ):
                    expanded_candidates = (
                        self._expand_near_loss_root_candidates(
                            board,
                            search_candidates,
                        )
                    )
                    if len(expanded_candidates) > len(
                        search_candidates
                    ):
                        search_candidates = (
                            self._filter_root_vcf_candidates(
                                expanded_candidates
                            )
                        )
                        self._register_expanded_candidates_as_unknown(
                            search_candidates
                        )
                        root_candidates_expanded = True
                        result = self._search_root(
                            board,
                            self.player,
                            depth,
                            search_candidates,
                            alpha=-INFINITY,
                            beta=INFINITY,
                        )

                result = self._quarantine_unproven_root_scores(
                    board,
                    result,
                    preserve_order=preserve_frontier_order,
                )
            except SearchTimeout:
                self._interrupted_depth = depth
                search_completed = False
                stop_reason = (
                    "root_safety_reserve"
                    if self._search_phase_timeout_hit
                    else "hard_deadline"
                )
                break

            final_pvs_result = None
            final_risk_result = None
            if defense_probe is not None:
                result = self._apply_defense_probe_tiebreak(
                    result,
                    defense_probe,
                )
            elif self._proof_candidates:
                pvs_result = result
                result = self._apply_proof_tiebreak(pvs_result)
                if self._is_unknown_risk_override(
                    pvs_result,
                    result,
                ):
                    final_pvs_result = pvs_result
                    final_risk_result = result

            best_result = result
            completed_depth = depth
            root_history.append(result)

            if not preserve_frontier_order:
                search_candidates = self._promote_move(
                    search_candidates,
                    result.move,
                )

        self._search_phase_deadline = None
        if completed_depth < self.config.max_depth:
            search_completed = False
            if self._interrupted_depth == 0:
                self._interrupted_depth = completed_depth + 1
                stop_reason = "search_incomplete"

        if (
            final_pvs_result is not None
            and final_risk_result is not None
            and completed_depth > 0
        ):
            best_result = self._finalize_risk_override(
                board,
                final_pvs_result,
                final_risk_result,
                completed_depth=completed_depth,
                root_history=root_history,
            )
        elif not search_completed and completed_depth > 0:
            safety_probe = self._maybe_run_root_safety_probe(
                board,
                best_result,
                completed_depth=completed_depth,
                root_history=root_history,
            )
            if safety_probe is not None:
                self._root_safety_probe = safety_probe
                revised = self._apply_root_safety_probe(
                    best_result,
                    safety_probe,
                )
                self._root_safety_applied = (
                    revised.move != best_result.move
                )
                best_result = revised

        return IterativeSearchOutcome(
            result=best_result,
            candidates=search_candidates,
            completed_depth=completed_depth,
            search_completed=search_completed,
            stop_reason=stop_reason,
            root_candidates_expanded=root_candidates_expanded,
        )

    def _prepare_root_candidate_plan(
        self,
        board: Board,
        legal_moves: list[Move],
    ) -> root_candidates.RootCandidatePlan:
        """Build the root set; final move selection remains in PVS/policy."""
        preserve_frontier_order = False
        allow_near_loss_expansion = True
        defense_probe: DefenseProbeResult | None = None

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
        opponent_frontier_moves = (
            root_candidates.strongest_frontier_moves(
                opponent_frontiers
            )
        )
        candidate_mode = root_candidates.classify_mode(
            own_forcing_moves=own_forcing_moves,
            opponent_forcing_moves=opponent_forcing_moves,
            opponent_frontier_moves=opponent_frontier_moves,
        )

        if (
            candidate_mode
            is root_candidates.RootCandidateMode.MERGED_FORCING
        ):
            allow_near_loss_expansion = False
            search_candidates = self._order_specific_moves(
                board,
                root_candidates.merge_unique(
                    opponent_forcing_moves,
                    own_forcing_moves,
                ),
                self.player,
                ply=0,
                tt_move=self._tt_best_move(board, self.player),
                full_evaluation=True,
            )
            tactical_reason = "合并攻方强制点与对手必防点搜索"
            if 2 <= len(opponent_forcing_moves) <= (
                self.config.defense_vct_max_candidates
            ):
                defense_probe = self._run_mandatory_defense_probe(
                    board,
                    self.player,
                    self._order_specific_moves(
                        board,
                        opponent_forcing_moves,
                        self.player,
                        ply=0,
                        tt_move=None,
                        full_evaluation=True,
                    ),
                )
                if defense_probe is not None:
                    self._defense_probe = defense_probe
                    tactical_reason = (
                        "合并攻方强制点与对手必防点搜索；"
                        "必防分支独立仲裁"
                    )
        elif (
            candidate_mode
            is root_candidates.RootCandidateMode.OWN_FORCING
        ):
            search_candidates = self._order_specific_moves(
                board,
                own_forcing_moves,
                self.player,
                ply=0,
                tt_move=self._tt_best_move(board, self.player),
                full_evaluation=True,
            )
            tactical_reason = "搜索自身强制威胁的最佳变化"
        elif (
            candidate_mode
            is root_candidates.RootCandidateMode.MANDATORY_DEFENSE
        ):
            allow_near_loss_expansion = False
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
        elif (
            candidate_mode
            is root_candidates.RootCandidateMode.FRONTIER_DEFENSE
        ):
            allow_near_loss_expansion = False
            frontier_candidates = self._order_specific_moves(
                board,
                opponent_frontier_moves,
                self.player,
                ply=0,
                tt_move=self._tt_best_move(board, self.player),
                full_evaluation=True,
            )
            frontier_candidates = frontier_candidates[
                : self.config.frontier_reply_limit
            ]
            ordinary_candidates = (
                self._ordered_moves(
                    board,
                    self.player,
                    at_root=True,
                    ply=0,
                    tt_move=self._tt_best_move(board, self.player),
                )
                if len(frontier_candidates) == 1
                else []
            )
            counterattacks = self._order_specific_moves(
                board,
                root_candidates.active_counterattack_moves(
                    own_profiles
                ),
                self.player,
                ply=0,
                tt_move=None,
                full_evaluation=True,
            )
            search_candidates = (
                root_candidates.frontier_defense_moves(
                    frontier_moves=frontier_candidates,
                    ordinary_moves=ordinary_candidates,
                    counterattack_moves=counterattacks,
                    limit=self.config.root_candidate_limit,
                )
            )
            preserve_frontier_order = True
            tactical_reason = "多重威胁启动点候选的 PVS 防守"
            if ordinary_candidates:
                tactical_reason += "；单前沿已补入普通候选"
            if counterattacks:
                tactical_reason += "；已补入主动反击点"
        else:
            search_candidates = self._ordered_moves(
                board,
                self.player,
                at_root=True,
                ply=0,
            )
            tactical_reason = "PVS 搜索最佳变化"

        return root_candidates.RootCandidatePlan(
            moves=search_candidates,
            own_profiles=own_profiles,
            opponent_profiles=opponent_profiles,
            own_forcing_moves=own_forcing_moves,
            preserve_frontier_order=preserve_frontier_order,
            allow_near_loss_expansion=allow_near_loss_expansion,
            defense_probe=defense_probe,
            reason=tactical_reason,
        )

    def _run_root_opponent_vcf_scan(
        self,
        board: Board,
        candidates: list[Move],
    ) -> RootVCFScanResult | None:
        """Reject root moves that immediately concede a bounded opponent VCF."""
        if (
            not self.config.root_vcf_safety_enabled
            or not candidates
            or self._stone_count(board, self.opponent) < 3
        ):
            return None

        budget = self._root_vcf_safety_budget_seconds()
        if budget == 0:
            return None

        scanner = root_safety.RootVCFSafetyScanner(
            find_vcf=lambda position, attacker, deadline: (
                self._find_vcf_with_deadline(
                    position,
                    attacker,
                    deadline=deadline,
                    root_safety_channel=True,
                )
            ),
            node_count=lambda: self._counters.root_vcf_nodes,
            intercept_fraction=(
                self.config.root_vcf_safety_intercept_fraction
            ),
            candidate_limit=self.config.root_candidate_limit,
        )
        return scanner.scan(
            board,
            candidates,
            mover=self.player,
            opponent=self.opponent,
            budget_seconds=budget,
            hard_deadline=self._time.hard_deadline,
        )

    def _root_vcf_safety_budget_seconds(
        self,
    ) -> float | None:
        total = self.config.time_limit_seconds
        remaining = self._time.remaining_seconds
        if total is None or remaining is None:
            return None
        budget = min(
            self.config.root_vcf_safety_max_seconds,
            total * self.config.root_vcf_safety_time_fraction,
            max(0.0, remaining - 0.05),
        )
        if budget < self.config.root_vcf_safety_min_seconds:
            return 0.0
        return budget

    def _find_vcf_with_deadline(
        self,
        board: Board,
        attacker: int,
        *,
        deadline: float | None,
        root_safety_channel: bool,
    ) -> tuple[Move, ...] | None:
        previous_deadline = self._vcf_deadline
        self._vcf_deadline = deadline
        search = VCFSearch(
            position_key=self._position_key,
            forcing_candidates=lambda position, player: (
                self._forcing_attack_candidates(
                    position,
                    player,
                    vcf_only=True,
                    limit=10,
                    vcf_mode=True,
                )
            ),
            check_timeout=self._check_vcf_timeout,
            count_node=(
                self._count_root_vcf_node
                if root_safety_channel
                else self._count_vcf_node
            ),
        )
        try:
            return search.find(
                board,
                attacker,
                (
                    self.config.root_vcf_safety_max_attacker_moves
                    if root_safety_channel
                    else self.config.vcf_max_attacker_moves
                ),
            )
        finally:
            self._vcf_deadline = previous_deadline

    def _filter_root_vcf_candidates(
        self,
        candidates: list[Move],
    ) -> list[Move]:
        if self._root_vcf_scan is None:
            return candidates
        return root_safety.apply_vcf_scan(
            candidates,
            self._root_vcf_scan.analyses,
        )

    def _run_proof_arbitration(
        self,
        board: Board,
        candidates: list[Move],
        *,
        search_own_win: bool,
    ) -> ProofResult | None:
        """Run the strict proof channel without borrowing PVS state.

        A candidate result is always relative to ``self.opponent`` after
        our candidate has been placed.  PROVEN_WIN therefore means that the
        candidate is a strictly proved loss for us; UNKNOWN remains unknown.
        Threat-risk is only a secondary ordering hint among UNKNOWN results.
        """
        seconds = self._proof_budget_seconds()
        if seconds <= 0 or not candidates:
            return None

        deadline = time.perf_counter() + seconds
        analyzer = self._proof_analyzer

        if search_own_win and time.perf_counter() < deadline:
            root_deadline = min(
                deadline,
                time.perf_counter() + max(0.05, seconds * 0.25),
            )
            root_search = ProofSearch(
                budget=ProofBudget(
                    max_nodes=self.config.proof_max_nodes,
                    max_attacker_moves=(
                        self.config.proof_max_attacker_moves
                    ),
                    max_quiet_frontiers=0,
                    max_quiet_attacker_moves=0,
                    deadline=root_deadline,
                ),
                analyzer=analyzer,
                table=self._proof_table,
                clock=time.perf_counter,
            )
            self._proof_root_result = root_search.search(
                board,
                attacker=self.player,
                side_to_move=self.player,
            )
            self._counters.proof_nodes += (
                self._proof_root_result.nodes
            )
            if (
                self._proof_root_result.state
                is ProofState.PROVEN_WIN
            ):
                return self._proof_root_result

        probed = list(
            candidates[: self.config.proof_root_candidate_limit]
        )
        risks: dict[Move, int] = {}
        for move in probed:
            risks[move] = self._threat_risk_after_move(
                board,
                move,
                analyzer=analyzer,
            )

        analyses: list[ProofCandidateAnalysis] = []
        for index, move in enumerate(probed):
            now = time.perf_counter()
            if now >= deadline:
                analyses.append(
                    ProofCandidateAnalysis(
                        move=move,
                        state=ProofState.UNKNOWN.value,
                        completed=False,
                        nodes=0,
                        elapsed_seconds=0.0,
                        cutoff_reason="proof_budget_exhausted",
                        threat_risk=risks.get(move),
                    )
                )
                continue

            candidates_left = len(probed) - index
            candidate_seconds = max(
                0.05,
                (deadline - now) / candidates_left,
            )
            candidate_search = ProofSearch(
                budget=ProofBudget(
                    max_nodes=self.config.proof_max_nodes,
                    max_attacker_moves=(
                        self.config.proof_max_attacker_moves
                    ),
                    max_quiet_frontiers=(
                        self.config.proof_quiet_frontier_limit
                    ),
                    max_quiet_attacker_moves=(
                        self.config.proof_quiet_attacker_moves
                    ),
                    deadline=min(deadline, now + candidate_seconds),
                ),
                analyzer=analyzer,
                table=self._proof_table,
                clock=time.perf_counter,
            )
            result = candidate_search.search_after_move(
                board,
                move=move,
                mover=self.player,
                attacker=self.opponent,
                side_to_move=self.opponent,
            )
            self._counters.proof_nodes += result.nodes
            analyses.append(
                ProofCandidateAnalysis(
                    move=move,
                    state=result.state.value,
                    completed=result.completed,
                    nodes=result.nodes,
                    elapsed_seconds=result.elapsed_seconds,
                    cutoff_reason=result.cutoff_reason,
                    principal_variation=result.principal_variation,
                    threat_risk=risks.get(move),
                )
            )

        self._proof_candidates = tuple(analyses)
        return self._proof_root_result

    def _proof_budget_seconds(self) -> float:
        total = self.config.time_limit_seconds
        remaining = self._time.remaining_seconds
        if total is None or remaining is None or total < 4.0:
            return 0.0
        budget = min(
            self.config.proof_max_seconds,
            total * self.config.proof_time_fraction,
            max(0.0, remaining - 0.1),
        )
        return budget if budget >= 0.1 else 0.0

    def _new_threat_analyzer(self) -> ThreatAnalyzer:
        return ThreatAnalyzer(
            candidate_limit=16,
            frontier_scan_limit=self.config.proof_frontier_scan_limit,
            cache_enabled=self.config.proof_use_threat_cache,
            profile_cache=self._threat_cache,
            exact_cache=self._threat_exact_cache,
            candidate_cache=self._threat_candidate_cache,
        )

    def _threat_risk_after_move(
        self,
        board: Board,
        move: Move,
        *,
        analyzer: ThreatAnalyzer,
    ) -> int:
        """Return a deterministic UNKNOWN-ordering hint, never proof.

        Candidate risk must not depend on how quickly the host reaches the
        strict-proof deadline.  Exact frontier generation is intentionally
        not used here: on slower machines it could consume the whole proof
        slice and leave every candidate with ``None``.  Profiled forcing
        candidates are much cheaper and their dependency scores still
        measure the concrete next-threat pressure needed by root ordering.
        """
        if not board.is_empty(*move):
            return 0

        board.place(*move, self.player)
        try:
            batch = analyzer.generate_attack_candidates(
                board,
                self.opponent,
            )
        finally:
            board.undo()

        if not batch.generation_completed or not batch.candidates:
            return 0
        pressures = [
            (
                candidate.profile.tactical_rank
                + candidate.dependency_score
            )
            for candidate in batch.candidates
        ]
        return (
            max(pressures) * 10_000
            + sum(pressures) * 10
            + len(pressures)
        )

    def _filter_proven_losing_candidates(
        self,
        candidates: list[Move],
    ) -> list[Move]:
        states = {
            item.move: item.state
            for item in self._proof_candidates
        }
        survivors = [
            move
            for move in candidates
            if states.get(move) != ProofState.PROVEN_WIN.value
        ]
        return survivors or candidates

    def _register_expanded_candidates_as_unknown(
        self,
        candidates: list[Move],
    ) -> None:
        """Give every unprobed root move explicit UNKNOWN semantics.

        A bounded proof slice commonly checks only the first few root moves.
        The remaining legal root candidates must not disappear from later
        arbitration merely because they were outside that slice.
        """
        if not self._proof_candidates:
            return
        known = {
            candidate.move for candidate in self._proof_candidates
        }
        additions = tuple(
            ProofCandidateAnalysis(
                move=move,
                state=ProofState.UNKNOWN.value,
                completed=False,
                nodes=0,
                elapsed_seconds=0.0,
                cutoff_reason="root_expansion_unprobed",
                threat_risk=None,
            )
            for move in candidates
            if move not in known
        )
        self._proof_candidates = (
            *self._proof_candidates,
            *additions,
        )

    @staticmethod
    def _is_mate_like_score(score: int) -> bool:
        return root_policy.is_mate_like_score(score)

    def _heuristic_root_score(
        self,
        board: Board,
        move: Move,
    ) -> int:
        """Evaluate a root move without assigning proof semantics."""
        key = (board.zobrist_hash, move)
        cached = self._root_heuristic_score_cache.get(key)
        if cached is not None:
            return cached
        board.place(*move, self.player)
        try:
            if board.check_win(*move):
                score = MATE_SCORE
            else:
                score = self._static_score(board, self.player)
        finally:
            board.undo()
        self._root_heuristic_score_cache[key] = score
        return score

    def _root_score_has_strict_mate_evidence(
        self,
        score: int,
        proof_state: str | None,
    ) -> bool:
        return root_policy.has_strict_mate_evidence(
            score,
            proof_state,
        )

    def _quarantine_unproven_root_scores(
        self,
        board: Board,
        result: RootResult,
        *,
        preserve_order: bool = False,
    ) -> RootResult:
        proof_states = {
            candidate.move: candidate.state
            for candidate in self._proof_candidates
        }
        revised, quarantined = (
            root_policy.quarantine_unproven_scores(
                result,
                proof_states=proof_states,
                heuristic_score=lambda move: (
                    self._heuristic_root_score(board, move)
                ),
                preserve_order=preserve_order,
            )
        )
        self._root_mate_scores_quarantined |= quarantined
        return revised

    def _apply_proof_tiebreak(
        self,
        result: RootResult,
    ) -> RootResult:
        return root_policy.apply_proof_tiebreak(
            self.config,
            result,
            self._proof_candidates,
        )

    def _is_unknown_risk_override(
        self,
        pvs_result: RootResult,
        revised_result: RootResult,
    ) -> bool:
        states = {
            candidate.move: candidate.state
            for candidate in self._proof_candidates
        }
        return root_policy.is_unknown_risk_override(
            pvs_result,
            revised_result,
            proof_states=states,
        )

    def _maybe_run_risk_override_probe(
        self,
        board: Board,
        pvs_result: RootResult,
        risk_result: RootResult,
        *,
        completed_depth: int,
        root_history: list[RootResult],
    ) -> RootSafetyProbeResult | None:
        """Require independent confirmation before UNKNOWN risk overrides PVS.

        The original PVS result is always the fallback.  A missing budget,
        timeout, insufficient depth, or unstable probe ranking therefore
        preserves it instead of silently accepting a heuristic risk hint.
        """
        if not self.config.root_safety_enabled:
            return None

        proof_states = {
            candidate.move: candidate.state
            for candidate in self._proof_candidates
        }
        if (
            proof_states.get(pvs_result.move)
            != ProofState.UNKNOWN.value
            or proof_states.get(risk_result.move)
            != ProofState.UNKNOWN.value
        ):
            return None

        root_scores = dict(pvs_result.ranked_moves)
        if (
            pvs_result.move not in root_scores
            or risk_result.move not in root_scores
        ):
            return None

        pvs_gap = (
            root_scores[pvs_result.move]
            - root_scores[risk_result.move]
        )
        budget = self._root_safety_budget_seconds()
        if budget <= 0:
            return RootSafetyProbeResult(
                trigger="threat_risk_override",
                pvs_gap=pvs_gap,
                main_rank_stable=self._root_rank_is_stable(
                    root_history
                ),
                completed_depth=0,
                nodes=0,
                candidates=(),
            )

        probe = self._run_root_safety_probe(
            board,
            [pvs_result.move, risk_result.move],
            trigger="threat_risk_override",
            pvs_gap=pvs_gap,
            main_rank_stable=self._root_rank_is_stable(root_history),
            completed_depth=completed_depth,
            budget_seconds=budget,
        )
        if probe is not None:
            return probe
        return RootSafetyProbeResult(
            trigger="threat_risk_override",
            pvs_gap=pvs_gap,
            main_rank_stable=self._root_rank_is_stable(root_history),
            completed_depth=0,
            nodes=0,
            candidates=(),
        )

    def _finalize_risk_override(
        self,
        board: Board,
        pvs_result: RootResult,
        risk_result: RootResult,
        *,
        completed_depth: int,
        root_history: list[RootResult],
    ) -> RootResult:
        """Apply a risk change only after a stable independent recheck."""
        safety_probe = self._maybe_run_risk_override_probe(
            board,
            pvs_result,
            risk_result,
            completed_depth=completed_depth,
            root_history=root_history,
        )
        if safety_probe is None:
            self._root_safety_probe = None
            self._root_safety_applied = False
            return pvs_result

        self._root_safety_probe = safety_probe
        revised = self._apply_root_safety_probe(
            pvs_result,
            safety_probe,
        )
        self._root_safety_applied = (
            revised.move != pvs_result.move
        )
        return revised

    def _root_safety_trigger(
        self,
        result: RootResult,
        root_history: list[RootResult],
    ) -> str | None:
        proof_states = {
            candidate.move: candidate.state
            for candidate in self._proof_candidates
        }
        return root_safety.trigger(
            self.config,
            result,
            root_history,
            proof_states=proof_states,
            mate_scores_quarantined=(
                self._root_mate_scores_quarantined
            ),
        )

    @staticmethod
    def _root_rank_is_stable(
        root_history: list[RootResult],
    ) -> bool:
        return root_safety.rank_is_stable(root_history)

    def _root_safety_candidates(
        self,
        result: RootResult,
    ) -> tuple[list[Move], int]:
        return root_safety.candidates(self.config, result)

    def _root_safety_budget_seconds(self) -> float:
        return root_safety.budget_seconds(
            self.config,
            remaining_seconds=self._time.remaining_seconds,
        )

    def _maybe_run_root_safety_probe(
        self,
        board: Board,
        result: RootResult,
        *,
        completed_depth: int,
        root_history: list[RootResult],
    ) -> RootSafetyProbeResult | None:
        """Recheck a close, unfinished decision with isolated search state."""
        trigger = self._root_safety_trigger(result, root_history)
        if trigger is None:
            return None

        candidates, pvs_gap = self._root_safety_candidates(result)
        if len(candidates) < 2:
            return None

        budget = self._root_safety_budget_seconds()
        if budget <= 0:
            return None

        return self._run_root_safety_probe(
            board,
            candidates,
            trigger=trigger,
            pvs_gap=pvs_gap,
            main_rank_stable=self._root_rank_is_stable(root_history),
            completed_depth=completed_depth,
            budget_seconds=budget,
        )

    def _run_root_safety_probe(
        self,
        board: Board,
        candidates: list[Move],
        *,
        trigger: str,
        pvs_gap: int,
        main_rank_stable: bool,
        completed_depth: int,
        budget_seconds: float,
    ) -> RootSafetyProbeResult | None:
        """Compare near-tied moves independently with full root windows.

        The probe deliberately starts with fresh PVS history and a fresh
        normal TT.  Every root move receives a full window, so the earlier
        candidate cannot turn the following candidate into a zero-window
        bound.  A slightly longer threat extension gives continuous forcing
        replies priority, but the result remains heuristic and is never
        promoted to a proof state.
        """
        probe = SearchAI(
            player=self.player,
            max_depth=self.config.max_depth,
            time_limit_seconds=budget_seconds,
            root_candidate_limit=max(2, len(candidates)),
            branch_candidate_limit=self.config.branch_candidate_limit,
            threat_extension_depth=(
                self.config.threat_extension_depth
                + self.config.root_safety_extension_bonus
            ),
            diagnostics=False,
        )
        probe.config = replace(
            probe.config,
            use_aspiration=False,
            use_pvs=False,
            vcf_max_attacker_moves=0,
            root_safety_enabled=False,
        )
        probe._time = TimeManager.start(
            budget_seconds,
            soft_ratio=0.99,
        )
        probe._generation += 1
        probe._counters = SearchCounters()

        target_depth = min(
            self.config.max_depth,
            max(
                self.config.root_safety_min_completed_depth,
                completed_depth + 1,
            ),
        )
        latest: tuple[RootSafetyCandidateAnalysis, ...] = ()
        leader_history: list[Move] = []
        probe_completed_depth = 0
        original_priority = {
            move: len(candidates) - index
            for index, move in enumerate(candidates)
        }

        for depth in range(1, target_depth + 1):
            ranked: list[RootSafetyCandidateAnalysis] = []
            try:
                for move in candidates:
                    probe._check_timeout()
                    board.place(*move, self.player)
                    try:
                        if board.check_win(*move):
                            score = MATE_SCORE
                            child_pv: tuple[Move, ...] = ()
                        else:
                            child_score, child_pv = probe._negamax(
                                board,
                                self.opponent,
                                depth - 1,
                                -INFINITY,
                                INFINITY,
                                ply=1,
                                extension_depth=(
                                    probe.config.threat_extension_depth
                                ),
                            )
                            score = -child_score
                    finally:
                        board.undo()
                    ranked.append(
                        RootSafetyCandidateAnalysis(
                            move=move,
                            score=score,
                            principal_variation=(move, *child_pv),
                        )
                    )
            except SearchTimeout:
                break

            if any(
                self._is_mate_like_score(candidate.score)
                for candidate in ranked
            ):
                ranked = [
                    RootSafetyCandidateAnalysis(
                        move=candidate.move,
                        score=self._heuristic_root_score(
                            board,
                            candidate.move,
                        ),
                        principal_variation=(
                            candidate.principal_variation
                        ),
                    )
                    for candidate in ranked
                ]
            ranked.sort(
                key=lambda candidate: (
                    candidate.score,
                    original_priority.get(candidate.move, 0),
                ),
                reverse=True,
            )
            latest = tuple(ranked)
            probe_completed_depth = depth
            leader_history.append(ranked[0].move)

            if (
                depth >= self.config.root_safety_min_completed_depth
                and len(leader_history) >= 2
                and leader_history[-1] == leader_history[-2]
            ):
                break

        self._counters.root_safety_nodes += probe._counters.nodes
        if not latest:
            return None

        return RootSafetyProbeResult(
            trigger=trigger,
            pvs_gap=pvs_gap,
            main_rank_stable=main_rank_stable,
            completed_depth=probe_completed_depth,
            nodes=probe._counters.nodes,
            candidates=latest,
            leader_history=tuple(leader_history),
        )

    def _apply_root_safety_probe(
        self,
        result: RootResult,
        probe: RootSafetyProbeResult,
    ) -> RootResult:
        return root_safety.apply_probe(self.config, result, probe)

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
        recalibrate = any(
            self._is_mate_like_score(score)
            for score, _ in variations.values()
        )
        analyses: list[DefenseCandidateAnalysis] = []
        for move in candidates:
            score, pv = variations.get(
                move,
                (
                    dict(best.ranked_moves).get(move, -INFINITY),
                    (move,),
                ),
            )
            if recalibrate:
                score = self._heuristic_root_score(board, move)
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

    def _run_mandatory_defense_probe(
        self,
        board: Board,
        player: int,
        candidates: list[Move],
    ) -> DefenseProbeResult | None:
        """Compare multiple mandatory blocks without selective mate claims.

        This probe deliberately uses a wider ordinary reply set and only one
        threat-extension layer.  It is aimed at endpoint choices such as two
        ways to stop the same compound threat, where a narrow forcing probe
        can make both branches look like the same mate score.
        """
        budget = self._mandatory_defense_budget_seconds()
        if budget == 0:
            return None

        probe = SearchAI(
            player=player,
            max_depth=self.config.mandatory_defense_probe_depth,
            time_limit_seconds=budget,
            root_candidate_limit=max(2, len(candidates)),
            branch_candidate_limit=(
                self.config.mandatory_defense_branch_limit
            ),
            threat_extension_depth=(
                self.config.mandatory_defense_extension_depth
            ),
            diagnostics=False,
        )
        probe.config = replace(
            probe.config,
            use_aspiration=False,
            use_pvs=True,
            vcf_max_attacker_moves=0,
            root_safety_enabled=False,
        )
        probe._time = TimeManager.start(
            budget,
            soft_ratio=0.99,
        )
        probe._generation += 1
        probe._counters = SearchCounters()

        ordered = list(candidates)
        best: RootResult | None = None
        completed_depth = 0
        for depth in range(
            1,
            self.config.mandatory_defense_probe_depth + 1,
        ):
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
            or completed_depth
            < self.config.mandatory_defense_probe_depth
        ):
            return None

        variations = {
            move: (score, pv)
            for move, score, pv in best.ranked_variations
        }
        recalibrate = any(
            self._is_mate_like_score(score)
            for score, _ in variations.values()
        )
        original_priority = {
            move: len(candidates) - index
            for index, move in enumerate(candidates)
        }
        analyses = [
            DefenseCandidateAnalysis(
                move=move,
                score=(
                    self._heuristic_root_score(board, move)
                    if recalibrate
                    else variations.get(
                        move,
                        (-INFINITY, (move,)),
                    )[0]
                ),
                status=DefenseProof.UNKNOWN.value,
                principal_variation=variations.get(
                    move,
                    (-INFINITY, (move,)),
                )[1],
            )
            for move in candidates
        ]
        analyses.sort(
            key=lambda item: (
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

    def _mandatory_defense_budget_seconds(self) -> float | None:
        total = self.config.time_limit_seconds
        if total is None:
            return None

        remaining = self._time.remaining_seconds
        if remaining is None:
            return None
        budget = min(
            self.config.mandatory_defense_max_seconds,
            max(
                0.20,
                total
                * self.config.mandatory_defense_time_fraction,
            ),
            max(0.0, remaining - 0.05),
        )
        return budget if budget >= 0.10 else 0

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
        return self._find_vcf_with_deadline(
            board,
            attacker,
            deadline=self._time.sub_deadline(
                self.config.vcf_time_fraction,
                minimum_seconds=0.02,
                maximum_seconds=0.8,
            ),
            root_safety_channel=False,
        )

    def _count_vcf_node(self) -> None:
        self._counters.vcf_nodes += 1

    def _count_root_vcf_node(self) -> None:
        self._counters.root_vcf_nodes += 1

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

    @staticmethod
    def _all_root_candidates_near_forced_loss(
        result: RootResult,
        candidates: list[Move],
    ) -> bool:
        return root_candidates.all_near_forced_loss(
            result,
            candidates,
        )

    def _expand_near_loss_root_candidates(
        self,
        board: Board,
        candidates: list[Move],
    ) -> list[Move]:
        """Widen a collapsed losing root without discarding tactical moves."""
        expanded_limit = max(
            self.config.root_candidate_limit * 2,
            len(candidates) + 1,
        )
        ordinary_candidates = self._ordered_moves(
            board,
            self.player,
            at_root=True,
            ply=0,
            limit=expanded_limit,
            tt_move=self._tt_best_move(board, self.player),
        )
        merged = list(
            dict.fromkeys([*candidates, *ordinary_candidates])
        )
        return self._filter_proven_losing_candidates(merged)

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
        return self._proof_analyzer.analyze_profile(
            board,
            move,
            player,
        )

    def _static_score(self, board: Board, player: int) -> int:
        score = evaluate_board(board, player)
        return max(
            -HEURISTIC_SCORE_LIMIT,
            min(HEURISTIC_SCORE_LIMIT, score),
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
            if is_winning_move(board, row, column, player):
                winning.append((row, column))
        return winning

    def _check_timeout(self) -> None:
        if (
            self._search_phase_deadline is not None
            and time.perf_counter() >= self._search_phase_deadline
        ):
            self._search_phase_timeout_hit = True
            raise SearchTimeout
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
        proof_tt_delta = self._proof_table.stats().delta(
            self._proof_table_start_stats
        )
        threat_stats = (
            ThreatAnalyzerStats()
            if self._proof_analyzer is None
            else self._proof_analyzer.stats()
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
            proof_checked=(
                self._proof_root_result is not None
                or bool(self._proof_candidates)
            ),
            proof_state=(
                "not_checked"
                if self._proof_root_result is None
                else self._proof_root_result.state.value
            ),
            proof_nodes=self._counters.proof_nodes,
            proof_elapsed_seconds=(
                (
                    0.0
                    if self._proof_root_result is None
                    else self._proof_root_result.elapsed_seconds
                )
                + sum(
                    candidate.elapsed_seconds
                    for candidate in self._proof_candidates
                )
            ),
            proof_best_move=(
                None
                if self._proof_root_result is None
                else self._proof_root_result.best_move
            ),
            proof_principal_variation=(
                ()
                if self._proof_root_result is None
                else self._proof_root_result.principal_variation
            ),
            proof_cutoff_reason=(
                None
                if self._proof_root_result is None
                else self._proof_root_result.cutoff_reason
            ),
            proof_candidates=self._proof_candidates,
            proof_tt_queries=proof_tt_delta.queries,
            proof_tt_hits=proof_tt_delta.hits,
            proof_tt_compatible_hits=(
                proof_tt_delta.compatible_hits
            ),
            proof_tt_stores=proof_tt_delta.stores,
            proof_tt_skipped_stores=proof_tt_delta.skipped_stores,
            proof_tt_evictions=proof_tt_delta.evictions,
            proof_tt_size=proof_tt_delta.size,
            threat_candidate_batches=threat_stats.candidate_batches,
            threat_exact_descriptions=(
                threat_stats.exact_descriptions
            ),
            threat_frontier_batches=threat_stats.frontier_batches,
            threat_frontier_descriptions=(
                threat_stats.frontier_descriptions
            ),
            threat_cache_queries=threat_stats.cache_queries,
            threat_cache_hits=threat_stats.cache_hits,
            threat_cache_stores=threat_stats.cache_stores,
            threat_cache_skips=threat_stats.cache_skips,
            root_safety_checked=self._root_safety_probe is not None,
            root_safety_applied=self._root_safety_applied,
            root_safety_trigger=(
                None
                if self._root_safety_probe is None
                else self._root_safety_probe.trigger
            ),
            root_safety_pvs_gap=(
                None
                if self._root_safety_probe is None
                else self._root_safety_probe.pvs_gap
            ),
            root_safety_main_rank_stable=(
                True
                if self._root_safety_probe is None
                else self._root_safety_probe.main_rank_stable
            ),
            root_safety_depth=(
                0
                if self._root_safety_probe is None
                else self._root_safety_probe.completed_depth
            ),
            root_safety_nodes=self._counters.root_safety_nodes,
            root_safety_best_move=(
                None
                if self._root_safety_probe is None
                else self._root_safety_probe.best_move
            ),
            root_safety_leaders=(
                ()
                if self._root_safety_probe is None
                else self._root_safety_probe.leader_history
            ),
            root_safety_candidates=(
                ()
                if self._root_safety_probe is None
                else self._root_safety_probe.candidates
            ),
            root_vcf_checked=self._root_vcf_scan is not None,
            root_vcf_complete=(
                False
                if self._root_vcf_scan is None
                else self._root_vcf_scan.complete
            ),
            root_vcf_nodes=self._counters.root_vcf_nodes,
            root_vcf_baseline_line=(
                ()
                if self._root_vcf_scan is None
                else self._root_vcf_scan.baseline_line
            ),
            root_vcf_candidates=(
                ()
                if self._root_vcf_scan is None
                else self._root_vcf_scan.analyses
            ),
            mate_scores_quarantined=(
                self._root_mate_scores_quarantined
            ),
        )
