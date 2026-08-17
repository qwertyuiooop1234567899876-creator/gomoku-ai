from __future__ import annotations

import time
from dataclasses import replace

from engine.ai import (
    DefenseCandidateAnalysis,
    Move,
    ProofCandidateAnalysis,
    RootSafetyCandidateAnalysis,
    ScoringAI,
)
from engine.board import DIRECTIONS, EMPTY, WHITE, Board
from engine.evaluator import (
    DEFAULT_SEARCH_EVALUATION_CONFIG,
    EvaluationConfig,
    ThreatProfile,
    evaluate_move,
    evaluate_search_position,
    find_winning_moves,
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
from engine import root_candidates, root_policy, root_review, root_safety
from engine.vcf import VCFSearch
from engine.threats import (
    ThreatAnalyzer,
    ThreatFrontier,
    ThreatKind,
)
from engine.search_diagnostics import (
    build_search_analysis,
    compose_search_reason,
)
from engine.zobrist import MASK_64, get_zobrist_table
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
    V0.16.5 搜索 AI。

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
    V0.13.0 在严格 AND/OR 证明中加入 VCF 胜势证书优先收口，
    使跨阶段强制链能在完整防守集上快速得出严格结论；
    PVS 最终首选必须接受独立 Proof 复核，已证败着会从证明
    主线生成通用拦截候选。晚盘只剩单个 VCF 存活点时，根安全
    通道会扫描全部合法着，补齐远端救援点。
    V0.14.0 保留Python调度和严格Proof三态，将一步胜点、局部威胁
    画像、防守反击支撑以及VCF证书搜索迁移到无第三方依赖的C++
    NativeCore。所有原生VCF结果必须经Python逐手重放后才能作为
    证明；未编译原生库时自动回退到V0.13.0参考实现。
    V0.14.1 在对手威胁前沿模式下保留己方冲四反击，并对所有相关
    合法点做批量威胁前沿扫描，补回能提前占据安静长威胁启动点的
    防守着；初始Proof与最终审计共用同一总预算，避免UNKNOWN重复
    消耗两个完整时间片。
    V0.14.3 不改变搜索树、评分或候选上限：叶子一步胜扫描接回复用
    既有 NativeCore 批量内核，强制候选画像改为批量桥接，并缓存同一
    局面的快速排序整数分。纯 Python 模式继续使用同一参考实现。
    V0.14.4 将未证明 Mate 真正拉回普通启发式量纲；根节点完整扫描
    所有相关对手强制点，并仅在普通 PVS 出现未证实高分时扩展一批
    全盘生存候选。最终 Proof 会区分严格生存与 UNKNOWN，不再把超时
    或未完成检查描述成已经确认安全。
    V0.14.6 在威胁前沿的假 Mate 被隔离后，恢复使用搜索前的对手
    威胁图顺序，而不是继续沿用已被假 Mate 污染的 PVS 排名；同时
    有界保留前三个安静前沿预防点，使低排序的战略封锁点仍能进入
    根搜索。两者都只提供候选与 UNKNOWN 仲裁证据，不升级为 Proof。
    V0.14.7 让显著威胁风险可以纠正较旧的窄分支防守探针，并在假
    Mate 隔离后同时保留主动反击点的搜索前真值顺序。对与强前沿
    共享后续节点的安静兄弟点，只增加一个根候选和一条独立深层
    复核通道；普通 PVS 节点不展开安静前沿，严格 Proof 仍保持三态。
    V0.14.8 将 PVS、最低威胁风险、安静预防和进攻延续候选统一送入
    余时复核；深度搜索与攻防前沿净增益共同仲裁，并记录各阶段耗时。
    结构复核仍是启发式，不改变严格 Proof 的三态含义。
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
        evaluation_config: EvaluationConfig = (
            DEFAULT_SEARCH_EVALUATION_CONFIG
        ),
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
        if not isinstance(evaluation_config, EvaluationConfig):
            raise TypeError("evaluation_config 必须是 EvaluationConfig。")
        self._evaluation_config = evaluation_config
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
        self._root_expansion_hold_applied = False
        self._root_vcf_scan: RootVCFScanResult | None = None
        self._root_mate_scores_quarantined = False
        self._root_frontier_priority: tuple[Move, ...] = ()
        self._root_sibling_prevention: tuple[Move, ...] = ()
        self._root_pressure_prevention: tuple[Move, ...] = ()
        self._root_quiet_prevention: tuple[Move, ...] = ()
        self._root_offensive_continuations: tuple[Move, ...] = ()
        self._root_dual_frontier_bridges: tuple[Move, ...] = ()
        self._root_quiet_attack_frontiers: tuple[Move, ...] = ()
        self._root_own_frontiers: tuple[ThreatFrontier, ...] = ()
        self._root_attack_priority: tuple[Move, ...] = ()
        self._root_candidate_sources: dict[
            Move,
            frozenset[root_candidates.CandidateSource],
        ] = {}
        self._root_candidate_mode = root_candidates.RootCandidateMode.ORDINARY
        self._root_frontier_balance: dict[Move, int] = {}
        self._root_frontier_shape: dict[Move, tuple[int, int, int, int]] = {}
        self._defense_risk_override_applied = False
        self._quiet_frontier_extension_enabled = False
        self._root_heuristic_score_cache: dict[
            tuple[int, Move],
            int,
        ] = {}
        self._quick_order_cache: dict[
            tuple[int, int, int, int],
            int,
        ] = {}
        self._search_phase_deadline: float | None = None
        self._search_phase_timeout_hit = False
        self._final_proof_checked = False
        self._final_proof_state = "not_checked"
        self._final_proof_completed = False
        self._final_proof_rejected: tuple[Move, ...] = ()
        self._final_proof_selected: Move | None = None
        self._final_proof_selection_basis = "not_checked"
        self._final_proof_expected_candidates = 0
        self._phase_timings: dict[str, float] = {}

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
            plan = self._run_timed_phase(
                "candidate_generation",
                lambda: self._prepare_root_candidate_plan(
                    board,
                    legal_moves,
                ),
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
        allow_near_loss_expansion = plan.allow_near_loss_expansion
        defense_probe = plan.defense_probe
        tactical_reason = plan.reason

        root_vcf_scan = self._run_timed_phase(
            "root_vcf_scan",
            lambda: self._run_root_opponent_vcf_scan(
                board,
                search_candidates,
            ),
        )
        if root_vcf_scan is not None:
            self._root_vcf_scan = root_vcf_scan
            search_candidates = self._filter_root_vcf_candidates(
                list(root_vcf_scan.candidates)
            )
        self._prepare_root_verification_candidates(search_candidates)

        proof_win = self._run_timed_phase(
            "initial_proof",
            lambda: self._run_proof_arbitration(
                board,
                search_candidates,
                search_own_win=bool(own_forcing_moves),
            ),
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
        root_expansion_reason = outcome.root_expansion_reason

        best_result = self._run_timed_phase(
            "final_proof",
            lambda: self._run_final_proof_audit(board, best_result),
        )
        search_candidates = root_candidates.merge_unique(search_candidates, (best_result.move,))

        reason = (
            f"{tactical_reason}（完成深度 {completed_depth}）"
            if completed_depth > 0
            else f"{tactical_reason}（时间不足，使用快速回退）"
        )
        reason = compose_search_reason(
            reason,
            expansion_reason=root_expansion_reason,
            expansion_hold_applied=self._root_expansion_hold_applied,
            root_vcf_scan=self._root_vcf_scan,
            mate_scores_quarantined=self._root_mate_scores_quarantined,
            defense_risk_override=self._defense_risk_override_applied,
            root_safety_probe=self._root_safety_probe,
            root_safety_applied=self._root_safety_applied,
            final_proof_checked=self._final_proof_checked,
            final_proof_state=self._final_proof_state,
            final_proof_completed=self._final_proof_completed,
            final_proof_rejected=self._final_proof_rejected,
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
        self._root_expansion_hold_applied = False
        self._root_vcf_scan = None
        self._root_mate_scores_quarantined = False
        self._root_frontier_priority = ()
        self._root_sibling_prevention = ()
        self._root_pressure_prevention = ()
        self._root_quiet_prevention = ()
        self._root_offensive_continuations = ()
        self._root_dual_frontier_bridges = ()
        self._root_quiet_attack_frontiers = ()
        self._root_own_frontiers = ()
        self._root_attack_priority = ()
        self._root_candidate_sources.clear()
        self._root_candidate_mode = root_candidates.RootCandidateMode.ORDINARY
        self._root_frontier_balance.clear()
        self._root_frontier_shape.clear()
        self._defense_risk_override_applied = False
        self._quiet_frontier_extension_enabled = False
        self._root_heuristic_score_cache.clear()
        self._quick_order_cache.clear()
        self._search_phase_deadline = None
        self._search_phase_timeout_hit = False
        self._final_proof_checked = False
        self._final_proof_state = "not_checked"
        self._final_proof_completed = False
        self._final_proof_rejected = ()
        self._final_proof_selected = None
        self._final_proof_selection_basis = "not_checked"
        self._final_proof_expected_candidates = 0
        self._phase_timings.clear()
        self._proof_table_start_stats = self._proof_table.stats()
        self._proof_analyzer = self._new_threat_analyzer()
        self._decay_history()
        self._prune_transposition_table()

    def _run_timed_phase(self, phase: str, operation):
        """Run one coordinator phase and accumulate wall-clock time."""
        started_at = time.perf_counter()
        try:
            return operation()
        finally:
            self._phase_timings[phase] = (
                self._phase_timings.get(phase, 0.0)
                + time.perf_counter()
                - started_at
            )

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
        search_candidates, defense_probe = (
            self._maybe_run_post_filter_defense_probe(
                board,
                search_candidates,
                candidate_mode=self._root_candidate_mode,
                existing_probe=defense_probe,
            )
        )
        self._set_final_proof_expected_candidates(search_candidates)

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
        root_expansion_reason: str | None = None
        pvs_started_at = time.perf_counter()

        for depth in range(1, self.config.max_depth + 1):
            if depth > 1 and self._time.soft_expired():
                self._interrupted_depth = depth
                search_completed = False
                stop_reason = "soft_deadline"
                break

            self._search_phase_deadline = None
            self._search_phase_timeout_hit = False
            reserve = self._final_proof_reserve_seconds()
            if completed_depth > 0 and (
                root_candidates_expanded
                or self._root_mate_scores_quarantined
                or self._root_safety_trigger(
                    best_result,
                    root_history,
                )
                is not None
            ):
                # Root review and final Proof are serial phases.  Taking only
                # their maximum let the next iterative depth consume the
                # entire review allowance and left exactly the Proof reserve.
                reserve = root_policy.serial_verification_reserve(
                    final_proof_seconds=reserve,
                    root_review_seconds=(
                        self._root_safety_budget_seconds()
                    ),
                )
            if reserve > 0 and self._time.hard_deadline is not None:
                self._search_phase_deadline = (
                    self._time.hard_deadline - reserve
                )
                if time.perf_counter() >= self._search_phase_deadline:
                    self._interrupted_depth = depth
                    search_completed = False
                    stop_reason = "verification_reserve"
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
                        self._set_final_proof_expected_candidates(
                            search_candidates
                        )
                        root_candidates_expanded = True
                        root_expansion_reason = "near_forced_loss"
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
                    and self.config.max_depth
                    >= self.config.root_survival_min_depth
                    and self._has_unverified_root_advantage(result)
                ):
                    expanded_candidates = (
                        self._expand_unverified_advantage_root_candidates(
                            board,
                            search_candidates,
                        )
                    )
                    if len(expanded_candidates) > len(search_candidates):
                        search_candidates = (
                            self._filter_root_vcf_candidates(
                                expanded_candidates
                            )
                        )
                        self._register_expanded_candidates_as_unknown(
                            search_candidates
                        )
                        self._set_final_proof_expected_candidates(
                            search_candidates
                        )
                        root_candidates_expanded = True
                        root_expansion_reason = "unverified_advantage"
                        self._reserve_expansion_review_time()
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
                    preserved_order=(
                        root_candidates.merge_unique(
                            self._root_frontier_priority,
                            search_candidates,
                        )
                        if preserve_frontier_order
                        else None
                    ),
                )
            except SearchTimeout:
                self._interrupted_depth = depth
                search_completed = False
                stop_reason = (
                    "verification_reserve"
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
                if self._proof_candidates:
                    probe_result = result
                    result = self._apply_proof_tiebreak(
                        probe_result,
                        preserve_order=False,
                    )
                    if self._is_unknown_risk_override(
                        probe_result,
                        result,
                    ):
                        # The bounded defense probe and threat-risk channel
                        # are both heuristic here.  A material risk gap must
                        # still be allowed to correct the older probe instead
                        # of being skipped solely because it ran first.
                        self._defense_risk_override_applied = True
            elif self._proof_candidates:
                pvs_result = result
                result = self._apply_proof_tiebreak(
                    pvs_result,
                    preserve_order=preserve_frontier_order,
                )
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
        self._phase_timings["main_pvs"] = (
            self._phase_timings.get("main_pvs", 0.0)
            + time.perf_counter()
            - pvs_started_at
        )
        review_started_at = time.perf_counter()
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
        elif (
            not search_completed
            and completed_depth > 0
            and not self._root_mate_scores_quarantined
        ):
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

        if (
            root_expansion_reason == "unverified_advantage"
            and self._is_unverified_expansion_move(best_result.move)
            and not self._expansion_leader_has_base_review(
                best_result.move
            )
        ):
            held_result = self._hold_unverified_expansion_leader(best_result)
            self._root_expansion_hold_applied = (
                held_result.move != best_result.move
            )
            best_result = held_result

        dynamic_probe = self._maybe_run_dynamic_root_review(
            board,
            best_result,
            search_candidates,
            completed_depth=completed_depth,
        )
        if dynamic_probe is not None:
            self._root_safety_probe = dynamic_probe
            revised = self._apply_root_safety_probe(
                best_result,
                dynamic_probe,
            )
            self._root_safety_applied = (
                revised.move != best_result.move
            )
            best_result = revised
            if self._is_unverified_expansion_move(best_result.move):
                self._root_expansion_hold_applied = False
        elif (
            not search_completed
            and completed_depth > 0
            and self._root_mate_scores_quarantined
        ):
            # A quarantined selective root needs the source-aware equal-window
            # review first.  Fall back to the older two-score probe only when
            # that review could not start or finish inside its reserve.
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

        self._phase_timings["root_review"] = (
            self._phase_timings.get("root_review", 0.0)
            + time.perf_counter()
            - review_started_at
        )

        return IterativeSearchOutcome(
            result=best_result,
            candidates=search_candidates,
            completed_depth=completed_depth,
            search_completed=search_completed,
            stop_reason=stop_reason,
            root_candidates_expanded=root_candidates_expanded,
            root_expansion_reason=root_expansion_reason,
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
        source_groups: list[
            tuple[
                root_candidates.CandidateSource,
                list[Move] | tuple[Move, ...],
            ]
        ] = []
        ordinary_candidates: list[Move] = []
        frontier_candidates: list[Move] = []
        counterattack_truth_moves: list[Move] = []
        forcing_counterattacks: list[Move] = []
        prevention_moves: list[Move] = []
        pressure_prevention_moves: list[Move] = []
        offensive_continuations: list[Move] = []
        dual_frontier_bridges: list[Move] = []
        quiet_attack_frontiers: list[Move] = []

        root_pool = self._root_profile_pool(board, legal_moves)
        relevant_pool = self._root_relevant_pool(board, legal_moves)
        full_own_profiles = self._profile_moves_timed(
            board,
            relevant_pool,
            self.player,
        )
        own_profiles = {
            move: full_own_profiles[move]
            for move in root_pool
            if move in full_own_profiles
        }
        own_profiles.update(
            root_candidates.tactical_root_profiles(full_own_profiles)
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
        full_opponent_profiles = self._profile_moves_timed(
            board,
            relevant_pool,
            self.opponent,
        )
        opponent_forcing_moves = [
            move
            for move, profile in full_opponent_profiles.items()
            if profile.forced_win
        ]
        for move in opponent_forcing_moves:
            opponent_profiles.setdefault(
                move,
                full_opponent_profiles[move],
            )
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
        frontier_truth_moves = list(opponent_frontier_moves)
        opponent_pressure_frontiers = ()
        if (
            opponent_frontier_moves
            and self.config.max_depth
            >= self.config.root_quiet_prevention_min_depth
        ):
            opponent_pressure_frontiers = (
                self._proof_analyzer.generate_attack_frontiers(
                    board,
                    self.opponent,
                    frontier_limit=max(1, board.empty_count),
                    continuation_limit=max(
                        12,
                        self.config.frontier_reply_limit * 2,
                    ),
                    scan_all_relevant=True,
                    stop_requested=self._time.hard_expired,
                )
            )
            pressure_priority = {
                frontier.gain_move: index
                for index, frontier in enumerate(
                    opponent_pressure_frontiers
                )
            }
            fallback_priority = len(pressure_priority)
            frontier_truth_moves.sort(
                key=lambda move: pressure_priority.get(
                    move,
                    fallback_priority,
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
            source_groups = [
                (
                    root_candidates.CandidateSource.MANDATORY_DEFENSE,
                    opponent_forcing_moves,
                ),
                (
                    root_candidates.CandidateSource.OWN_FORCING,
                    own_forcing_moves,
                ),
            ]
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
            source_groups = [
                (
                    root_candidates.CandidateSource.OWN_FORCING,
                    own_forcing_moves,
                )
            ]
        elif (
            candidate_mode
            is root_candidates.RootCandidateMode.MANDATORY_DEFENSE
        ):
            allow_near_loss_expansion = False
            # A root collapsed to one direct block is especially vulnerable
            # to hiding a tempo defense.  Multiple direct defenses already
            # receive their dedicated bounded VCT comparison; widening those
            # roots as well would dilute that budget with unrelated fours.
            mandatory_own_profiles = (
                full_own_profiles
                if len(opponent_forcing_moves) == 1
                else {}
            )
            forcing_counterattacks = (
                self._order_specific_moves(
                    board,
                    root_candidates.forcing_counterattack_moves(
                        mandatory_own_profiles
                    ),
                    self.player,
                    ply=0,
                    tt_move=None,
                    full_evaluation=True,
                )
                if self.config.max_depth
                >= self.config.root_forcing_counterattack_min_depth
                else []
            )
            counterattack_truth_moves = self._order_specific_moves(
                board,
                root_candidates.active_counterattack_moves(
                    mandatory_own_profiles
                ),
                self.player,
                ply=0,
                tt_move=None,
                full_evaluation=True,
            )[
                : self.config.root_mandatory_active_counterattack_limit
            ]
            self._root_attack_priority = tuple(counterattack_truth_moves)
            search_candidates = self._order_specific_moves(
                board,
                root_candidates.mandatory_defense_moves(
                    defense_moves=opponent_forcing_moves,
                    forcing_counterattack_moves=forcing_counterattacks,
                    active_counterattack_moves=counterattack_truth_moves,
                    limit=self.config.root_candidate_limit,
                ),
                self.player,
                ply=0,
                tt_move=self._tt_best_move(board, self.player),
                full_evaluation=True,
            )
            source_groups = [
                (
                    root_candidates.CandidateSource.MANDATORY_DEFENSE,
                    opponent_forcing_moves,
                ),
                (
                    root_candidates.CandidateSource.FORCING_COUNTERATTACK,
                    forcing_counterattacks,
                ),
                (
                    root_candidates.CandidateSource.ACTIVE_COUNTERATTACK,
                    counterattack_truth_moves,
                ),
            ]
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
            # Search traversal keeps its established efficient order.  The
            # independent opponent-pressure order is retained separately and
            # is consulted only if selective mate values are quarantined.
            frontier_candidates = self._order_specific_moves(
                board,
                opponent_frontier_moves,
                self.player,
                ply=0,
                tt_move=self._tt_best_move(board, self.player),
                full_evaluation=True,
            )[
                : self.config.frontier_reply_limit
            ]
            ordinary_candidates = (
                self._ordered_moves(
                    board,
                    self.player,
                    at_root=True,
                    ply=0,
                    tt_move=self._tt_best_move(board, self.player),
                    use_search_heuristics=False,
                )
                if len(frontier_candidates) == 1
                else []
            )
            counterattack_truth_moves = (
                root_candidates.active_counterattack_moves(
                    own_profiles
                )
            )
            counterattacks = self._order_specific_moves(
                board,
                counterattack_truth_moves,
                self.player,
                ply=0,
                tt_move=None,
                full_evaluation=True,
            )
            forcing_counterattacks = (
                self._order_specific_moves(
                    board,
                    root_candidates.forcing_counterattack_moves(
                        own_profiles
                    ),
                    self.player,
                    ply=0,
                    tt_move=None,
                    full_evaluation=True,
                )
                if self.config.max_depth
                >= self.config.root_forcing_counterattack_min_depth
                else []
            )
            prevention_moves = self._quiet_frontier_prevention_moves(
                board,
                frontiers=opponent_pressure_frontiers,
            )
            sibling_prevention_moves = (
                root_candidates.quiet_frontier_sibling_prevention_moves(
                    frontiers=opponent_pressure_frontiers,
                    anchor_moves=counterattack_truth_moves,
                    strong_rank=(
                        self.config.root_quiet_prevention_min_rank
                    ),
                    minimum_continuations=(
                        self.config
                        .root_quiet_sibling_min_continuations
                    ),
                    limit=(
                        self.config
                        .root_quiet_sibling_prevention_limit
                    ),
                )
            )
            self._root_sibling_prevention = tuple(
                sibling_prevention_moves
            )
            prevention_moves = root_candidates.merge_unique(
                prevention_moves,
                sibling_prevention_moves,
            )
            self._root_quiet_prevention = tuple(prevention_moves)
            offensive_continuations = (
                self._offensive_continuation_moves(
                    board,
                    opponent_frontiers=opponent_pressure_frontiers,
                )
            )
            self._root_offensive_continuations = tuple(
                offensive_continuations
            )
            dual_frontier_bridges = (
                root_candidates.dual_frontier_gain_bridges(
                    own_frontiers=self._root_own_frontiers,
                    opponent_frontiers=opponent_pressure_frontiers,
                    minimum_own_rank=(
                        self.config.root_dual_frontier_min_own_rank
                    ),
                    minimum_opponent_rank=(
                        self.config.root_dual_frontier_min_opponent_rank
                    ),
                    minimum_own_continuations=(
                        self.config
                        .root_dual_frontier_min_own_continuations
                    ),
                    minimum_opponent_continuations=(
                        self.config
                        .root_dual_frontier_min_opponent_continuations
                    ),
                    limit=self.config.root_dual_frontier_bridge_limit,
                )
            )
            self._root_dual_frontier_bridges = tuple(
                dual_frontier_bridges
            )
            quiet_attack_frontiers = (
                root_candidates.quiet_attack_frontier_moves(
                    frontiers=self._root_own_frontiers,
                    minimum_rank=(
                        self.config.root_quiet_attack_frontier_min_rank
                    ),
                    minimum_continuations=(
                        self.config
                        .root_quiet_attack_frontier_min_continuations
                    ),
                    limit=self.config.root_quiet_attack_frontier_limit,
                    covered_moves=root_candidates.merge_unique(
                        frontier_candidates,
                        ordinary_candidates,
                        counterattacks,
                        forcing_counterattacks,
                        prevention_moves,
                        offensive_continuations,
                        dual_frontier_bridges,
                    ),
                )
            )
            pressure_prevention_moves = (
                root_candidates.pressure_prevention_moves(
                    frontiers=opponent_pressure_frontiers,
                    covered_moves=root_candidates.merge_unique(
                        frontier_candidates,
                        ordinary_candidates,
                        counterattacks,
                        forcing_counterattacks,
                        prevention_moves,
                        offensive_continuations,
                        dual_frontier_bridges,
                    ),
                    strong_rank=self.config.root_quiet_prevention_min_rank,
                    minimum_continuations=max(
                        4,
                        self.config
                        .root_offensive_continuation_min_continuations
                        * 2,
                    ),
                    limit=1,
                )
            )
            self._root_pressure_prevention = tuple(
                pressure_prevention_moves
            )
            search_candidates = root_candidates.frontier_defense_moves(
                frontier_moves=frontier_candidates,
                ordinary_moves=ordinary_candidates,
                counterattack_moves=counterattacks,
                limit=self.config.root_candidate_limit,
                forcing_counterattack_moves=forcing_counterattacks,
                pressure_prevention_moves=pressure_prevention_moves,
                prevention_moves=prevention_moves,
                offensive_continuation_moves=offensive_continuations,
                dual_frontier_moves=dual_frontier_bridges,
            )
            spare_slots = max(
                0,
                self.config.root_candidate_limit - len(search_candidates),
            )
            quiet_attack_frontiers = [
                move
                for move in quiet_attack_frontiers
                if move not in search_candidates
            ][:spare_slots]
            self._root_quiet_attack_frontiers = tuple(
                quiet_attack_frontiers
            )
            search_candidates = root_candidates.merge_unique(
                search_candidates,
                quiet_attack_frontiers,
            )
            source_groups = [
                (
                    root_candidates.CandidateSource.THREAT_FRONTIER,
                    frontier_candidates,
                ),
                (
                    root_candidates.CandidateSource.ORDINARY,
                    ordinary_candidates,
                ),
                (
                    root_candidates.CandidateSource.ACTIVE_COUNTERATTACK,
                    counterattacks,
                ),
                (
                    root_candidates.CandidateSource.OWN_FORCING,
                    forcing_counterattacks,
                ),
                (
                    root_candidates.CandidateSource.PRESSURE_PREVENTION,
                    pressure_prevention_moves,
                ),
                (
                    root_candidates.CandidateSource.QUIET_PREVENTION,
                    prevention_moves,
                ),
                (
                    root_candidates.CandidateSource.OFFENSIVE_CONTINUATION,
                    offensive_continuations,
                ),
                (
                    root_candidates.CandidateSource.DUAL_FRONTIER_BRIDGE,
                    dual_frontier_bridges,
                ),
                (
                    root_candidates.CandidateSource.QUIET_ATTACK_FRONTIER,
                    quiet_attack_frontiers,
                ),
            ]
            preserve_frontier_order = True
            tactical_reason = "多重威胁启动点候选的 PVS 防守"
            if ordinary_candidates:
                tactical_reason += "；单前沿已补入普通候选"
            if counterattacks:
                tactical_reason += "；已补入主动反击点"
            if forcing_counterattacks:
                tactical_reason += "；已补入强制反击点"
            if prevention_moves:
                tactical_reason += "；已补入安静前沿预防点"
            if offensive_continuations:
                tactical_reason += "；已补入进攻延续桥接点"
            if quiet_attack_frontiers:
                tactical_reason += "；已补入安静进攻前沿"
            self._root_frontier_priority = tuple(
                root_candidates.merge_unique(
                    (
                        frontier_truth_moves
                        if opponent_pressure_frontiers
                        else search_candidates
                    ),
                    counterattack_truth_moves,
                    offensive_continuations,
                    quiet_attack_frontiers,
                    search_candidates,
                )
            )
        else:
            search_candidates = self._ordered_moves(
                board,
                self.player,
                at_root=True,
                ply=0,
            )
            tactical_reason = "PVS 搜索最佳变化"
            source_groups = [
                (
                    root_candidates.CandidateSource.ORDINARY,
                    search_candidates,
                )
            ]

        provenance = root_candidates.with_sources(source_groups)
        sources_by_move = {
            entry.move: entry.sources for entry in provenance
        }
        entries = tuple(
            root_candidates.CandidateEntry(
                move,
                sources_by_move.get(
                    move,
                    frozenset({root_candidates.CandidateSource.ORDINARY}),
                ),
            )
            for move in search_candidates
        )
        self._root_candidate_sources = {
            entry.move: entry.sources for entry in entries
        }
        self._root_candidate_mode = candidate_mode
        return root_candidates.RootCandidatePlan(
            moves=search_candidates,
            own_profiles=own_profiles,
            opponent_profiles=opponent_profiles,
            own_forcing_moves=own_forcing_moves,
            preserve_frontier_order=preserve_frontier_order,
            allow_near_loss_expansion=allow_near_loss_expansion,
            defense_probe=defense_probe,
            reason=tactical_reason,
            entries=entries,
            mode=candidate_mode,
        )

    def _maybe_run_post_filter_defense_probe(
        self,
        board: Board,
        candidates: list[Move],
        *,
        candidate_mode: root_candidates.RootCandidateMode,
        existing_probe: DefenseProbeResult | None,
    ) -> tuple[list[Move], DefenseProbeResult | None]:
        """Give a filtered mandatory root one bounded VCT opportunity.

        Candidate generation may exceed the narrow probe cap, while root VCF
        and strict Proof subsequently remove unsafe branches.  Re-evaluate the
        gate exactly once on that effective survivor set; ordinary roots and
        roots with an existing probe remain unchanged.
        """
        if (
            existing_probe is not None
            or candidate_mode
            is not root_candidates.RootCandidateMode.MANDATORY_DEFENSE
            or not 2 <= len(candidates)
            <= self.config.defense_vct_max_candidates
            or not any(
                root_candidates.CandidateSource.MANDATORY_DEFENSE
                in self._root_candidate_sources.get(move, ())
                for move in candidates
            )
        ):
            return candidates, existing_probe

        probe = self._run_defense_vct_probe(
            board,
            self.player,
            candidates,
        )
        if probe is None:
            return candidates, None

        self._defense_probe = probe
        probe_order = [candidate.move for candidate in probe.candidates]
        ordered = [
            *probe_order,
            *(move for move in candidates if move not in probe_order),
        ]
        return ordered, probe

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
            exhaustive_rescue_enabled=True,
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
                    use_vcf_oracle=True,
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

        probed = root_candidates.source_diverse_subset(
            candidates,
            self._root_candidate_sources,
            limit=self.config.proof_root_candidate_limit,
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
            candidate_seconds = root_policy.proof_candidate_slice_seconds(
                remaining_seconds=deadline - now,
                checks_left=candidates_left,
                maximum_seconds=self.config.proof_candidate_max_seconds,
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
                    use_vcf_oracle=True,
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
        self._register_unprobed_proof_candidates(
            candidates,
            cutoff_reason="initial_proof_unprobed",
        )
        return self._proof_root_result

    def _set_final_proof_expected_candidates(
        self,
        candidates: list[Move],
    ) -> None:
        self._final_proof_expected_candidates = min(
            len(tuple(dict.fromkeys(candidates))),
            self.config.proof_final_candidate_limit,
        )

    def _prepare_root_verification_candidates(
        self,
        candidates: list[Move],
    ) -> None:
        self._register_expanded_candidates_as_unknown(candidates)
        self._set_final_proof_expected_candidates(candidates)

    def _proof_budget_seconds(self) -> float:
        total = self.config.time_limit_seconds
        remaining = self._time.remaining_seconds
        if total is None or remaining is None or total < 4.0:
            return 0.0
        total_proof_cap = min(
            self.config.proof_max_seconds,
            total * self.config.proof_time_fraction,
        )
        final_reserve = self._final_proof_reserve_seconds()
        initial_cap = min(
            self.config.proof_initial_max_seconds,
            max(0.0, total_proof_cap - final_reserve),
        )
        budget = min(
            initial_cap,
            max(0.0, remaining - final_reserve - 0.1),
        )
        return budget if budget >= 0.1 else 0.0

    def _final_proof_reserve_seconds(self) -> float:
        total = self.config.time_limit_seconds
        if (
            not self.config.proof_final_check_enabled
            or total is None
            or total < 4.0
        ):
            return 0.0
        reserve = min(
            self.config.proof_final_max_seconds,
            total * self.config.proof_final_time_fraction,
        )
        if self._final_proof_expected_candidates > 0:
            reserve = min(
                reserve,
                self._final_proof_expected_candidates
                * self.config.proof_candidate_max_seconds,
            )
        return (
            reserve
            if reserve >= self.config.proof_final_min_seconds
            else 0.0
        )

    def _final_proof_budget_seconds(self) -> float:
        reserve = self._final_proof_reserve_seconds()
        remaining = self._time.remaining_seconds
        if reserve <= 0 or remaining is None:
            return 0.0
        return max(0.0, min(reserve, remaining - 0.02))

    def _run_final_proof_audit(
        self,
        board: Board,
        result: RootResult,
    ) -> RootResult:
        """Strictly recheck the move that will actually be returned.

        Earlier proof slices are scheduled before PVS and therefore cannot
        know its eventual leader.  This reserved pass follows the real final
        order.  A proved opponent win rejects the candidate; UNKNOWN remains
        eligible but does not stop the audit from looking for a strictly safe
        alternative.  Points from the losing certificate are inserted as
        generic interception candidates, so the recovery is property-based
        rather than tied to one recorded coordinate.
        """
        seconds = self._final_proof_budget_seconds()
        if seconds < self.config.proof_final_min_seconds:
            return result

        deadline = time.perf_counter() + seconds
        self._final_proof_checked = True
        proof_by_move = {
            candidate.move: candidate
            for candidate in self._proof_candidates
        }
        root_vcf_by_move = (
            {}
            if self._root_vcf_scan is None
            else {
                candidate.move: candidate
                for candidate in self._root_vcf_scan.analyses
            }
        )
        queue = [
            result.move,
            *(
                move
                for group in self._critical_root_review_groups(
                    tuple(move for move, _score in result.ranked_moves)
                )
                for move in group
                if move != result.move
            ),
            *(
                move
                for move, _score in result.ranked_moves
                if move != result.move
            ),
        ]
        seen: set[Move] = set()
        rejected: list[Move] = []
        unknown: list[Move] = []
        selected: Move | None = None
        selection_basis = "not_checked"
        checks = 0

        while (
            queue
            and checks < self.config.proof_final_candidate_limit
            and time.perf_counter() < deadline
        ):
            move = queue.pop(0)
            if move in seen or not board.is_empty(*move):
                continue
            seen.add(move)

            vcf_analysis = root_vcf_by_move.get(move)
            if (
                vcf_analysis is not None
                and vcf_analysis.status
                == root_safety.RootCandidateSafety.PROVEN_LOSS.value
            ):
                rejected.append(move)
                self._prepend_certificate_intercepts(
                    queue,
                    board,
                    vcf_analysis.principal_variation,
                    seen,
                )
                continue

            known = proof_by_move.get(move)
            if (
                known is not None
                and known.state == ProofState.PROVEN_WIN.value
            ):
                rejected.append(move)
                self._prepend_certificate_intercepts(
                    queue,
                    board,
                    known.principal_variation,
                    seen,
                )
                continue
            if (
                known is not None
                and known.state == ProofState.PROVEN_LOSS.value
            ):
                selected = move
                selection_basis = "strict_survivor"
                break

            checks += 1
            now = time.perf_counter()
            queued_unseen = len(
                {
                    candidate
                    for candidate in queue
                    if candidate not in seen
                    and board.is_empty(*candidate)
                }
            )
            checks_left = root_policy.pending_proof_checks(
                candidate_limit=self.config.proof_final_candidate_limit,
                checks_completed=checks,
                queued_unseen=queued_unseen,
            )
            candidate_deadline = min(
                deadline,
                now
                + root_policy.proof_candidate_slice_seconds(
                    remaining_seconds=deadline - now,
                    checks_left=checks_left,
                    maximum_seconds=(
                        self.config.proof_candidate_max_seconds
                    ),
                ),
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
                    use_vcf_oracle=True,
                    deadline=candidate_deadline,
                ),
                analyzer=self._proof_analyzer,
                table=self._proof_table,
                clock=time.perf_counter,
            )
            proof = candidate_search.search_after_move(
                board,
                move=move,
                mover=self.player,
                attacker=self.opponent,
                side_to_move=self.opponent,
            )
            self._counters.proof_nodes += proof.nodes
            analysis = ProofCandidateAnalysis(
                move=move,
                state=proof.state.value,
                completed=proof.completed,
                nodes=proof.nodes,
                elapsed_seconds=proof.elapsed_seconds,
                cutoff_reason=proof.cutoff_reason,
                principal_variation=proof.principal_variation,
                threat_risk=(
                    None if known is None else known.threat_risk
                ),
                phase="final_selection",
            )
            self._upsert_proof_candidate(analysis)
            proof_by_move[move] = analysis
            self._final_proof_checked = True

            if proof.state is ProofState.PROVEN_WIN:
                rejected.append(move)
                self._prepend_certificate_intercepts(
                    queue,
                    board,
                    proof.principal_variation,
                    seen,
                )
                continue

            if proof.state is ProofState.PROVEN_LOSS:
                selected = move
                selection_basis = "strict_survivor"
                break

            # UNKNOWN is not safety evidence.  Keep it as a fallback while
            # spending the remaining bounded audit on later candidates that
            # may have a strict survival result.
            unknown.append(move)

        self._final_proof_rejected = tuple(dict.fromkeys(rejected))
        if selected is None:
            selected = root_review.preferred_unknown_move(
                tuple(
                    move for move in unknown if board.is_empty(*move)
                ),
                self._root_pressure_prevention,
            )
            if selected is not None:
                selection_basis = "checked_unknown"
            if selected is None:
                emergency = next(
                    (
                        move
                        for move in queue
                        if move not in seen and board.is_empty(*move)
                    ),
                    None,
                )
                if emergency is None:
                    selected = result.move
                    selection_basis = "proved_loss_fallback"
                else:
                    selected = emergency
                    selection_basis = "emergency_unknown"
                    emergency_analysis = ProofCandidateAnalysis(
                        move=selected,
                        state=ProofState.UNKNOWN.value,
                        completed=False,
                        nodes=0,
                        elapsed_seconds=0.0,
                        cutoff_reason="final_proof_budget_exhausted",
                        threat_risk=None,
                        phase="final_selection",
                    )
                    self._upsert_proof_candidate(emergency_analysis)
                    proof_by_move[selected] = emergency_analysis
        self._final_proof_selected = selected
        self._final_proof_selection_basis = selection_basis
        selected_analysis = proof_by_move.get(selected)
        if selected_analysis is None:
            self._final_proof_state = "not_checked"
            self._final_proof_completed = False
        else:
            self._final_proof_state = selected_analysis.state
            self._final_proof_completed = (
                selected_analysis.completed
                and selected_analysis.state
                != ProofState.UNKNOWN.value
            )
        if selected == result.move:
            return result

        scores = dict(result.ranked_moves)
        return root_policy.promote_root_move(
            result,
            selected,
            score=scores.get(
                selected,
                self._heuristic_root_score(board, selected),
            ),
        )

    def _prepend_certificate_intercepts(
        self,
        queue: list[Move],
        board: Board,
        line: tuple[Move, ...],
        seen: set[Move],
    ) -> None:
        attacker_points = line[::2]
        defender_points = line[1::2]
        additions = [
            move
            for move in (*attacker_points, *defender_points)
            if move not in seen and board.is_empty(*move)
        ]
        if additions:
            for move in additions:
                current = self._root_candidate_sources.get(
                    move,
                    frozenset(),
                )
                self._root_candidate_sources[move] = frozenset(
                    (
                        *current,
                        root_candidates.CandidateSource.CERTIFICATE_INTERCEPT,
                    )
                )
            queue[:0] = [
                *dict.fromkeys(additions),
            ]

    def _upsert_proof_candidate(
        self,
        analysis: ProofCandidateAnalysis,
    ) -> None:
        self._proof_candidates = (
            analysis,
            *(
                candidate
                for candidate in self._proof_candidates
                if candidate.move != analysis.move
            ),
        )

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
        for move in candidates:
            self._root_candidate_sources.setdefault(
                move,
                frozenset({root_candidates.CandidateSource.ROOT_EXPANSION}),
            )
        self._register_unprobed_proof_candidates(
            candidates,
            cutoff_reason="root_expansion_unprobed",
        )

    def _register_unprobed_proof_candidates(
        self,
        candidates: list[Move],
        *,
        cutoff_reason: str,
    ) -> None:
        """Represent every unsearched root candidate as explicit UNKNOWN."""
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
                cutoff_reason=cutoff_reason,
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
                score = self._static_score(
                    board,
                    perspective=self.player,
                    side_to_move=self.opponent,
                )
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
        preserved_order: list[Move] | None = None,
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
                preserved_order=preserved_order,
                preserve_order_margin=(
                    self.config.root_frontier_truth_score_margin
                ),
            )
        )
        self._root_mate_scores_quarantined |= quarantined
        return revised

    def _apply_proof_tiebreak(
        self,
        result: RootResult,
        *,
        preserve_order: bool = False,
    ) -> RootResult:
        return root_policy.apply_proof_tiebreak(
            self.config,
            result,
            self._proof_candidates,
            preserve_order=preserve_order,
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
        budget = self._risk_override_budget_seconds()
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
        remaining = self._time.remaining_seconds
        if remaining is not None:
            remaining = max(
                0.0,
                remaining - self._final_proof_reserve_seconds(),
            )
        return root_safety.budget_seconds(
            self.config,
            remaining_seconds=remaining,
        )

    def _risk_override_budget_seconds(self) -> float:
        """Borrow a bounded slice of the already reserved final audit time."""
        remaining = self._time.remaining_seconds
        if remaining is None:
            return 0.0
        shared_cap = (
            self._final_proof_reserve_seconds()
            * self.config.root_risk_override_shared_fraction
        )
        return root_safety.budget_seconds(
            self.config,
            remaining_seconds=min(remaining, shared_cap),
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

    def _run_quiet_sibling_probe(
        self,
        board: Board,
        result: RootResult,
        candidates: list[Move],
        *,
        completed_depth: int,
    ) -> RootSafetyProbeResult | None:
        """Review structural quiet siblings with bounded deep extensions.

        Normal PVS deliberately does not expand quiet gain points at every
        leaf.  A sibling admitted by the root threat graph can therefore sit
        far below a flashy active move even though the opponent's quiet reply
        is the decisive continuation.  Only those already-admitted siblings
        receive this two-candidate, equal-window review.
        """
        if completed_depth < 5 or not self._root_sibling_prevention:
            return None

        root_scores = dict(result.ranked_moves)
        legal_candidates = set(candidates)
        siblings = [
            move
            for move in self._root_sibling_prevention
            if (
                move in legal_candidates
                and move in root_scores
                and board.is_empty(*move)
            )
        ]
        if not siblings or result.move in siblings:
            return None

        comparison = root_candidates.merge_unique(
            (result.move,),
            siblings,
        )
        if len(comparison) < 2:
            return None

        budget = self._quiet_sibling_budget_seconds()
        if budget <= 0:
            return None
        pvs_gap = max(root_scores.values()) - min(
            root_scores[move] for move in comparison
        )
        return self._run_root_safety_probe(
            board,
            comparison,
            trigger="quiet_frontier_sibling",
            pvs_gap=pvs_gap,
            main_rank_stable=True,
            completed_depth=completed_depth,
            budget_seconds=budget,
            quiet_frontier_extension=True,
            target_depth_override=6,
            minimum_stable_depth=5,
            stable_leader_count=3,
            start_depth=3,
        )

    def _maybe_run_dynamic_root_review(
        self,
        board: Board,
        result: RootResult,
        candidates: list[Move],
        *,
        completed_depth: int,
    ) -> RootSafetyProbeResult | None:
        """Spend remaining time on pairwise, source-aware root reviews."""
        if (
            not self.config.root_dynamic_review_enabled
            or completed_depth < 3
        ):
            return None

        budget = self._dynamic_review_budget_seconds()
        if budget <= 0:
            return None
        deadline = time.perf_counter() + budget
        root_scores = dict(result.ranked_moves)
        legal = {
            move
            for move in candidates
            if move in root_scores and board.is_empty(*move)
        }
        critical_groups = self._critical_root_review_groups(
            tuple(
                move
                for move, _score in result.ranked_moves
                if move in legal
            )
        )
        critical_moves = root_candidates.merge_unique(*critical_groups)
        quiet_moves = root_candidates.merge_unique(
            self._root_quiet_prevention,
            self._root_sibling_prevention,
        )
        lowest_risk = root_review.lowest_unknown_risk_move(
            self._proof_candidates,
            legal,
        )
        offensive_moves = root_candidates.merge_unique(
            self._root_offensive_continuations,
            self._root_dual_frontier_bridges,
            self._root_quiet_attack_frontiers,
        )
        active_moves = tuple(
            move
            for move in root_candidates.merge_unique(
                self._root_attack_priority,
                tuple(move for move, _score in result.ranked_moves),
            )
            if (
                move in legal
                and root_candidates.CandidateSource.ACTIVE_COUNTERATTACK
                in self._root_candidate_sources.get(move, ())
            )
        )
        pool = root_review.review_pool(
            self.config,
            result,
            self._proof_candidates,
            critical_moves=critical_moves,
            active_moves=active_moves,
            quiet_moves=quiet_moves,
            offensive_moves=offensive_moves,
        )
        if len(pool) < 2:
            return None

        structure_scores: dict[Move, int] = {}
        for move in pool:
            if time.perf_counter() >= deadline - 0.5:
                break
            try:
                structure_scores[move] = self._frontier_balance_after_move(
                    board,
                    move,
                )
            except SearchTimeout:
                break

        preferred_groups = [
            active_moves,
            (() if lowest_risk is None else (lowest_risk,)),
            self._root_dual_frontier_bridges,
            self._root_offensive_continuations,
            self._root_quiet_attack_frontiers,
            quiet_moves,
            self._root_attack_priority,
        ]
        finalists = root_review.finalists(
            self.config,
            result,
            pool,
            structure_scores,
            critical_groups=critical_groups,
            preferred_groups=preferred_groups,
        )
        pending = [move for move in finalists if move != result.move]
        if not pending:
            return None

        current = result.move
        latest: RootSafetyProbeResult | None = None
        for index, challenger in enumerate(pending):
            if challenger == current:
                continue
            remaining = deadline - time.perf_counter()
            checks_left = max(1, len(pending) - index)
            fair_budget = remaining / checks_left
            pair_budget = (
                min(
                    remaining,
                    max(fair_budget, remaining * 0.65),
                )
                if index == 0 and checks_left > 1
                else fair_budget
            )
            if pair_budget < 0.5:
                break
            pair_result = root_policy.promote_root_move(
                result,
                current,
                score=root_scores[current],
            )
            latest = self._run_dynamic_pair_review(
                board,
                pair_result,
                challenger,
                completed_depth=completed_depth,
                budget_seconds=pair_budget,
            )
            if latest is None:
                continue
            previous = current
            current = self._apply_root_safety_probe(
                pair_result,
                latest,
            ).move
            if current != previous:
                # The highest-priority completed comparison has already
                # changed the leader.  Do not let a later, smaller time slice
                # reverse it with weaker source-only evidence.
                break

        if latest is None:
            return None
        return replace(latest, approved_move=current)

    def _critical_root_review_groups(
        self,
        available: tuple[Move, ...],
    ) -> tuple[tuple[Move, ...], ...]:
        """Return source-priority groups that must reach bounded review."""
        ordered = tuple(dict.fromkeys(available))
        available_set = set(ordered)
        pressure = tuple(
            move
            for move in self._root_pressure_prevention
            if move in available_set
        )
        frontier = tuple(
            move
            for move in ordered
            if (
                move not in pressure
                and root_candidates.CandidateSource.THREAT_FRONTIER
                in self._root_candidate_sources.get(move, ())
            )
        )
        return tuple(group for group in (pressure, frontier) if group)

    def _best_structural_challenger(
        self,
        board: Board,
        moves: list[Move],
        *,
        legal: set[Move],
        excluded: set[Move],
        deadline: float,
    ) -> Move | None:
        scored: list[tuple[int, int, Move]] = []
        for index, move in enumerate(moves):
            if move not in legal or move in excluded:
                continue
            if time.perf_counter() >= deadline - 0.5:
                break
            try:
                score = self._frontier_balance_after_move(board, move)
            except SearchTimeout:
                break
            scored.append((score, -index, move))
        return max(scored)[-1] if scored else None

    def _run_dynamic_pair_review(
        self,
        board: Board,
        result: RootResult,
        challenger: Move,
        *,
        completed_depth: int,
        budget_seconds: float,
    ) -> RootSafetyProbeResult | None:
        pair = [result.move, challenger]
        structure_scores = {
            move: self._frontier_balance_after_move(board, move)
            for move in pair
        }
        structure_keys = {
            move: self._frontier_shape_after_move(board, move)
            for move in pair
        }
        unknown_moves = {
            candidate.move
            for candidate in self._proof_candidates
            if candidate.state == ProofState.UNKNOWN.value
        }
        root_scores = dict(result.ranked_moves)
        mandatory_defense_pair = all(
            root_candidates.CandidateSource.MANDATORY_DEFENSE
            in self._root_candidate_sources.get(move, ())
            for move in pair
        )
        probe = self._run_root_safety_probe(
            board,
            pair,
            trigger="dynamic_remaining_review",
            pvs_gap=abs(root_scores[result.move] - root_scores[challenger]),
            main_rank_stable=True,
            completed_depth=completed_depth,
            budget_seconds=budget_seconds,
            quiet_frontier_extension=not mandatory_defense_pair,
            target_depth_override=7,
            minimum_stable_depth=(
                self.config.root_dynamic_review_min_completed_depth
            ),
            stable_leader_count=(2 if mandatory_defense_pair else 3),
            start_depth=(1 if mandatory_defense_pair else 2),
            extension_depth_override=(
                self.config.threat_extension_depth
                if mandatory_defense_pair
                else None
            ),
            branch_candidate_limit_override=(
                max(self.config.branch_candidate_limit, 12)
                if mandatory_defense_pair
                else None
            ),
            recalibrate_mate_like=not mandatory_defense_pair,
            credible_score_margin=(
                self.config.root_safety_score_margin
                if mandatory_defense_pair
                else None
            ),
        )
        if probe is None:
            return None
        annotated = tuple(
            replace(
                candidate,
                frontier_balance=structure_scores[candidate.move],
            )
            for candidate in probe.candidates
        )
        probe = replace(probe, candidates=annotated)
        approved, basis = root_review.approve_move(
            self.config,
            result,
            probe,
            structure_scores,
            structure_keys=structure_keys,
            unknown_moves=unknown_moves,
            mandatory_defense_consensus=mandatory_defense_pair,
        )
        return replace(
            probe,
            approved_move=approved,
            selection_basis=basis,
        )

    def _dynamic_review_budget_seconds(self) -> float:
        total = self.config.time_limit_seconds
        remaining = self._time.remaining_seconds
        if total is None or remaining is None:
            return 0.0
        available = max(
            0.0,
            remaining - self._final_proof_reserve_seconds() - 0.5,
        )
        budget = min(
            self.config.root_dynamic_review_max_seconds,
            available * self.config.root_dynamic_review_time_fraction,
        )
        return (
            budget
            if budget >= self.config.root_dynamic_review_min_seconds
            else 0.0
        )

    def _quiet_sibling_budget_seconds(self) -> float:
        total = self.config.time_limit_seconds
        remaining = self._time.remaining_seconds
        if total is None or remaining is None:
            return 0.0
        available = max(
            0.0,
            remaining - self._final_proof_reserve_seconds() - 0.05,
        )
        budget = min(
            self.config.root_sibling_probe_max_seconds,
            total * self.config.root_sibling_probe_time_fraction,
            available,
        )
        return (
            budget
            if budget >= self.config.root_sibling_probe_min_seconds
            else 0.0
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
        quiet_frontier_extension: bool = False,
        target_depth_override: int | None = None,
        minimum_stable_depth: int | None = None,
        stable_leader_count: int = 2,
        start_depth: int = 1,
        extension_depth_override: int | None = None,
        branch_candidate_limit_override: int | None = None,
        recalibrate_mate_like: bool = True,
        credible_score_margin: int | None = None,
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
            branch_candidate_limit=(
                self.config.branch_candidate_limit
                if branch_candidate_limit_override is None
                else branch_candidate_limit_override
            ),
            threat_extension_depth=(
                extension_depth_override
                if extension_depth_override is not None
                else (
                    max(
                        6,
                        self.config.threat_extension_depth
                        + self.config.root_safety_extension_bonus,
                    )
                    if quiet_frontier_extension
                    else (
                        self.config.threat_extension_depth
                        + self.config.root_safety_extension_bonus
                    )
                )
            ),
            evaluation_config=self.evaluation_config,
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
        probe._quiet_frontier_extension_enabled = (
            quiet_frontier_extension
        )

        target_depth = (
            min(self.config.max_depth, target_depth_override)
            if target_depth_override is not None
            else min(
                self.config.max_depth,
                max(
                    self.config.root_safety_min_completed_depth,
                    completed_depth + 1,
                ),
            )
        )
        stable_depth = (
            self.config.root_safety_min_completed_depth
            if minimum_stable_depth is None
            else minimum_stable_depth
        )
        latest: tuple[RootSafetyCandidateAnalysis, ...] = ()
        leader_history: list[Move] = []
        probe_completed_depth = 0
        original_priority = {
            move: len(candidates) - index
            for index, move in enumerate(candidates)
        }

        for depth in range(start_depth, target_depth + 1):
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

            if recalibrate_mate_like and any(
                self._is_mate_like_score(candidate.score)
                for candidate in ranked
            ):
                if quiet_frontier_extension:
                    ranked = [
                        RootSafetyCandidateAnalysis(
                            move=candidate.move,
                            score=max(
                                -HEURISTIC_SCORE_LIMIT,
                                min(
                                    HEURISTIC_SCORE_LIMIT,
                                    candidate.score,
                                ),
                            ),
                            principal_variation=(
                                candidate.principal_variation
                            ),
                        )
                        for candidate in ranked
                    ]
                else:
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
            if credible_score_margin is None:
                leader_history.append(ranked[0].move)
            else:
                credible_leader = root_review.credible_layer_leader(
                    ranked,
                    score_margin=credible_score_margin,
                )
                if credible_leader is not None:
                    leader_history.append(credible_leader)

            if (
                depth >= stable_depth
                and len(leader_history) >= stable_leader_count
                and len(
                    set(leader_history[-stable_leader_count:])
                ) == 1
                and not root_review.has_horizon_boundary(ranked)
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
            evaluation_config=self.evaluation_config,
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
            evaluation_config=self.evaluation_config,
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
                    tt_score = self._score_from_tt(entry.score, ply)
                    if entry.bound == BoundType.EXACT:
                        return tt_score, entry.principal_variation
                    if entry.bound == BoundType.LOWER:
                        alpha = max(alpha, tt_score)
                    elif entry.bound == BoundType.UPPER:
                        beta = min(beta, tt_score)
                    if alpha >= beta:
                        self._counters.transposition_cutoffs += 1
                        return tt_score, entry.principal_variation

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
                ply=ply,
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
            return self._static_score(
                board,
                perspective=player,
                side_to_move=player,
            ), ()

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
            ply=ply,
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
            return self._static_score(
                board,
                perspective=player,
                side_to_move=player,
            ), ()

        if len(opponent_wins) == 1:
            forcing_moves = opponent_wins
        else:
            forcing_moves = self._forcing_attack_candidates(
                board,
                player,
                vcf_only=False,
                limit=4,
            )

        if (
            not forcing_moves
            and self._quiet_frontier_extension_enabled
        ):
            forcing_moves = self._quiet_frontier_extension_moves(
                board,
                player,
            )

        if not forcing_moves:
            return self._static_score(
                board,
                perspective=player,
                side_to_move=player,
            ), ()

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

    def _quiet_frontier_extension_moves(
        self,
        board: Board,
        player: int,
    ) -> list[Move]:
        """Return one bounded quiet fan-out for the sibling-only probe."""
        frontiers = self._proof_analyzer.generate_attack_frontiers(
            board,
            player,
            frontier_limit=4,
            continuation_limit=10,
            scan_all_relevant=False,
            stop_requested=self._time.hard_expired,
        )
        self._check_timeout()
        return [
            frontier.gain_move
            for frontier in frontiers
            if (
                frontier.kind is ThreatKind.QUIET
                and len(frontier.continuations)
                >= self.config.root_quiet_sibling_min_continuations
                and frontier.continuation_ranks
                and max(frontier.continuation_ranks) >= 40
            )
        ][:2]

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
        self._check_timeout()
        profiles = self._proof_analyzer.analyze_profiles(
            board,
            shortlist,
            player,
        )
        self._check_timeout()

        forcing: list[tuple[Move, ThreatProfile, int]] = []
        for move in shortlist:
            if vcf_mode:
                self._check_vcf_timeout()
            else:
                self._check_timeout()
            profile = profiles[move]
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

    def _quiet_frontier_prevention_moves(
        self,
        board: Board,
        *,
        frontiers: tuple[ThreatFrontier, ...] | None = None,
    ) -> list[Move]:
        """Find quiet opponent gain points before the root cap is applied.

        These are candidate-completeness hints, not proof results.  The scan
        covers every relevant legal point and uses batched native profiles
        when NativeCore is available; PVS still decides the final move.
        """
        limit = self.config.root_quiet_prevention_limit
        if (
            limit <= 0
            or self.config.max_depth
            < self.config.root_quiet_prevention_min_depth
        ):
            return []

        if frontiers is None:
            frontiers = (
                self._proof_analyzer.generate_attack_frontiers(
                    board,
                    self.opponent,
                    frontier_limit=max(1, board.empty_count),
                    continuation_limit=max(
                        12,
                        self.config.frontier_reply_limit * 2,
                    ),
                    scan_all_relevant=True,
                    stop_requested=self._time.hard_expired,
                )
            )
        quiet = [
            frontier.gain_move
            for frontier in frontiers
            if (
                frontier.kind is ThreatKind.QUIET
                and frontier.continuation_ranks
                and max(frontier.continuation_ranks)
                >= self.config.root_quiet_prevention_min_rank
            )
        ]
        return quiet[:limit]

    def _offensive_continuation_moves(
        self,
        board: Board,
        *,
        opponent_frontiers: tuple[ThreatFrontier, ...],
    ) -> list[Move]:
        """Keep a few quiet attack/defense bridges before the root cap."""
        limit = self.config.root_offensive_continuation_limit
        if limit <= 0 or not opponent_frontiers:
            return []

        own_frontiers = self._proof_analyzer.generate_attack_frontiers(
            board,
            self.player,
            frontier_limit=max(1, board.empty_count),
            continuation_limit=max(
                12,
                self.config.frontier_reply_limit * 2,
            ),
            scan_all_relevant=True,
            stop_requested=self._time.hard_expired,
        )
        self._root_own_frontiers = tuple(own_frontiers)
        self._root_attack_priority = tuple(
            frontier.gain_move
            for frontier in own_frontiers
            if frontier.kind is not ThreatKind.QUIET
        )
        bridges = root_candidates.offensive_continuation_bridges(
            own_frontiers=own_frontiers,
            opponent_frontiers=opponent_frontiers,
            strong_rank=self.config.root_quiet_prevention_min_rank,
            minimum_continuations=(
                self.config
                .root_offensive_continuation_min_continuations
            ),
        )[: max(limit * 3, limit)]
        ranked = [
            (
                self._frontier_balance_after_move(board, move),
                -index,
                move,
            )
            for index, move in enumerate(bridges)
        ]
        ranked.sort(reverse=True)
        return [move for _score, _order, move in ranked[:limit]]

    def _frontier_balance_after_move(
        self,
        board: Board,
        move: Move,
    ) -> int:
        self._frontier_metrics_after_move(board, move)
        return self._root_frontier_balance[move]

    def _frontier_shape_after_move(
        self,
        board: Board,
        move: Move,
    ) -> tuple[int, int, int, int]:
        self._frontier_metrics_after_move(board, move)
        return self._root_frontier_shape[move]

    def _frontier_metrics_after_move(
        self,
        board: Board,
        move: Move,
    ) -> None:
        if (
            move in self._root_frontier_balance
            and move in self._root_frontier_shape
        ):
            return

        board.place(*move, self.player)
        try:
            frontiers = []
            for player in (self.player, self.opponent):
                frontiers.append(
                    self._proof_analyzer.generate_attack_frontiers(
                        board,
                        player,
                        frontier_limit=max(1, board.empty_count),
                        continuation_limit=max(
                            12,
                            self.config.frontier_reply_limit * 2,
                        ),
                        scan_all_relevant=True,
                        stop_requested=self._time.hard_expired,
                    )
                )
        finally:
            board.undo()
        score = root_review.frontier_balance_score(
            frontiers[0],
            frontiers[1],
        )
        self._root_frontier_balance[move] = score
        self._root_frontier_shape[move] = root_review.frontier_shape_key(
            frontiers[0],
            frontiers[1],
        )

    def _root_profile_pool(
        self,
        board: Board,
        legal_moves: list[Move],
    ) -> list[Move]:
        """Return the bounded pool used for normal root classification."""
        return self._root_relevant_pool(
            board,
            legal_moves,
        )[: max(self.config.root_candidate_limit * 2, 20)]

    def _root_relevant_pool(
        self,
        board: Board,
        legal_moves: list[Move],
    ) -> list[Move]:
        """Return all relevant points for completeness-only root scans."""
        raw = self._raw_candidates(
            board,
            legal_moves,
            at_root=True,
        )
        return sorted(
            raw,
            key=lambda move: self._quick_order_score(
                board,
                move,
                self.player,
            ),
            reverse=True,
        )

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

    def _has_unverified_root_advantage(
        self,
        result: RootResult,
    ) -> bool:
        """Return whether selective PVS reached an unproved tactical band."""
        return len(result.ranked_moves) > 1 and any(
            score >= self.config.root_unverified_advantage_threshold
            for _move, score in result.ranked_moves
        )

    def _expand_unverified_advantage_root_candidates(
        self,
        board: Board,
        candidates: list[Move],
    ) -> list[Move]:
        """Add bounded strategic coverage after a suspicious high score.

        The normal 12-move root remains unchanged on ordinary positions.  If
        selective search reports a mate-like or four-chain advantage without
        proof, a second-stage root may grow to at most twice that width.  Its
        extra moves come from a full relevant-board static ranking rather than
        one recorded coordinate, so quiet survival moves and rotated shapes
        receive the same treatment.
        """
        expanded_limit = max(
            self.config.root_candidate_limit * 2,
            len(candidates) + 1,
        )
        raw = self._raw_candidates(
            board,
            board.get_legal_moves(),
            at_root=True,
        )
        strategic = sorted(
            raw,
            key=lambda move: (
                self._heuristic_root_score(board, move),
                self._quick_order_score(board, move, self.player),
            ),
            reverse=True,
        )[: self.config.root_survival_scan_limit]
        ordinary = self._ordered_moves(
            board,
            self.player,
            at_root=True,
            ply=0,
            limit=expanded_limit,
            tt_move=self._tt_best_move(board, self.player),
        )
        merged = list(
            dict.fromkeys((*candidates, *strategic, *ordinary))
        )
        return self._filter_proven_losing_candidates(
            merged
        )[:expanded_limit]

    def _reserve_expansion_review_time(self) -> None:
        """Tighten PVS so an expanded root cannot consume its own audit."""
        hard_deadline = self._time.hard_deadline
        if hard_deadline is None:
            return
        reserve = root_policy.serial_verification_reserve(
            final_proof_seconds=self._final_proof_reserve_seconds(),
            root_review_seconds=self._root_safety_budget_seconds(),
        )
        review_deadline = hard_deadline - reserve
        if self._search_phase_deadline is None:
            self._search_phase_deadline = review_deadline
        else:
            self._search_phase_deadline = min(
                self._search_phase_deadline,
                review_deadline,
            )
        self._check_timeout()

    def _is_unverified_expansion_move(self, move: Move) -> bool:
        return (
            root_candidates.CandidateSource.ROOT_EXPANSION
            in self._root_candidate_sources.get(move, ())
        )

    def _expansion_leader_has_base_review(self, move: Move) -> bool:
        probe = self._root_safety_probe
        if probe is None:
            return False
        reviewed = {candidate.move for candidate in probe.candidates}
        return (
            move in reviewed
            and any(
                not self._is_unverified_expansion_move(candidate)
                for candidate in reviewed
            )
        )

    def _hold_unverified_expansion_leader(
        self,
        result: RootResult,
    ) -> RootResult:
        """Keep an unreviewed expansion leader as a challenger, not policy."""
        base = next(
            (
                move
                for move, _score in result.ranked_moves
                if not self._is_unverified_expansion_move(move)
            ),
            None,
        )
        if base is None:
            return result
        scores = dict(result.ranked_moves)
        return root_policy.promote_root_move(
            result,
            base,
            score=scores[base],
        )

    def _profile_moves_timed(
        self,
        board: Board,
        candidates: list[Move],
        player: int,
    ) -> dict[Move, ThreatProfile]:
        self._check_timeout()
        profiles = self._proof_analyzer.analyze_profiles(
            board,
            candidates,
            player,
        )
        self._check_timeout()
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
        use_search_heuristics: bool = True,
    ) -> list[Move]:
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return []

        own_wins = self._timed_winning_moves(board, player, legal_moves)
        if own_wins:
            return (
                self._promote_move(own_wins, tt_move)
                if use_search_heuristics
                else own_wins
            )

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
                use_search_heuristics=use_search_heuristics,
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
            key=(
                lambda move: self._quick_order_score(
                    board,
                    move,
                    player,
                )
            ),
            reverse=True,
        )[:preselection_limit]

        ranked = self._order_specific_moves(
            board,
            raw_candidates,
            player,
            ply=ply,
            tt_move=tt_move,
            full_evaluation=(at_root or ply <= 1),
            # Search heuristics may order a fixed selective set, but must not
            # decide membership: TT values assume a stable tree for each key.
            use_search_heuristics=False,
        )
        selected = ranked[:desired_limit]
        # TT/history/killer state may reorder the bounded set, but it must
        # never decide which structurally ranked moves are members.
        return self._order_specific_moves(
            board,
            selected,
            player,
            ply=ply,
            tt_move=tt_move,
            full_evaluation=True,
            use_search_heuristics=use_search_heuristics,
        )

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
        use_search_heuristics: bool = True,
    ) -> list[Move]:
        center = (board.size - 1) / 2
        scored: list[tuple[Move, int, int, int, int]] = []
        opponent = other_side(player)

        for move in dict.fromkeys(moves):
            self._check_timeout()
            killer_priority = (
                self._killer_priority(move, ply)
                if use_search_heuristics
                else 0
            )
            history_score = (
                self._history_score(player, move)
                if use_search_heuristics
                else 0
            )
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
                use_search_heuristics and item[0] == tt_move,
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
        cache_key = (board.zobrist_hash, row, column, player)
        cached = self._quick_order_cache.get(cache_key)
        if cached is not None:
            return cached
        opponent = other_side(player)
        size = board.size
        grid = board.grid
        score = 0
        distance_weights = (0, 24, 8, 3, 1)

        for row_step, column_step in DIRECTIONS:
            for sign in (-1, 1):
                for distance in range(1, 5):
                    neighbor_row = row + sign * distance * row_step
                    neighbor_column = (
                        column + sign * distance * column_step
                    )
                    if not (
                        0 <= neighbor_row < size
                        and 0 <= neighbor_column < size
                    ):
                        break
                    cell = grid[neighbor_row][neighbor_column]
                    weight = distance_weights[distance]
                    if cell == player:
                        score += weight * 3
                    elif cell == opponent:
                        score += weight * 4
                    elif cell == EMPTY:
                        score += weight

        # This integer form is exactly floor(distance squared) from the former
        # half-integer center calculation, without float creation or powers.
        doubled_row_distance = 2 * row - (size - 1)
        doubled_column_distance = 2 * column - (size - 1)
        score -= (
            doubled_row_distance * doubled_row_distance
            + doubled_column_distance * doubled_column_distance
        ) // 4
        self._quick_order_cache[cache_key] = score
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

    @property
    def evaluation_config(self) -> EvaluationConfig:
        """Return the immutable evaluation profile fixed for this AI."""
        return self._evaluation_config

    def _static_score(
        self,
        board: Board,
        perspective: int,
        side_to_move: int | None = None,
    ) -> int:
        """Evaluate one leaf with score sign and move tempo kept separate."""
        if side_to_move is None:
            side_to_move = perspective
        score = evaluate_search_position(
            board,
            perspective=perspective,
            side_to_move=side_to_move,
            config=self.evaluation_config,
        )
        return max(
            -HEURISTIC_SCORE_LIMIT,
            min(HEURISTIC_SCORE_LIMIT, score),
        )

    def _position_key(self, board: Board, player: int) -> int:
        position_key = (
            board.zobrist_hash
            ^ get_zobrist_table(board.size).side_key(player)
        )
        # Non-root move generation deliberately gives a wider radius to the
        # most recent moves.  The same colored stones can therefore produce a
        # different selective tree when reached in another order.  Include
        # that bounded order history in the TT key instead of reusing a score
        # computed from a different candidate set.
        history_key = 1_469_598_103_934_665_603
        recent = board.move_history[-self.config.recent_move_count :]
        for ordinal, (row, column, stone) in enumerate(recent, start=1):
            token = (
                ((row * board.size + column + 1) << 2)
                ^ stone
                ^ (ordinal << 12)
            )
            history_key ^= token
            history_key = (
                history_key * 1_099_511_628_211
            ) & MASK_64
        return (position_key ^ history_key) & MASK_64

    @staticmethod
    def _score_to_tt(score: int, ply: int) -> int:
        """Store mate-band scores independently of the current root ply."""
        if score > HEURISTIC_SCORE_LIMIT:
            return score + ply
        if score < -HEURISTIC_SCORE_LIMIT:
            return score - ply
        return score

    @staticmethod
    def _score_from_tt(score: int, ply: int) -> int:
        """Restore a TT mate-band score for the current root ply."""
        if score > HEURISTIC_SCORE_LIMIT:
            return score - ply
        if score < -HEURISTIC_SCORE_LIMIT:
            return score + ply
        return score

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
        *,
        ply: int,
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
        if previous is not None:
            if previous.depth > depth:
                return
            if (
                previous.depth == depth
                and previous.extension_depth > extension_depth
            ):
                return
            if (
                previous.depth == depth
                and previous.extension_depth == extension_depth
                and previous.bound is BoundType.EXACT
                and bound is not BoundType.EXACT
            ):
                return

        self._transposition_table[key] = TTEntry(
            depth=depth,
            extension_depth=extension_depth,
            score=self._score_to_tt(score, ply),
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
        # NativeCore already exposes an exact, order-preserving batch kernel
        # for this scan.  Search used to call the scalar Python predicate once
        # per legal move, although leaf extension performs the same scan twice
        # at nearly every node.  Check the deadline on both sides of the small
        # bounded native call; the Python fallback remains bit-for-bit the same
        # through evaluator.find_winning_moves().
        self._check_timeout()
        winning = find_winning_moves(board, player, candidates)
        self._check_timeout()
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
        self.last_analysis = build_search_analysis(
            self,
            selected_move=selected_move,
            reason=reason,
            candidate_count=candidate_count,
            ranked_moves=ranked_moves,
            completed_depth=completed_depth,
            principal_variation=principal_variation,
            search_completed=search_completed,
            own_profiles=own_profiles,
            opponent_profiles=opponent_profiles,
            vcf_found=vcf_found,
            vcf_depth=vcf_depth,
            stop_reason=stop_reason,
        )
