"""Build immutable search diagnostics from one completed decision.

The hot search path owns mutable state; this module owns the translation of
that state into the public :class:`DecisionAnalysis` record.  Keeping the
translation here prevents serialization growth from bloating PVS/Negamax.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from engine import root_candidates, root_safety
from engine.ai import (
    CandidateAnalysis,
    DecisionAnalysis,
    FinalProofEmergencyVCFProvenance,
    Move,
    ProofCandidateAnalysis,
    RootReviewPairAnalysis,
    RootReviewUnpairedFinalistAnalysis,
    SearchPhaseTiming,
)
from engine.evaluator import EvaluationConfig, ThreatProfile
from engine.proof_search import (
    ProofResult,
    ProofState,
    ProofTable,
    ProofTableStats,
)
from engine.search_types import (
    DefenseProbeResult,
    RootSafetyProbeResult,
    RootVCFScanResult,
    SearchConfig,
    SearchCounters,
)
from engine.threats import ThreatAnalyzer, ThreatAnalyzerStats
from engine.time_manager import TimeManager


class SearchDiagnosticsSource(Protocol):
    """Explicit read-only boundary consumed by the diagnostics builder."""

    config: SearchConfig
    diagnostics: bool
    top_n: int
    evaluation_config: EvaluationConfig
    _time: TimeManager
    _counters: SearchCounters
    _interrupted_depth: int
    _transposition_table: Mapping[int, object]
    _defense_probe: DefenseProbeResult | None
    _proof_root_result: ProofResult | None
    _proof_candidates: tuple[ProofCandidateAnalysis, ...]
    _proof_table: ProofTable
    _proof_table_start_stats: ProofTableStats
    _proof_analyzer: ThreatAnalyzer | None
    _root_candidate_sources: dict[
        Move,
        frozenset[root_candidates.CandidateSource],
    ]
    _final_proof_checked: bool
    _final_proof_state: str
    _final_proof_completed: bool
    _final_proof_selected: Move | None
    _final_proof_rejected: tuple[Move, ...]
    _final_proof_selection_basis: str
    _final_proof_overrode_review: bool
    _final_proof_emergency_vcf: FinalProofEmergencyVCFProvenance | None
    _root_safety_probe: RootSafetyProbeResult | None
    _root_safety_applied: bool
    _root_review_incoming_move: Move | None
    _root_review_approved_move: Move | None
    _root_review_result_changed: bool
    _root_review_apply_reason: str | None
    _root_review_confirmed_move: Move | None
    _root_review_confirmed_depth: int
    _root_review_confirmed_basis: str | None
    _root_review_confirmed_rank_stable: bool
    _root_review_confirmed_boundary: bool
    _root_vcf_scan: RootVCFScanResult | None
    _root_mate_scores_quarantined: bool
    _root_review_finalists: tuple[Move, ...]
    _root_review_trace: list[tuple[str, RootSafetyProbeResult]]
    _root_review_unpaired_finalists: tuple[
        RootReviewUnpairedFinalistAnalysis, ...
    ]
    _phase_timings: dict[str, float]


def compose_search_reason(
    reason: str,
    *,
    expansion_reason: str | None,
    expansion_hold_applied: bool,
    root_vcf_scan: RootVCFScanResult | None,
    mate_scores_quarantined: bool,
    defense_risk_override: bool,
    root_safety_probe: RootSafetyProbeResult | None,
    root_safety_applied: bool,
    final_proof_checked: bool,
    final_proof_state: str,
    final_proof_completed: bool,
    final_proof_rejected: tuple[Move, ...],
) -> str:
    """Add post-search verification details to one user-facing reason."""
    expansion_messages = {
        "near_forced_loss": "；近必败候选已自动扩展",
        "unverified_advantage": "；未证实高分已触发全盘生存候选扩展",
    }
    reason += expansion_messages.get(expansion_reason, "")
    if expansion_hold_applied:
        reason += "；扩展领跑未完成等窗复核，已按原候选与 Proof 保守仲裁"

    if root_vcf_scan is not None:
        if any(
            move not in root_vcf_scan.original_candidates
            for move in root_vcf_scan.candidates
        ):
            reason += "；已合并对手 VCF 拦截点"
        if any(
            candidate.status
            == root_safety.RootCandidateSafety.PROVEN_LOSS.value
            for candidate in root_vcf_scan.analyses
        ):
            reason += "；已淘汰对手 VCF 可证败着"
        if not root_vcf_scan.complete:
            reason += "；对手 VCF 生存检查部分候选未知"

    if mate_scores_quarantined:
        reason += "；未证明 Mate 分已降级为启发式分"
    if defense_risk_override:
        reason += "；防守探针已由更低的对手威胁风险纠正"

    if root_safety_probe is not None:
        if root_safety_probe.trigger == "dynamic_remaining_review":
            dynamic_messages = {
                "frontier_balance": "；动态余时复核按攻防前沿净增益改选",
                "equal_window": "；动态余时同窗深层复核完成",
                "pvs_fallback": "；动态余时复核未形成稳定改选证据",
                "boundary_tie_pvs_fallback": (
                    "；边界平局升级复核仍未完成，"
                    "显式回退 PVS 候选"
                ),
                "boundary_secondary_equal_window": (
                    "；边界夹值由无安静延伸次级同窗证据裁决"
                ),
            }
            reason += dynamic_messages.get(
                root_safety_probe.selection_basis,
                "；动态余时复核完成",
            )
        else:
            safety_messages = {
                ("threat_risk_override", True): (
                    "；独立复核批准风险指标改选"
                ),
                ("threat_risk_override", False): (
                    "；风险改选未获确认，保留 PVS 首选"
                ),
                ("quiet_frontier_sibling", True): (
                    "；安静前沿深层复核改选候选"
                ),
                ("quiet_frontier_sibling", False): (
                    "；安静前沿深层复核保持原候选"
                ),
            }
            reason += safety_messages.get(
                (root_safety_probe.trigger, root_safety_applied),
                (
                    "；决胜节点复核改选近分候选"
                    if root_safety_applied
                    else "；决胜节点复核保持原候选"
                ),
            )

    if final_proof_rejected:
        reason += "；最终 Proof 已淘汰可证败着"
    if (
        final_proof_completed
        and final_proof_state == ProofState.PROVEN_LOSS.value
    ):
        return reason + "；最终候选通过 Proof 生存确认"
    if final_proof_checked and final_proof_state == ProofState.UNKNOWN.value:
        return reason + "；最终 Proof 为 UNKNOWN，安全性未确认"
    if final_proof_checked:
        return reason + "；最终候选未取得严格 Proof 结论"
    return reason


def review_arbitration_state(
    probe: RootSafetyProbeResult | None,
    config: SearchConfig,
) -> str:
    """Describe whether the recorded root evidence reached a decision."""
    if probe is None:
        return "not_checked"
    if probe.selection_basis == "boundary_tie_pvs_fallback":
        return "boundary_tie_unresolved"
    minimum_depth = (
        config.root_dynamic_review_min_completed_depth
        if probe.trigger == "dynamic_remaining_review"
        else config.root_safety_min_completed_depth
    )
    if (
        probe.selection_basis == "pvs_fallback"
        and (
            probe.completed_depth < minimum_depth
            or not probe.rank_stable
        )
    ):
        return "insufficient_depth"
    return "completed"


def build_search_analysis(
    source: SearchDiagnosticsSource,
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
) -> DecisionAnalysis:
    """Create the public immutable analysis without mutating search state."""
    elapsed = source._time.elapsed_seconds
    nps = int(source._counters.nodes / elapsed) if elapsed > 0 else 0
    top_candidates: list[CandidateAnalysis] = []

    if source.diagnostics:
        own_profiles = own_profiles or {}
        opponent_profiles = opponent_profiles or {}
        for move, score in ranked_moves[: source.top_n]:
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
        if source.config.time_limit_seconds is None
        else source.config.time_limit_seconds * source.config.soft_time_ratio
    )
    time_used_ratio = (
        None
        if source.config.time_limit_seconds is None
        else min(1.0, elapsed / source.config.time_limit_seconds)
    )
    proof_tt_delta = source._proof_table.stats().delta(
        source._proof_table_start_stats
    )
    threat_stats = (
        ThreatAnalyzerStats()
        if source._proof_analyzer is None
        else source._proof_analyzer.stats()
    )
    defense_probe = source._defense_probe
    proof_root = source._proof_root_result
    safety_probe = source._root_safety_probe
    root_vcf = source._root_vcf_scan
    counters = source._counters
    review_pairs: tuple[RootReviewPairAnalysis, ...] = ()
    review_finalists: tuple[Move, ...] = ()
    review_source_coverage: tuple[
        tuple[str, tuple[Move, ...]], ...
    ] = ()
    review_unpaired_finalists: tuple[
        RootReviewUnpairedFinalistAnalysis, ...
    ] = ()
    if source.diagnostics and source._root_review_trace:
        review_pairs = tuple(
            RootReviewPairAnalysis(
                channel=channel,
                trigger=probe.trigger,
                completed_depth=probe.completed_depth,
                nodes=probe.nodes,
                candidates=probe.candidates,
                leader_history=probe.leader_history,
                approved_move=probe.approved_move,
                selection_basis=probe.selection_basis,
                requested_budget_seconds=(
                    probe.requested_budget_seconds
                ),
                boundary_tie_detected=probe.boundary_tie_detected,
                mate_like_hit_depths=probe.mate_like_hit_depths,
                final_dimension_recovered=(
                    probe.final_dimension_recovered
                ),
            )
            for channel, probe in source._root_review_trace
        )
        review_finalists = source._root_review_finalists
        reviewed_moves = tuple(
            dict.fromkeys(
                candidate.move
                for _channel, probe in source._root_review_trace
                for candidate in probe.candidates
            )
        )
        review_source_coverage = tuple(
            (
                candidate_source.value,
                tuple(
                    move
                    for move in reviewed_moves
                    if candidate_source
                    in source._root_candidate_sources.get(move, ())
                ),
            )
            for candidate_source in root_candidates.CandidateSource
            if any(
                candidate_source
                in source._root_candidate_sources.get(move, ())
                for move in reviewed_moves
            )
        )
    if source.diagnostics or source._root_review_unpaired_finalists:
        review_unpaired_finalists = (
            source._root_review_unpaired_finalists
        )

    return DecisionAnalysis(
        selected_move=selected_move,
        reason=reason,
        candidate_count=candidate_count,
        evaluation_profile=source.evaluation_config.profile_name,
        evaluation_parameters=source.evaluation_config.parameter_items(),
        top_candidates=tuple(top_candidates),
        root_candidate_sources=tuple(
            (
                move,
                tuple(
                    sorted(
                        item.value
                        for item in source._root_candidate_sources[move]
                    )
                ),
            )
            for move in dict.fromkeys(
                (selected_move, *(item[0] for item in ranked_moves))
            )
            if move in source._root_candidate_sources
        ),
        search_depth=completed_depth,
        requested_depth=source.config.max_depth,
        interrupted_depth=source._interrupted_depth,
        nodes=counters.nodes,
        nps=nps,
        cutoffs=counters.cutoffs,
        transposition_hits=counters.transposition_hits,
        transposition_cutoffs=counters.transposition_cutoffs,
        transposition_size=len(source._transposition_table),
        killer_hits=counters.killer_hits,
        history_hits=counters.history_hits,
        extensions=counters.extensions,
        pvs_researches=counters.pvs_researches,
        aspiration_researches=counters.aspiration_researches,
        vcf_found=vcf_found,
        vcf_depth=vcf_depth,
        vcf_nodes=counters.vcf_nodes,
        elapsed_seconds=elapsed,
        soft_time_limit_seconds=soft_limit,
        hard_time_limit_seconds=source.config.time_limit_seconds,
        principal_variation=principal_variation,
        search_completed=search_completed,
        stop_reason=stop_reason,
        time_used_ratio=time_used_ratio,
        defense_vct_checked=defense_probe is not None,
        defense_vct_depth=(
            0 if defense_probe is None else defense_probe.completed_depth
        ),
        defense_vct_nodes=(0 if defense_probe is None else defense_probe.nodes),
        defense_vct_best_move=(
            None if defense_probe is None else defense_probe.best_move
        ),
        defense_vct_candidates=(
            () if defense_probe is None else defense_probe.candidates
        ),
        proof_checked=(proof_root is not None or bool(source._proof_candidates)),
        proof_state=(
            "not_checked" if proof_root is None else proof_root.state.value
        ),
        proof_nodes=counters.proof_nodes,
        proof_elapsed_seconds=(
            (0.0 if proof_root is None else proof_root.elapsed_seconds)
            + sum(
                candidate.elapsed_seconds
                for candidate in source._proof_candidates
            )
        ),
        proof_best_move=(None if proof_root is None else proof_root.best_move),
        proof_principal_variation=(
            () if proof_root is None else proof_root.principal_variation
        ),
        proof_cutoff_reason=(
            None if proof_root is None else proof_root.cutoff_reason
        ),
        proof_candidates=source._proof_candidates,
        proof_tt_queries=proof_tt_delta.queries,
        proof_tt_hits=proof_tt_delta.hits,
        proof_tt_compatible_hits=proof_tt_delta.compatible_hits,
        proof_tt_stores=proof_tt_delta.stores,
        proof_tt_skipped_stores=proof_tt_delta.skipped_stores,
        proof_tt_evictions=proof_tt_delta.evictions,
        proof_tt_size=proof_tt_delta.size,
        proof_hint_queries=proof_tt_delta.hint_queries,
        proof_hint_hits=proof_tt_delta.hint_hits,
        proof_hint_stores=proof_tt_delta.hint_stores,
        proof_hint_evictions=proof_tt_delta.hint_evictions,
        proof_hint_size=proof_tt_delta.hint_size,
        final_proof_checked=source._final_proof_checked,
        final_proof_state=source._final_proof_state,
        final_proof_completed=source._final_proof_completed,
        final_proof_selected_move=source._final_proof_selected,
        final_proof_rejected_moves=source._final_proof_rejected,
        final_proof_selection_basis=source._final_proof_selection_basis,
        final_proof_overrode_review=source._final_proof_overrode_review,
        final_proof_emergency_vcf=source._final_proof_emergency_vcf,
        threat_candidate_batches=threat_stats.candidate_batches,
        threat_exact_descriptions=threat_stats.exact_descriptions,
        threat_frontier_batches=threat_stats.frontier_batches,
        threat_frontier_descriptions=threat_stats.frontier_descriptions,
        threat_cache_queries=threat_stats.cache_queries,
        threat_cache_hits=threat_stats.cache_hits,
        threat_cache_stores=threat_stats.cache_stores,
        threat_cache_skips=threat_stats.cache_skips,
        root_safety_checked=safety_probe is not None,
        root_safety_applied=source._root_safety_applied,
        root_safety_trigger=(
            None if safety_probe is None else safety_probe.trigger
        ),
        root_safety_pvs_gap=(
            None if safety_probe is None else safety_probe.pvs_gap
        ),
        root_safety_main_rank_stable=(
            True if safety_probe is None else safety_probe.main_rank_stable
        ),
        root_safety_depth=(
            0 if safety_probe is None else safety_probe.completed_depth
        ),
        root_safety_nodes=counters.root_safety_nodes,
        root_safety_best_move=(
            None if safety_probe is None else safety_probe.best_move
        ),
        root_safety_leaders=(
            () if safety_probe is None else safety_probe.leader_history
        ),
        root_safety_candidates=(
            () if safety_probe is None else safety_probe.candidates
        ),
        root_review_incoming_move=source._root_review_incoming_move,
        root_review_approved_move=source._root_review_approved_move,
        root_review_result_changed=source._root_review_result_changed,
        root_review_apply_reason=source._root_review_apply_reason,
        root_review_confirmed_move=source._root_review_confirmed_move,
        root_review_confirmed_depth=source._root_review_confirmed_depth,
        root_review_confirmed_basis=source._root_review_confirmed_basis,
        root_review_confirmed_rank_stable=(
            source._root_review_confirmed_rank_stable
        ),
        root_review_confirmed_boundary=(
            source._root_review_confirmed_boundary
        ),
        root_vcf_checked=root_vcf is not None,
        root_vcf_complete=False if root_vcf is None else root_vcf.complete,
        root_vcf_nodes=counters.root_vcf_nodes,
        root_vcf_exhaustive_rescue_scanned=(
            False if root_vcf is None else root_vcf.exhaustive_rescue_scanned
        ),
        root_vcf_rescue_candidates_checked=(
            0 if root_vcf is None else root_vcf.rescue_candidates_checked
        ),
        root_vcf_baseline_line=(
            () if root_vcf is None else root_vcf.baseline_line
        ),
        root_vcf_candidates=(() if root_vcf is None else root_vcf.analyses),
        mate_scores_quarantined=source._root_mate_scores_quarantined,
        phase_timings=tuple(
            SearchPhaseTiming(phase, seconds)
            for phase, seconds in source._phase_timings.items()
        ),
        root_safety_selection_basis=(
            None if safety_probe is None else safety_probe.selection_basis
        ),
        review_arbitration_state=review_arbitration_state(
            safety_probe,
            source.config,
        ),
        review_completed_depth=(
            0 if safety_probe is None else safety_probe.completed_depth
        ),
        review_rank_stable=(
            False if safety_probe is None else safety_probe.rank_stable
        ),
        review_boundary_tie_detected=(
            False
            if safety_probe is None
            else safety_probe.boundary_tie_detected
        ),
        review_budget_seconds=(
            0.0
            if safety_probe is None
            else safety_probe.requested_budget_seconds
        ),
        review_escalation_budget_seconds=(
            0.0
            if safety_probe is None
            else safety_probe.escalation_budget_seconds
        ),
        boundary_secondary_attempted=(
            source._root_boundary_secondary_attempted
        ),
        boundary_secondary_mate_like_hit_depths=(
            source._root_boundary_secondary_mate_like_hit_depths
        ),
        boundary_secondary_final_dimension_recovered=(
            source._root_boundary_secondary_final_dimension_recovered
        ),
        root_review_finalists=review_finalists,
        root_review_pairs=review_pairs,
        root_review_source_coverage=review_source_coverage,
        root_review_unpaired_finalists=review_unpaired_finalists,
    )
