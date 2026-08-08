"""Shared search configuration, result types, and score constants.

This module is intentionally free of search implementation details.  Keeping
the data contract separate lets root policies and future proof engines share
the same states without importing ``SearchAI`` or creating circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from engine.ai import (
    DefenseCandidateAnalysis,
    Move,
    RootVCFCandidateAnalysis,
    RootSafetyCandidateAnalysis,
)

MATE_SCORE = 1_000_000_000
HEURISTIC_SCORE_LIMIT = 100_000_000
INFINITY = MATE_SCORE * 2


class SearchTimeout(RuntimeError):
    """The main search exceeded its hard time budget."""


class VCFTimeout(RuntimeError):
    """The VCF sub-search exhausted its own time slice."""


class BoundType(str, Enum):
    EXACT = "exact"
    LOWER = "lower"
    UPPER = "upper"


class DefenseProof(str, Enum):
    """Bounded defense-probe status; UNKNOWN is never treated as safe."""

    SURVIVES_PROBE = "survives_probe"
    UNKNOWN = "unknown"
    FORCED_LOSS = "forced_loss"


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Search parameters shared by the coordinator and helper modules."""

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
    frontier_counterattack_min_moves: int = 4
    defense_vct_max_candidates: int = 3
    defense_vct_probe_depth: int = 3
    defense_vct_extension_depth: int = 4
    defense_vct_branch_limit: int = 6
    defense_vct_time_fraction: float = 0.12
    defense_vct_max_seconds: float = 2.5
    defense_vct_score_margin: int = 20_000
    mandatory_defense_probe_depth: int = 5
    mandatory_defense_extension_depth: int = 1
    mandatory_defense_branch_limit: int = 10
    mandatory_defense_time_fraction: float = 0.14
    mandatory_defense_max_seconds: float = 8.0
    proof_time_fraction: float = 0.35
    proof_max_seconds: float = 15.0
    proof_root_candidate_limit: int = 4
    proof_max_nodes: int = 150_000
    proof_max_attacker_moves: int = 10
    proof_quiet_frontier_limit: int = 16
    proof_quiet_attacker_moves: int = 1
    proof_frontier_scan_limit: int = 24
    proof_risk_pvs_margin: int = 20_000
    proof_use_threat_cache: bool = True
    proof_final_check_enabled: bool = True
    proof_final_time_fraction: float = 0.14
    proof_final_max_seconds: float = 8.0
    proof_final_min_seconds: float = 0.25
    proof_final_candidate_limit: int = 4
    root_safety_enabled: bool = True
    root_safety_candidate_limit: int = 2
    root_safety_score_margin: int = 20_000
    root_safety_micro_margin: int = 2_000
    root_safety_time_fraction: float = 0.10
    root_safety_max_seconds: float = 6.0
    root_safety_min_seconds: float = 0.75
    root_safety_extension_bonus: int = 2
    root_safety_min_completed_depth: int = 3
    root_sibling_probe_time_fraction: float = 0.20
    root_sibling_probe_max_seconds: float = 12.0
    root_sibling_probe_min_seconds: float = 2.0
    root_dynamic_review_enabled: bool = True
    root_dynamic_review_pvs_limit: int = 3
    root_dynamic_review_candidate_limit: int = 8
    root_dynamic_review_finalist_limit: int = 4
    root_dynamic_review_time_fraction: float = 0.80
    root_dynamic_review_max_seconds: float = 30.0
    root_dynamic_review_min_seconds: float = 2.0
    root_dynamic_review_min_completed_depth: int = 5
    root_dynamic_review_structure_margin: int = 3_000
    root_vcf_safety_enabled: bool = True
    root_vcf_safety_max_attacker_moves: int = 5
    root_vcf_safety_time_fraction: float = 0.08
    root_vcf_safety_max_seconds: float = 3.0
    root_vcf_safety_min_seconds: float = 0.05
    root_vcf_safety_intercept_fraction: float = 0.25
    root_quiet_prevention_limit: int = 3
    root_quiet_prevention_min_rank: int = 80
    root_quiet_sibling_prevention_limit: int = 1
    root_quiet_sibling_min_continuations: int = 3
    root_offensive_continuation_limit: int = 4
    root_offensive_continuation_min_continuations: int = 4
    root_dual_frontier_bridge_limit: int = 1
    root_dual_frontier_min_own_rank: int = 60
    root_dual_frontier_min_opponent_rank: int = 40
    root_dual_frontier_min_own_continuations: int = 4
    root_dual_frontier_min_opponent_continuations: int = 2
    root_quiet_prevention_min_depth: int = 4
    root_frontier_truth_score_margin: int = 30_000
    root_forcing_counterattack_min_depth: int = 4
    root_survival_scan_limit: int = 16
    root_survival_min_depth: int = 4
    root_unverified_advantage_threshold: int = 900_000

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
        if self.frontier_counterattack_min_moves < 2:
            raise ValueError(
                "frontier_counterattack_min_moves 不能小于 2。"
            )
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
        if self.mandatory_defense_probe_depth < 1:
            raise ValueError("mandatory_defense_probe_depth 必须大于 0。")
        if self.mandatory_defense_extension_depth < 0:
            raise ValueError(
                "mandatory_defense_extension_depth 不能小于 0。"
            )
        if self.mandatory_defense_branch_limit < 2:
            raise ValueError(
                "mandatory_defense_branch_limit 不能小于 2。"
            )
        if not 0 < self.mandatory_defense_time_fraction <= 1:
            raise ValueError(
                "mandatory_defense_time_fraction 必须在 0～1 之间。"
            )
        if self.mandatory_defense_max_seconds <= 0:
            raise ValueError(
                "mandatory_defense_max_seconds 必须大于 0。"
            )
        if not 0 < self.proof_time_fraction <= 1:
            raise ValueError("proof_time_fraction 必须在 0～1 之间。")
        if self.proof_max_seconds <= 0:
            raise ValueError("proof_max_seconds 必须大于 0。")
        if self.proof_root_candidate_limit < 1:
            raise ValueError("proof_root_candidate_limit 必须大于 0。")
        if self.proof_max_nodes < 1:
            raise ValueError("proof_max_nodes 必须大于 0。")
        if self.proof_max_attacker_moves < 1:
            raise ValueError("proof_max_attacker_moves 必须大于 0。")
        if self.proof_quiet_frontier_limit < 0:
            raise ValueError("proof_quiet_frontier_limit 不能小于 0。")
        if self.proof_quiet_attacker_moves < 0:
            raise ValueError("proof_quiet_attacker_moves 不能小于 0。")
        if self.proof_frontier_scan_limit < 1:
            raise ValueError("proof_frontier_scan_limit 必须大于 0。")
        if self.proof_risk_pvs_margin < 0:
            raise ValueError("proof_risk_pvs_margin 不能小于 0。")
        if not 0 < self.proof_final_time_fraction < 0.5:
            raise ValueError(
                "proof_final_time_fraction 必须在 0～0.5 之间。"
            )
        if self.proof_final_max_seconds <= 0:
            raise ValueError("proof_final_max_seconds 必须大于 0。")
        if self.proof_final_min_seconds <= 0:
            raise ValueError("proof_final_min_seconds 必须大于 0。")
        if self.proof_final_min_seconds > self.proof_final_max_seconds:
            raise ValueError(
                "proof_final_min_seconds 不能大于 "
                "proof_final_max_seconds。"
            )
        if self.proof_final_candidate_limit < 1:
            raise ValueError("proof_final_candidate_limit 必须大于 0。")
        if self.root_safety_candidate_limit < 2:
            raise ValueError("root_safety_candidate_limit 不能小于 2。")
        if self.root_safety_score_margin < 0:
            raise ValueError("root_safety_score_margin 不能小于 0。")
        if self.root_safety_micro_margin < 0:
            raise ValueError("root_safety_micro_margin 不能小于 0。")
        if (
            self.root_safety_micro_margin
            > self.root_safety_score_margin
        ):
            raise ValueError(
                "root_safety_micro_margin 不能大于 "
                "root_safety_score_margin。"
            )
        if not 0 < self.root_safety_time_fraction < 0.5:
            raise ValueError(
                "root_safety_time_fraction 必须在 0～0.5 之间。"
            )
        if self.root_safety_max_seconds <= 0:
            raise ValueError("root_safety_max_seconds 必须大于 0。")
        if self.root_safety_min_seconds <= 0:
            raise ValueError("root_safety_min_seconds 必须大于 0。")
        if (
            self.root_safety_min_seconds
            > self.root_safety_max_seconds
        ):
            raise ValueError(
                "root_safety_min_seconds 不能大于 "
                "root_safety_max_seconds。"
            )
        if self.root_safety_extension_bonus < 0:
            raise ValueError("root_safety_extension_bonus 不能小于 0。")
        if self.root_safety_min_completed_depth < 2:
            raise ValueError(
                "root_safety_min_completed_depth 不能小于 2。"
            )
        if not 0 < self.root_sibling_probe_time_fraction < 0.5:
            raise ValueError(
                "root_sibling_probe_time_fraction 必须在 0～0.5 之间。"
            )
        if self.root_sibling_probe_max_seconds <= 0:
            raise ValueError(
                "root_sibling_probe_max_seconds 必须大于 0。"
            )
        if self.root_sibling_probe_min_seconds <= 0:
            raise ValueError(
                "root_sibling_probe_min_seconds 必须大于 0。"
            )
        if (
            self.root_sibling_probe_min_seconds
            > self.root_sibling_probe_max_seconds
        ):
            raise ValueError(
                "root_sibling_probe_min_seconds 不能大于最大值。"
            )
        if self.root_dynamic_review_pvs_limit < 1:
            raise ValueError("root_dynamic_review_pvs_limit 必须大于 0。")
        if self.root_dynamic_review_candidate_limit < 2:
            raise ValueError("root_dynamic_review_candidate_limit 不能小于 2。")
        if not 2 <= self.root_dynamic_review_finalist_limit <= (
            self.root_dynamic_review_candidate_limit
        ):
            raise ValueError("root_dynamic_review_finalist_limit 范围无效。")
        if not 0 < self.root_dynamic_review_time_fraction < 1:
            raise ValueError("root_dynamic_review_time_fraction 必须在 0～1 之间。")
        if self.root_dynamic_review_max_seconds <= 0:
            raise ValueError("root_dynamic_review_max_seconds 必须大于 0。")
        if self.root_dynamic_review_min_seconds <= 0:
            raise ValueError("root_dynamic_review_min_seconds 必须大于 0。")
        if (
            self.root_dynamic_review_min_seconds
            > self.root_dynamic_review_max_seconds
        ):
            raise ValueError("root_dynamic_review_min_seconds 不能大于最大值。")
        if self.root_dynamic_review_min_completed_depth < 2:
            raise ValueError("root_dynamic_review_min_completed_depth 不能小于 2。")
        if self.root_dynamic_review_structure_margin < 0:
            raise ValueError("root_dynamic_review_structure_margin 不能小于 0。")
        if self.root_vcf_safety_max_attacker_moves < 1:
            raise ValueError(
                "root_vcf_safety_max_attacker_moves 必须大于 0。"
            )
        if not 0 < self.root_vcf_safety_time_fraction < 0.5:
            raise ValueError(
                "root_vcf_safety_time_fraction 必须在 0～0.5 之间。"
            )
        if self.root_vcf_safety_max_seconds <= 0:
            raise ValueError(
                "root_vcf_safety_max_seconds 必须大于 0。"
            )
        if self.root_vcf_safety_min_seconds <= 0:
            raise ValueError(
                "root_vcf_safety_min_seconds 必须大于 0。"
            )
        if (
            self.root_vcf_safety_min_seconds
            > self.root_vcf_safety_max_seconds
        ):
            raise ValueError(
                "root_vcf_safety_min_seconds 不能大于 "
                "root_vcf_safety_max_seconds。"
            )
        if not 0 < self.root_vcf_safety_intercept_fraction < 1:
            raise ValueError(
                "root_vcf_safety_intercept_fraction 必须在 0～1 之间。"
            )
        if self.root_quiet_prevention_limit < 0:
            raise ValueError("root_quiet_prevention_limit 不能小于 0。")
        if self.root_quiet_prevention_min_rank < 1:
            raise ValueError(
                "root_quiet_prevention_min_rank 必须大于 0。"
            )
        if self.root_quiet_sibling_prevention_limit < 0:
            raise ValueError(
                "root_quiet_sibling_prevention_limit 不能小于 0。"
            )
        if self.root_quiet_sibling_min_continuations < 2:
            raise ValueError(
                "root_quiet_sibling_min_continuations 不能小于 2。"
            )
        if self.root_offensive_continuation_limit < 0:
            raise ValueError("root_offensive_continuation_limit 不能小于 0。")
        if self.root_offensive_continuation_min_continuations < 2:
            raise ValueError(
                "root_offensive_continuation_min_continuations 不能小于 2。"
            )
        if self.root_dual_frontier_bridge_limit < 0:
            raise ValueError("root_dual_frontier_bridge_limit 不能小于 0。")
        if self.root_dual_frontier_min_own_rank < 1:
            raise ValueError("root_dual_frontier_min_own_rank 必须大于 0。")
        if self.root_dual_frontier_min_opponent_rank < 1:
            raise ValueError(
                "root_dual_frontier_min_opponent_rank 必须大于 0。"
            )
        if self.root_dual_frontier_min_own_continuations < 2:
            raise ValueError(
                "root_dual_frontier_min_own_continuations 不能小于 2。"
            )
        if self.root_dual_frontier_min_opponent_continuations < 2:
            raise ValueError(
                "root_dual_frontier_min_opponent_continuations 不能小于 2。"
            )
        if self.root_quiet_prevention_min_depth < 1:
            raise ValueError(
                "root_quiet_prevention_min_depth 必须大于 0。"
            )
        if self.root_frontier_truth_score_margin < 0:
            raise ValueError(
                "root_frontier_truth_score_margin 不能小于 0。"
            )
        if self.root_survival_scan_limit < 1:
            raise ValueError("root_survival_scan_limit 必须大于 0。")
        if self.root_survival_min_depth < 1:
            raise ValueError("root_survival_min_depth 必须大于 0。")
        if self.root_unverified_advantage_threshold < 1:
            raise ValueError(
                "root_unverified_advantage_threshold 必须大于 0。"
            )
        if self.root_forcing_counterattack_min_depth < 1:
            raise ValueError(
                "root_forcing_counterattack_min_depth 必须大于 0。"
            )


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
    proof_nodes: int = 0
    root_safety_nodes: int = 0
    root_vcf_nodes: int = 0


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


@dataclass(slots=True)
class IterativeSearchOutcome:
    result: RootResult
    candidates: list[Move]
    completed_depth: int
    search_completed: bool
    stop_reason: str
    root_candidates_expanded: bool
    root_expansion_reason: str | None = None


@dataclass(frozen=True, slots=True)
class DefenseProbeResult:
    """One bounded narrow-branch defense comparison."""

    completed_depth: int
    nodes: int
    candidates: tuple[DefenseCandidateAnalysis, ...]

    @property
    def best_move(self) -> Move | None:
        return self.candidates[0].move if self.candidates else None


@dataclass(frozen=True, slots=True)
class RootSafetyProbeResult:
    """Independent full-window comparison without proof semantics."""

    trigger: str
    pvs_gap: int
    main_rank_stable: bool
    completed_depth: int
    nodes: int
    candidates: tuple[RootSafetyCandidateAnalysis, ...]
    leader_history: tuple[Move, ...] = ()
    approved_move: Move | None = None
    selection_basis: str = "equal_window"

    @property
    def best_move(self) -> Move | None:
        if self.approved_move is not None:
            return self.approved_move
        return self.candidates[0].move if self.candidates else None

    @property
    def rank_stable(self) -> bool:
        return (
            len(self.leader_history) >= 2
            and self.leader_history[-1] == self.leader_history[-2]
        )


@dataclass(frozen=True, slots=True)
class RootVCFScanResult:
    """One bounded opponent-VCF pass over the complete root set."""

    original_candidates: tuple[Move, ...]
    candidates: tuple[Move, ...]
    baseline_line: tuple[Move, ...]
    analyses: tuple[RootVCFCandidateAnalysis, ...]
    nodes: int
    elapsed_seconds: float
    exhaustive_rescue_scanned: bool = False
    rescue_candidates_checked: int = 0

    @property
    def complete(self) -> bool:
        return bool(self.analyses) and all(
            candidate.completed
            for candidate in self.analyses
        )
