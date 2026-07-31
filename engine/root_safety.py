"""Pure root-safety policy.

The expensive comparison search remains owned by the search coordinator.
This module decides *when* that comparison is allowed to run and *how* a
completed probe may reorder a root result.  It deliberately has no board,
clock, transposition-table, or PVS dependencies.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from engine.ai import Move, RootVCFCandidateAnalysis
from engine.board import Board
from engine.proof_search import ProofState
from engine.search_types import (
    RootResult,
    RootSafetyProbeResult,
    RootVCFScanResult,
    SearchConfig,
    SearchTimeout,
    VCFTimeout,
)

FindVCF = Callable[
    [Board, int, float | None],
    tuple[Move, ...] | None,
]
NodeCount = Callable[[], int]
Clock = Callable[[], float]


class RootCandidateSafety(str, Enum):
    """Safety semantics reserved for the later opponent-VCF scan.

    ``SURVIVES_VCF_SCAN`` only means that the bounded scan found no VCF.  It
    must never be promoted to a globally proven safe state.
    """

    PROVEN_LOSS = "proven_loss"
    SURVIVES_VCF_SCAN = "survives_vcf_scan"
    UNKNOWN = "unknown"


def vcf_intercept_moves(line: tuple[Move, ...]) -> tuple[Move, ...]:
    """Return every playable point from an alternating VCF witness.

    Pre-occupying either an attacker gain point or a forced defender reply can
    invalidate the concrete line. The later per-candidate scan decides whether
    the resulting position actually survives.
    """
    return tuple(dict.fromkeys(line))


def merge_vcf_intercepts(
    candidates: list[Move],
    line: tuple[Move, ...],
    *,
    limit: int | None = None,
) -> list[Move]:
    """Append the complete witness while preserving the fixed root limit."""
    intercepts = vcf_intercept_moves(line)
    if limit is None:
        return list(dict.fromkeys((*candidates, *intercepts)))
    if limit < 1:
        return []
    if len(intercepts) >= limit:
        return list(intercepts[:limit])

    pending = set(intercepts)
    selected: list[Move] = []
    for move in dict.fromkeys((*candidates, *intercepts)):
        is_intercept = move in pending
        if (
            not is_intercept
            and limit - len(selected) <= len(pending)
        ):
            continue
        selected.append(move)
        if is_intercept:
            pending.remove(move)
        if len(selected) >= limit:
            break
    return selected


def vcf_scan_order(
    candidates: list[Move],
    line: tuple[Move, ...],
) -> list[Move]:
    """Check known interception points first, then the normal root order."""
    intercepts = set(vcf_intercept_moves(line))
    return [
        *(move for move in candidates if move in intercepts),
        *(move for move in candidates if move not in intercepts),
    ]


def apply_vcf_scan(
    candidates: list[Move],
    analyses: tuple[RootVCFCandidateAnalysis, ...],
) -> list[Move]:
    """Apply bounded VCF evidence without treating a miss as global proof.

    A completed bounded scan is stronger than an unscanned/timeout result,
    while ``UNKNOWN`` remains eligible when no completed survivor exists.
    If every candidate has a proven opponent VCF, retain the original set so
    the ordinary search can still choose the longest practical resistance.
    """
    if not candidates or not analyses:
        return list(candidates)

    statuses = {
        candidate.move: candidate.status
        for candidate in analyses
    }
    survivors = [
        move
        for move in candidates
        if (
            statuses.get(move)
            == RootCandidateSafety.SURVIVES_VCF_SCAN.value
        )
    ]
    if survivors:
        return survivors

    unknown = [
        move
        for move in candidates
        if (
            statuses.get(move, RootCandidateSafety.UNKNOWN.value)
            == RootCandidateSafety.UNKNOWN.value
        )
    ]
    return unknown or list(candidates)


@dataclass(frozen=True, slots=True)
class RootVCFSafetyScanner:
    """Run the bounded opponent-VCF channel without owning PVS state."""

    find_vcf: FindVCF
    node_count: NodeCount
    intercept_fraction: float = 0.25
    minimum_candidate_seconds: float = 0.01
    candidate_limit: int | None = None
    exhaustive_rescue_enabled: bool = False
    rescue_survivor_threshold: int = 1
    clock: Clock = time.perf_counter

    def scan(
        self,
        board: Board,
        candidates: list[Move],
        *,
        mover: int,
        opponent: int,
        budget_seconds: float | None,
        hard_deadline: float | None,
    ) -> RootVCFScanResult:
        started_at = self.clock()
        global_deadline = self._global_deadline(
            started_at,
            budget_seconds,
            hard_deadline,
        )
        start_nodes = self.node_count()
        baseline_line = self._baseline_line(
            board,
            opponent,
            started_at=started_at,
            budget_seconds=budget_seconds,
            global_deadline=global_deadline,
        )
        merged = [
            move
            for move in merge_vcf_intercepts(
                candidates,
                baseline_line,
                limit=self.candidate_limit,
            )
            if board.is_empty(*move)
        ]
        ordered = vcf_scan_order(merged, baseline_line)
        analyses = self._scan_candidates(
            board,
            ordered,
            mover=mover,
            opponent=opponent,
            global_deadline=global_deadline,
        )
        rescue_scanned = False
        rescue_checked = 0
        if self._should_scan_global_rescues(
            baseline_line,
            analyses,
            global_deadline=global_deadline,
        ):
            rescue_scanned = True
            already_scanned = set(ordered)
            remaining = [
                move
                for move in board.get_legal_moves()
                if move not in already_scanned
            ]
            rescue_checked = len(remaining)
            rescue_analyses = self._scan_candidates(
                board,
                remaining,
                mover=mover,
                opponent=opponent,
                global_deadline=global_deadline,
            )
            analyses.extend(rescue_analyses)
            discovered = [
                candidate.move
                for candidate in rescue_analyses
                if candidate.status
                == RootCandidateSafety.SURVIVES_VCF_SCAN.value
            ]
            merged = list(dict.fromkeys((*merged, *discovered)))
        return RootVCFScanResult(
            original_candidates=tuple(candidates),
            candidates=tuple(merged),
            baseline_line=baseline_line,
            analyses=tuple(analyses),
            nodes=self.node_count() - start_nodes,
            elapsed_seconds=self.clock() - started_at,
            exhaustive_rescue_scanned=rescue_scanned,
            rescue_candidates_checked=rescue_checked,
        )

    def _should_scan_global_rescues(
        self,
        baseline_line: tuple[Move, ...],
        analyses: list[RootVCFCandidateAnalysis],
        *,
        global_deadline: float | None,
    ) -> bool:
        if not self.exhaustive_rescue_enabled or not baseline_line:
            return False
        if global_deadline is not None and self.clock() >= global_deadline:
            return False
        if not analyses or not all(item.completed for item in analyses):
            return False
        survivors = sum(
            item.status == RootCandidateSafety.SURVIVES_VCF_SCAN.value
            for item in analyses
        )
        return survivors <= self.rescue_survivor_threshold

    @staticmethod
    def _global_deadline(
        started_at: float,
        budget_seconds: float | None,
        hard_deadline: float | None,
    ) -> float | None:
        budget_deadline = (
            None
            if budget_seconds is None
            else started_at + budget_seconds
        )
        deadlines = [
            deadline
            for deadline in (budget_deadline, hard_deadline)
            if deadline is not None
        ]
        return min(deadlines) if deadlines else None

    def _baseline_line(
        self,
        board: Board,
        opponent: int,
        *,
        started_at: float,
        budget_seconds: float | None,
        global_deadline: float | None,
    ) -> tuple[Move, ...]:
        intercept_deadline = global_deadline
        if budget_seconds is not None:
            local_deadline = (
                started_at
                + budget_seconds * self.intercept_fraction
            )
            intercept_deadline = (
                local_deadline
                if global_deadline is None
                else min(global_deadline, local_deadline)
            )
        try:
            return (
                self.find_vcf(
                    board,
                    opponent,
                    intercept_deadline,
                )
                or ()
            )
        except (SearchTimeout, VCFTimeout):
            return ()

    def _scan_candidates(
        self,
        board: Board,
        candidates: list[Move],
        *,
        mover: int,
        opponent: int,
        global_deadline: float | None,
    ) -> list[RootVCFCandidateAnalysis]:
        analyses: list[RootVCFCandidateAnalysis] = []
        for index, move in enumerate(candidates):
            started_at = self.clock()
            node_start = self.node_count()
            if (
                global_deadline is not None
                and started_at >= global_deadline
            ):
                analyses.append(self._unknown(move))
                continue

            candidate_deadline = self._candidate_deadline(
                started_at,
                global_deadline,
                len(candidates) - index,
            )
            completed = True
            line: tuple[Move, ...] | None = None
            board.place(*move, mover)
            try:
                if not board.check_win(*move):
                    try:
                        line = self.find_vcf(
                            board,
                            opponent,
                            candidate_deadline,
                        )
                    except (SearchTimeout, VCFTimeout):
                        completed = False
            finally:
                board.undo()

            analyses.append(
                RootVCFCandidateAnalysis(
                    move=move,
                    status=self._status(completed, line),
                    completed=completed,
                    nodes=self.node_count() - node_start,
                    elapsed_seconds=self.clock() - started_at,
                    principal_variation=line or (),
                )
            )
        return analyses

    def _candidate_deadline(
        self,
        started_at: float,
        global_deadline: float | None,
        candidates_left: int,
    ) -> float | None:
        if global_deadline is None:
            return None
        remaining = max(0.0, global_deadline - started_at)
        return min(
            global_deadline,
            started_at
            + max(
                self.minimum_candidate_seconds,
                remaining / candidates_left,
            ),
        )

    @staticmethod
    def _status(
        completed: bool,
        line: tuple[Move, ...] | None,
    ) -> str:
        if not completed:
            return RootCandidateSafety.UNKNOWN.value
        if line:
            return RootCandidateSafety.PROVEN_LOSS.value
        return RootCandidateSafety.SURVIVES_VCF_SCAN.value

    @staticmethod
    def _unknown(move: Move) -> RootVCFCandidateAnalysis:
        return RootVCFCandidateAnalysis(
            move=move,
            status=RootCandidateSafety.UNKNOWN.value,
            completed=False,
            nodes=0,
            elapsed_seconds=0.0,
        )


def rank_is_stable(root_history: list[RootResult]) -> bool:
    if len(root_history) < 2:
        return False
    recent = root_history[-3:]
    return all(
        result.move == recent[-1].move
        for result in recent[:-1]
    )


def trigger(
    config: SearchConfig,
    result: RootResult,
    root_history: list[RootResult],
    *,
    proof_states: dict[Move, str],
    mate_scores_quarantined: bool,
) -> str | None:
    """Return why an incomplete near-tie needs an independent check."""
    if not config.root_safety_enabled or mate_scores_quarantined:
        return None
    if proof_states.get(result.move) == ProofState.PROVEN_LOSS.value:
        return None

    ranked = sorted(
        result.ranked_moves,
        key=lambda item: item[1],
        reverse=True,
    )
    if len(ranked) < 2:
        return None

    pvs_gap = ranked[0][1] - ranked[1][1]
    if pvs_gap > config.root_safety_score_margin:
        return None
    if pvs_gap <= config.root_safety_micro_margin:
        return "micro_pvs_gap"
    if not rank_is_stable(root_history):
        return "root_rank_changed"
    return None


def candidates(
    config: SearchConfig,
    result: RootResult,
) -> tuple[list[Move], int]:
    ranked = sorted(
        result.ranked_moves,
        key=lambda item: item[1],
        reverse=True,
    )
    if len(ranked) < 2:
        return [], 0

    pvs_gap = ranked[0][1] - ranked[1][1]
    best_score = ranked[0][1]
    selected = [
        move
        for move, score in ranked
        if best_score - score <= config.root_safety_score_margin
    ][: config.root_safety_candidate_limit]
    return selected, pvs_gap


def budget_seconds(
    config: SearchConfig,
    *,
    remaining_seconds: float | None,
) -> float:
    total = config.time_limit_seconds
    if total is None or remaining_seconds is None:
        return 0.0

    budget = min(
        config.root_safety_max_seconds,
        total * config.root_safety_time_fraction,
        max(0.0, remaining_seconds - 0.05),
    )
    if budget < config.root_safety_min_seconds:
        return 0.0
    return budget


def apply_probe(
    config: SearchConfig,
    result: RootResult,
    probe: RootSafetyProbeResult,
) -> RootResult:
    """Use only a repeated probe leader; preserve original PVS scores."""
    if (
        not probe.rank_stable
        or probe.completed_depth < config.root_safety_min_completed_depth
        or probe.best_move is None
        or probe.best_move == result.move
    ):
        return result

    root_scores = dict(result.ranked_moves)
    if probe.best_move not in root_scores:
        return result

    chosen = probe.best_move
    probe_variations = {
        candidate.move: candidate.principal_variation
        for candidate in probe.candidates
    }
    variation_map = {
        move: pv
        for move, _, pv in result.ranked_variations
    }
    chosen_pv = probe_variations.get(
        chosen,
        variation_map.get(chosen, (chosen,)),
    )
    reordered = (
        (chosen, root_scores[chosen]),
        *(
            item
            for item in result.ranked_moves
            if item[0] != chosen
        ),
    )
    reordered_variations = (
        (chosen, root_scores[chosen], chosen_pv),
        *(
            item
            for item in result.ranked_variations
            if item[0] != chosen
        ),
    )
    return RootResult(
        move=chosen,
        score=root_scores[chosen],
        principal_variation=chosen_pv,
        ranked_moves=reordered,
        ranked_variations=reordered_variations,
    )
