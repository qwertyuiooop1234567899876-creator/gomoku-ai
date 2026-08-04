"""Policy helpers for the bounded, post-PVS root review.

The review is intentionally heuristic.  It compares candidates on equal
search windows and uses threat-frontier balance only when that structural
signal materially disagrees with a shallow or horizon-clamped probe.  None of
the results in this module are proof states.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from engine.ai import Move, ProofCandidateAnalysis
from engine.proof_search import ProofState
from engine.search_types import (
    HEURISTIC_SCORE_LIMIT,
    RootResult,
    RootSafetyProbeResult,
    SearchConfig,
)
from engine.threats import ThreatFrontier


def frontier_balance_score(
    own_frontiers: Sequence[ThreatFrontier],
    opponent_frontiers: Sequence[ThreatFrontier],
) -> int:
    """Return one bounded structural attack-versus-pressure differential.

    The score rewards both the number of high-rank future gain points and the
    breadth of their continuations.  Small forcing/medium-rank terms provide
    deterministic tie-breaks without turning the value into a mate claim.
    """

    own = _frontier_summary(own_frontiers)
    opponent = _frontier_summary(opponent_frontiers)
    high_delta = own[0] - opponent[0]
    continuation_delta = own[1] - opponent[1]
    forcing_delta = own[2] - opponent[2]
    medium_delta = own[3] - opponent[3]
    return (
        (high_delta + continuation_delta) * 100
        + forcing_delta * 10
        + medium_delta
    )


def _frontier_summary(
    frontiers: Sequence[ThreatFrontier],
) -> tuple[int, int, int, int]:
    ranks = [
        max(frontier.continuation_ranks, default=0)
        for frontier in frontiers
    ]
    return (
        sum(rank >= 80 for rank in ranks),
        sum(
            len(frontier.continuations)
            for frontier, rank in zip(frontiers, ranks, strict=True)
            if rank >= 80
        ),
        sum(rank >= 90 for rank in ranks),
        sum(rank >= 60 for rank in ranks),
    )


def lowest_unknown_risk_move(
    analyses: Sequence[ProofCandidateAnalysis],
    available: set[Move],
) -> Move | None:
    candidates = [
        analysis
        for analysis in analyses
        if (
            analysis.move in available
            and analysis.state == ProofState.UNKNOWN.value
            and analysis.threat_risk is not None
        )
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda analysis: int(analysis.threat_risk),
    ).move


def review_pool(
    config: SearchConfig,
    result: RootResult,
    proof_candidates: Sequence[ProofCandidateAnalysis],
    *,
    quiet_moves: Sequence[Move],
    offensive_moves: Sequence[Move],
) -> list[Move]:
    """Unify PVS, risk, quiet, and offensive sources under one fixed cap."""

    available = {move for move, _score in result.ranked_moves}
    ranked_others = [
        move
        for move, _score in result.ranked_moves
        if move != result.move
    ]
    pvs = [
        result.move,
        *ranked_others[: config.root_dynamic_review_pvs_limit],
    ]
    risk = lowest_unknown_risk_move(proof_candidates, available)
    source_order = [
        *pvs,
        *(() if risk is None else (risk,)),
        *quiet_moves,
        *offensive_moves,
    ]
    eligible = [move for move in dict.fromkeys(source_order) if move in available]

    required = list(
        dict.fromkeys(
            (
                result.move,
                *pvs[: config.root_dynamic_review_pvs_limit],
                *offensive_moves,
            )
        )
    )
    if len(required) >= config.root_dynamic_review_candidate_limit:
        return required[: config.root_dynamic_review_candidate_limit]

    pending = set(required)
    selected: list[Move] = []
    for move in eligible:
        is_required = move in pending
        if (
            not is_required
            and config.root_dynamic_review_candidate_limit - len(selected)
            <= len(pending)
        ):
            continue
        selected.append(move)
        pending.discard(move)
        if len(selected) >= config.root_dynamic_review_candidate_limit:
            break
    return selected


def finalists(
    config: SearchConfig,
    result: RootResult,
    pool: Sequence[Move],
    structure_scores: Mapping[Move, int],
    *,
    preferred_moves: Sequence[Move] = (),
) -> list[Move]:
    """Keep the PVS leader plus the strongest structural challengers."""

    pvs = [
        result.move,
        *(
            move
            for move, _score in result.ranked_moves
            if move != result.move and move in pool
        ),
    ][: config.root_dynamic_review_pvs_limit]
    preferred = sorted(
        (
            move
            for move in preferred_moves
            if move in pool and move not in pvs
        ),
        key=lambda move: (
            structure_scores.get(move, -10**18),
            -pool.index(move),
        ),
        reverse=True,
    )
    challengers = sorted(
        (move for move in pool if move != result.move),
        key=lambda move: (
            structure_scores.get(move, -10**18),
            -pool.index(move),
        ),
        reverse=True,
    )
    return list(
        dict.fromkeys(
            (*pvs, *preferred, *challengers)
        )
    )[: config.root_dynamic_review_finalist_limit]


def approve_move(
    config: SearchConfig,
    result: RootResult,
    probe: RootSafetyProbeResult,
    structure_scores: Mapping[Move, int],
) -> tuple[Move, str]:
    """Choose the review recommendation without inventing proof evidence."""

    if not probe.candidates:
        return result.move, "pvs_fallback"

    searched = [candidate.move for candidate in probe.candidates]
    structural = max(
        searched,
        key=lambda move: structure_scores.get(move, -10**18),
    )
    probe_leader = probe.candidates[0].move
    structural_gap = (
        structure_scores.get(structural, -10**18)
        - structure_scores.get(probe_leader, -10**18)
    )
    if probe.completed_depth >= 3:
        top_score = probe.candidates[0].score
        tied = [
            candidate.move
            for candidate in probe.candidates
            if candidate.score == top_score
        ]
        if len(tied) > 1:
            tied_structural = max(
                tied,
                key=lambda move: structure_scores.get(move, -10**18),
            )
            tied_gap = (
                structure_scores.get(tied_structural, -10**18)
                - structure_scores.get(result.move, -10**18)
            )
            if tied_gap >= config.root_dynamic_review_structure_margin:
                return tied_structural, "frontier_balance"
            return result.move, "pvs_fallback"
    if (
        probe.rank_stable
        and probe.completed_depth
        >= config.root_dynamic_review_min_completed_depth
        and abs(probe.candidates[0].score) < HEURISTIC_SCORE_LIMIT
        and structural_gap
        < config.root_dynamic_review_structure_margin * 2
    ):
        return probe_leader, "equal_window"
    if (
        probe.completed_depth >= 4
        and structural_gap
        >= config.root_dynamic_review_structure_margin
    ):
        return structural, "frontier_balance"
    return result.move, "pvs_fallback"
