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


def frontier_shape_key(
    own_frontiers: Sequence[ThreatFrontier],
    opponent_frontiers: Sequence[ThreatFrontier],
) -> tuple[int, int, int, int]:
    """Return a lexicographic topology key for unresolved horizon ties.

    The ordinary scalar balance intentionally treats one extra high-rank gain
    and one extra continuation as equivalent.  When an equal-window probe is
    saturated at the selective horizon, prefer broader independent gain points
    before continuation count.  The key remains heuristic and has no proof
    semantics.
    """
    own = _frontier_summary(own_frontiers)
    opponent = _frontier_summary(opponent_frontiers)
    return (
        own[0] - opponent[0],
        own[2] - opponent[2],
        own[1] - opponent[1],
        own[3] - opponent[3],
    )


def has_horizon_boundary(
    candidates: Sequence[object],
) -> bool:
    """Return whether any candidate score is a selective clamp boundary."""
    return any(
        abs(int(getattr(candidate, "score"))) >= HEURISTIC_SCORE_LIMIT
        for candidate in candidates
    )


def credible_layer_leader(
    candidates: Sequence[object],
    *,
    score_margin: int,
) -> Move | None:
    """Return a bounded equal-window winner suitable for depth consensus.

    Mandatory-defense searches are especially prone to alternating between
    a quiet heuristic value and a selective tactical cliff.  A layer only
    votes when both candidates stayed away from the clamp boundary and the
    numeric gap is small enough to compare as an ordinary heuristic result.
    Exact ties deliberately abstain instead of preserving move-order bias.
    """
    if len(candidates) < 2 or has_horizon_boundary(candidates):
        return None
    scores = sorted(
        (int(getattr(candidate, "score")) for candidate in candidates),
        reverse=True,
    )
    gap = scores[0] - scores[1]
    if gap <= 0 or gap > score_margin:
        return None
    return getattr(candidates[0], "move")


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
    preferred_groups: Sequence[Sequence[Move]] = (),
) -> list[Move]:
    """Keep the PVS leader plus bounded, source-diverse challengers."""

    root_rank = {
        move: index
        for index, (move, _score) in enumerate(result.ranked_moves)
    }
    selected = [result.move]
    # Source diversity must not crowd out the strongest searched challenger.
    # ``ranked_moves`` may deliberately preserve tactical-source order after
    # false-mate quarantine, so choose this one reservation by the bounded
    # numeric score rather than by tuple position.
    score_challengers = [
        (move, score)
        for move, score in result.ranked_moves
        if move in pool and move != result.move
    ]
    if score_challengers:
        selected.append(
            max(
                score_challengers,
                key=lambda item: (
                    item[1],
                    -root_rank.get(item[0], len(root_rank)),
                ),
            )[0]
        )
        if len(selected) >= config.root_dynamic_review_finalist_limit:
            return selected
    groups = [
        *((preferred_moves,) if preferred_moves else ()),
        *preferred_groups,
    ]
    for group in groups:
        eligible = [
            move
            for move in dict.fromkeys(group)
            if move in pool and move not in selected
        ]
        if not eligible:
            continue
        selected.append(
            min(
                eligible,
                key=lambda move: (
                    root_rank.get(move, len(root_rank)),
                    -structure_scores.get(move, -10**18),
                    pool.index(move),
                ),
            )
        )
        if len(selected) >= config.root_dynamic_review_finalist_limit:
            return selected

    pvs = [
        move
        for move, _score in result.ranked_moves
        if move in pool and move not in selected
    ][: config.root_dynamic_review_pvs_limit]
    challengers = sorted(
        (move for move in pool if move not in selected and move not in pvs),
        key=lambda move: (
            structure_scores.get(move, -10**18),
            -pool.index(move),
        ),
        reverse=True,
    )
    return list(
        dict.fromkeys(
            (*selected, *pvs, *challengers)
        )
    )[: config.root_dynamic_review_finalist_limit]


def approve_move(
    config: SearchConfig,
    result: RootResult,
    probe: RootSafetyProbeResult,
    structure_scores: Mapping[Move, int],
    *,
    structure_keys: Mapping[Move, tuple[int, ...]] | None = None,
    unknown_moves: set[Move] | None = None,
    mandatory_defense_consensus: bool = False,
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
    if mandatory_defense_consensus and probe.completed_depth >= 4:
        ordinary = [
            candidate.move
            for candidate in probe.candidates
            if abs(candidate.score) < HEURISTIC_SCORE_LIMIT
        ]
        if len(ordinary) == 1:
            # The paired search is still heuristic, but a single candidate
            # escaping the selective cliff is materially safer than keeping
            # an old PVS tie whose sibling remains pinned to that boundary.
            return ordinary[0], "mandatory_boundary_escape"
        if (
            probe.rank_stable
            and probe.completed_depth
            >= config.root_dynamic_review_min_completed_depth
            and not has_horizon_boundary(probe.candidates)
        ):
            return probe_leader, "mandatory_depth_consensus"
    if (
        probe.rank_stable
        and probe.completed_depth >= 4
        and probe.candidates[0].score >= HEURISTIC_SCORE_LIMIT
        and probe.pvs_gap <= config.root_safety_score_margin
        and unknown_moves
        and all(move in unknown_moves for move in searched)
    ):
        # A repeated positive clamp from the isolated full-window probe is
        # useful move-order evidence even though it is not strict mate proof.
        # Keep the state UNKNOWN; only approve the heuristic root choice.
        return probe_leader, "equal_window"
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
            if (
                probe.pvs_gap <= config.root_safety_score_margin
                and tied_gap
                >= min(
                    config.root_dynamic_review_structure_margin,
                    config.root_safety_micro_margin,
                )
            ):
                return tied_structural, "frontier_balance"
            return result.move, "pvs_fallback"
    if (
        probe.rank_stable
        and probe.completed_depth >= 4
        and abs(probe.candidates[0].score) < HEURISTIC_SCORE_LIMIT
        and has_horizon_boundary(probe.candidates)
        and probe.pvs_gap > config.root_safety_micro_margin
    ):
        # A stable, ordinary-score leader is useful escape evidence when its
        # sibling alone remains pinned to the selective clamp.  Near-exact
        # original PVS ties are excluded so the topology fallback can still
        # arbitrate symmetric horizon cases.
        return probe_leader, "equal_window"
    if (
        probe.rank_stable
        and probe.completed_depth
        >= config.root_dynamic_review_min_completed_depth
        and not has_horizon_boundary(probe.candidates)
        and structural_gap
        < config.root_dynamic_review_structure_margin * 2
    ):
        return probe_leader, "equal_window"
    if (
        probe.completed_depth >= 4
        and probe.pvs_gap <= config.root_safety_score_margin
        and structural_gap
        >= config.root_dynamic_review_structure_margin
    ):
        return structural, "frontier_balance"
    if (
        has_horizon_boundary(probe.candidates)
        and structure_keys
        and unknown_moves
        and all(move in unknown_moves for move in searched)
    ):
        root_scores = dict(result.ranked_moves)
        pvs_scores = [root_scores[move] for move in searched]
        if max(pvs_scores) - min(pvs_scores) <= config.root_safety_micro_margin:
            shaped = max(
                searched,
                key=lambda move: structure_keys.get(move, ()),
            )
            if shaped != result.move:
                return shaped, "frontier_shape"
    return result.move, "pvs_fallback"
