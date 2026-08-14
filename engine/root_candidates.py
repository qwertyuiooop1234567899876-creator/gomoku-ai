"""Root-candidate set construction helpers.

The module only combines and classifies candidate sets.  It does not score
moves, search variations, or decide which move is finally returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from engine.ai import Move
from engine.evaluator import ThreatProfile
from engine.search_types import (
    DefenseProbeResult,
    MATE_SCORE,
    RootResult,
)
from engine.threats import ThreatFrontier, ThreatKind


class CandidateSource(str, Enum):
    ORDINARY = "ordinary"
    OWN_FORCING = "own_forcing"
    ACTIVE_COUNTERATTACK = "active_counterattack"
    FORCING_COUNTERATTACK = "forcing_counterattack"
    MANDATORY_DEFENSE = "mandatory_defense"
    THREAT_FRONTIER = "threat_frontier"
    PRESSURE_PREVENTION = "pressure_prevention"
    QUIET_PREVENTION = "quiet_prevention"
    OFFENSIVE_CONTINUATION = "offensive_continuation"
    DUAL_FRONTIER_BRIDGE = "dual_frontier_bridge"
    VCF_INTERCEPT = "vcf_intercept"
    ROOT_EXPANSION = "root_expansion"


class RootCandidateMode(str, Enum):
    MERGED_FORCING = "merged_forcing"
    OWN_FORCING = "own_forcing"
    MANDATORY_DEFENSE = "mandatory_defense"
    FRONTIER_DEFENSE = "frontier_defense"
    ORDINARY = "ordinary"


PROOF_SOURCE_PRIORITY = (
    CandidateSource.MANDATORY_DEFENSE,
    CandidateSource.VCF_INTERCEPT,
    CandidateSource.THREAT_FRONTIER,
    CandidateSource.PRESSURE_PREVENTION,
    CandidateSource.QUIET_PREVENTION,
    CandidateSource.DUAL_FRONTIER_BRIDGE,
    CandidateSource.OWN_FORCING,
    CandidateSource.FORCING_COUNTERATTACK,
    CandidateSource.ACTIVE_COUNTERATTACK,
    CandidateSource.OFFENSIVE_CONTINUATION,
    CandidateSource.ROOT_EXPANSION,
    CandidateSource.ORDINARY,
)


@dataclass(frozen=True, slots=True)
class CandidateEntry:
    move: Move
    sources: frozenset[CandidateSource]


@dataclass(slots=True)
class RootCandidatePlan:
    """Search-ready root set plus the evidence used to build it."""

    moves: list[Move]
    own_profiles: dict[Move, ThreatProfile]
    opponent_profiles: dict[Move, ThreatProfile]
    own_forcing_moves: list[Move]
    preserve_frontier_order: bool
    allow_near_loss_expansion: bool
    defense_probe: DefenseProbeResult | None
    reason: str
    entries: tuple[CandidateEntry, ...] = ()


def classify_mode(
    *,
    own_forcing_moves: list[Move],
    opponent_forcing_moves: list[Move],
    opponent_frontier_moves: list[Move],
) -> RootCandidateMode:
    if own_forcing_moves and opponent_forcing_moves:
        return RootCandidateMode.MERGED_FORCING
    if own_forcing_moves:
        return RootCandidateMode.OWN_FORCING
    if opponent_forcing_moves:
        return RootCandidateMode.MANDATORY_DEFENSE
    if opponent_frontier_moves:
        return RootCandidateMode.FRONTIER_DEFENSE
    return RootCandidateMode.ORDINARY


def strongest_frontier_moves(
    frontiers: dict[Move, tuple[Move, ...]],
) -> list[Move]:
    if not frontiers:
        return []
    maximum_size = max(len(replies) for replies in frontiers.values())
    return [
        move
        for move, replies in frontiers.items()
        if len(replies) == maximum_size
    ]


def active_counterattack_moves(
    profiles: dict[Move, ThreatProfile],
) -> list[Move]:
    """Keep quiet open-three counterplay available during frontier defense.

    Direct four-making moves remain outside this first-stage widening until
    they receive their own reply-aware arbitration; admitting them solely by
    static rank can displace already verified defensive regressions.
    """
    return [
        move
        for move, profile in profiles.items()
        if (
            not profile.forced_win
            and profile.open_three_directions >= 1
            and profile.four_directions == 0
        )
    ]


def forcing_counterattack_moves(
    profiles: dict[Move, ThreatProfile],
) -> list[Move]:
    """Keep tempo-gaining fours during opponent-frontier defense.

    A single four is not itself a forced win, but it obliges an immediate
    reply.  Dropping it before PVS can remove the only move that both gains a
    tempo and breaks a longer opponent threat chain.
    """
    return [
        move
        for move, profile in profiles.items()
        if not profile.forced_win and profile.four_directions >= 1
    ]


def merge_unique(*groups: Iterable[Move]) -> list[Move]:
    return list(
        dict.fromkeys(
            move
            for group in groups
            for move in group
        )
    )


def source_diverse_subset(
    moves: Iterable[Move],
    sources_by_move: dict[Move, frozenset[CandidateSource]],
    *,
    limit: int,
) -> list[Move]:
    """Keep the leading move plus bounded representatives for Proof.

    Initial Proof has fewer slots than the full searched root.  Reserving a
    representative from each tactical source prevents all strict checks from
    being consumed by several near-identical frontier moves.  Membership and
    order of the full PVS root are unchanged.
    """
    ordered = list(dict.fromkeys(moves))
    if limit <= 0:
        return []
    if len(ordered) <= limit or not sources_by_move:
        return ordered[:limit]

    selected = [ordered[0]]
    for source in PROOF_SOURCE_PRIORITY:
        if any(
            source in sources_by_move.get(move, ())
            for move in selected
        ):
            continue
        representative = next(
            (
                move
                for move in ordered
                if (
                    move not in selected
                    and source in sources_by_move.get(move, ())
                )
            ),
            None,
        )
        if representative is None:
            continue
        selected.append(representative)
        if len(selected) >= limit:
            return selected

    selected.extend(move for move in ordered if move not in selected)
    return selected[:limit]


def merge_with_required(
    *,
    ordered_groups: Iterable[Iterable[Move]],
    required_groups: Iterable[Iterable[Move]],
    limit: int,
) -> list[Move]:
    """Merge ordered groups while reserving room for tactical evidence.

    The normal group order remains authoritative whenever possible. Moves
    from ``required_groups`` are nevertheless guaranteed a slot before quiet
    fillers consume the fixed root limit. This keeps candidate completeness
    separate from search scoring and avoids silently widening the root.
    """
    if limit < 1:
        return []

    ordered_group_list = [tuple(group) for group in ordered_groups]
    required_group_list = [tuple(group) for group in required_groups]
    ordered = merge_unique(*ordered_group_list)
    required = merge_unique(*required_group_list)
    if len(required) >= limit:
        # When required evidence itself exceeds the fixed root cap, a plain
        # flattened slice lets the earliest large source erase every later
        # source. Reserve one representative from each non-empty group, then
        # fill the remaining slots in the original tactical order.
        representatives = merge_unique(
            *(
                group[:1]
                for group in required_group_list
                if group
            )
        )[:limit]
        pending = set(representatives)
        selected: list[Move] = []
        for move in required:
            is_reserved = move in pending
            if (
                not is_reserved
                and limit - len(selected) <= len(pending)
            ):
                continue
            selected.append(move)
            pending.discard(move)
            if len(selected) >= limit:
                break
        return selected

    pending = set(required)
    selected: list[Move] = []
    for move in ordered:
        is_required = move in pending
        if (
            not is_required
            and limit - len(selected) <= len(pending)
        ):
            continue
        selected.append(move)
        if is_required:
            pending.remove(move)
        if len(selected) >= limit:
            break
    return selected


def frontier_defense_moves(
    *,
    frontier_moves: Iterable[Move],
    ordinary_moves: Iterable[Move],
    counterattack_moves: Iterable[Move],
    limit: int,
    forcing_counterattack_moves: Iterable[Move] = (),
    pressure_prevention_moves: Iterable[Move] = (),
    prevention_moves: Iterable[Move] = (),
    offensive_continuation_moves: Iterable[Move] = (),
    dual_frontier_moves: Iterable[Move] = (),
) -> list[Move]:
    """Build a complete but capped frontier-defense root set."""
    frontier = tuple(frontier_moves)
    ordinary = tuple(ordinary_moves)
    counterattacks = tuple(counterattack_moves)
    forcing_counterattacks = tuple(forcing_counterattack_moves)
    pressure_prevention = tuple(pressure_prevention_moves)
    prevention = tuple(prevention_moves)
    offensive_continuations = tuple(offensive_continuation_moves)
    dual_frontiers = tuple(dual_frontier_moves)
    return merge_with_required(
        ordered_groups=(
            frontier,
            ordinary,
            forcing_counterattacks,
            pressure_prevention,
            prevention,
            dual_frontiers,
            counterattacks,
            offensive_continuations,
        ),
        required_groups=(
            frontier,
            forcing_counterattacks,
            pressure_prevention,
            prevention,
            dual_frontiers,
            counterattacks,
        ),
        limit=limit,
    )


def pressure_prevention_moves(
    *,
    frontiers: Iterable[ThreatFrontier],
    covered_moves: Iterable[Move],
    strong_rank: int,
    minimum_continuations: int,
    limit: int,
) -> list[Move]:
    """Keep bounded non-quiet pressure points omitted by frontier truth.

    Multi-frontier mode deliberately avoids admitting the whole ordinary
    root list.  The exhaustive pressure scan can nevertheless expose a
    serious open-three or four gain that is not one of the strongest
    multi-threat anchors.  Such a point is defensive evidence and must keep
    one bounded route into PVS; it is never promoted to a proof result.
    """
    if limit <= 0:
        return []

    covered = set(covered_moves)
    ranked: list[tuple[int, int, int, int, Move]] = []
    for order, frontier in enumerate(frontiers):
        maximum_rank = max(frontier.continuation_ranks, default=0)
        if (
            frontier.gain_move in covered
            or frontier.kind is ThreatKind.QUIET
            or len(frontier.continuations) < minimum_continuations
            or maximum_rank < strong_rank
        ):
            continue
        ranked.append(
            (
                maximum_rank,
                len(frontier.continuations),
                sum(frontier.continuation_ranks),
                -order,
                frontier.gain_move,
            )
        )

    ranked.sort(reverse=True)
    return [item[-1] for item in ranked[:limit]]


def mandatory_defense_moves(
    *,
    defense_moves: Iterable[Move],
    forcing_counterattack_moves: Iterable[Move],
    limit: int,
) -> list[Move]:
    """Keep direct blocks and bounded tempo-gaining counterattacks.

    A direct double-threat block is not always the only defensive resource.
    A four-making move can force the opponent to answer first and thereby
    interrupt the threatened sequence.  Both classes are only candidates;
    PVS and Proof still decide whether either line actually survives.
    """
    defenses = tuple(defense_moves)
    counterattacks = tuple(forcing_counterattack_moves)
    return merge_with_required(
        ordered_groups=(defenses, counterattacks),
        required_groups=(defenses, counterattacks),
        limit=limit,
    )


def quiet_frontier_sibling_prevention_moves(
    *,
    frontiers: Iterable[ThreatFrontier],
    anchor_moves: Iterable[Move],
    strong_rank: int,
    minimum_continuations: int,
    limit: int,
) -> list[Move]:
    """Keep bounded quiet alternatives linked to a strong root anchor.

    A lower-rank quiet gain can still matter when it fans out into several
    continuations and shares one of those continuation nodes with a stronger
    gain point that is already an active root move.  This is candidate-only
    evidence: it never upgrades the sibling to a proof result.
    """
    if limit <= 0:
        return []

    ordered_frontiers = tuple(frontiers)
    by_gain = {
        frontier.gain_move: frontier
        for frontier in ordered_frontiers
    }
    anchors = [
        frontier
        for move in anchor_moves
        if (frontier := by_gain.get(move)) is not None
        and frontier.continuation_ranks
        and max(frontier.continuation_ranks) >= strong_rank
    ]
    if not anchors:
        return []

    anchor_continuations = [
        set(frontier.continuations) for frontier in anchors
    ]
    anchor_gains = {frontier.gain_move for frontier in anchors}
    candidates: list[tuple[int, int, int, int, Move]] = []
    for order, frontier in enumerate(ordered_frontiers):
        if (
            frontier.gain_move in anchor_gains
            or frontier.kind is not ThreatKind.QUIET
            or len(frontier.continuations) < minimum_continuations
            or not frontier.continuation_ranks
            or max(frontier.continuation_ranks) >= strong_rank
        ):
            continue
        continuation_set = set(frontier.continuations)
        overlap = sum(
            len(continuation_set & anchor_set)
            for anchor_set in anchor_continuations
        )
        if overlap <= 0:
            continue
        candidates.append(
            (
                overlap,
                len(frontier.continuations),
                max(frontier.continuation_ranks),
                -order,
                frontier.gain_move,
            )
        )

    candidates.sort(reverse=True)
    return [item[-1] for item in candidates[:limit]]


def offensive_continuation_bridges(
    *,
    own_frontiers: Iterable[ThreatFrontier],
    opponent_frontiers: Iterable[ThreatFrontier],
    strong_rank: int,
    minimum_continuations: int,
) -> list[Move]:
    """Return quiet gain points shared by attack and defense networks.

    A bridge fans out into several own continuations while occupying one of
    the continuation points of an opponent high-rank frontier.  It is only a
    candidate-completeness hint and never a proof result.
    """

    opponent_continuations = {
        continuation
        for frontier in opponent_frontiers
        if (
            frontier.continuation_ranks
            and max(frontier.continuation_ranks) >= strong_rank
        )
        for continuation in frontier.continuations
    }
    return [
        frontier.gain_move
        for frontier in own_frontiers
        if (
            frontier.kind is ThreatKind.QUIET
            and len(frontier.continuations) >= minimum_continuations
            and frontier.gain_move in opponent_continuations
        )
    ]


def dual_frontier_gain_bridges(
    *,
    own_frontiers: Iterable[ThreatFrontier],
    opponent_frontiers: Iterable[ThreatFrontier],
    minimum_own_rank: int,
    minimum_opponent_rank: int,
    minimum_own_continuations: int,
    minimum_opponent_continuations: int,
    limit: int,
) -> list[Move]:
    """Return bounded quiet gain points shared by both threat networks.

    A move can be strategically defensive even when neither side has a direct
    three or four there.  If the same quiet gain expands several of our future
    continuations while also occupying an opponent gain point, keep it as a
    candidate-only counter-bridge.  This never upgrades the move to proof.
    """
    if limit <= 0:
        return []

    opponent_by_gain = {
        frontier.gain_move: frontier
        for frontier in opponent_frontiers
        if (
            frontier.kind is ThreatKind.QUIET
            and len(frontier.continuations)
            >= minimum_opponent_continuations
            and max(frontier.continuation_ranks, default=0)
            >= minimum_opponent_rank
        )
    }
    ranked: list[tuple[int, int, int, int, Move]] = []
    for order, frontier in enumerate(own_frontiers):
        opponent = opponent_by_gain.get(frontier.gain_move)
        if (
            opponent is None
            or frontier.kind is not ThreatKind.QUIET
            or len(frontier.continuations) < minimum_own_continuations
            or max(frontier.continuation_ranks, default=0)
            < minimum_own_rank
        ):
            continue
        ranked.append(
            (
                max(frontier.continuation_ranks, default=0)
                + max(opponent.continuation_ranks, default=0),
                len(frontier.continuations)
                + len(opponent.continuations),
                sum(frontier.continuation_ranks),
                -order,
                frontier.gain_move,
            )
        )
    ranked.sort(reverse=True)
    return [item[-1] for item in ranked[:limit]]


def with_sources(
    groups: Iterable[tuple[CandidateSource, Iterable[Move]]],
) -> tuple[CandidateEntry, ...]:
    """Create deterministic provenance metadata without changing ordering."""
    ordered: list[Move] = []
    sources: dict[Move, set[CandidateSource]] = {}
    for source, moves in groups:
        for move in moves:
            if move not in sources:
                ordered.append(move)
                sources[move] = set()
            sources[move].add(source)
    return tuple(
        CandidateEntry(move, frozenset(sources[move]))
        for move in ordered
    )


def all_near_forced_loss(
    result: RootResult,
    candidates: list[Move],
) -> bool:
    """Return whether every searched root move has a mate-like loss."""
    unique_candidates = tuple(dict.fromkeys(candidates))
    scores = dict(result.ranked_moves)
    return bool(unique_candidates) and all(
        move in scores
        and scores[move] <= -MATE_SCORE + 10_000
        for move in unique_candidates
    )
