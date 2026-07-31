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


class CandidateSource(str, Enum):
    ORDINARY = "ordinary"
    OWN_FORCING = "own_forcing"
    ACTIVE_COUNTERATTACK = "active_counterattack"
    MANDATORY_DEFENSE = "mandatory_defense"
    THREAT_FRONTIER = "threat_frontier"
    VCF_INTERCEPT = "vcf_intercept"
    ROOT_EXPANSION = "root_expansion"


class RootCandidateMode(str, Enum):
    MERGED_FORCING = "merged_forcing"
    OWN_FORCING = "own_forcing"
    MANDATORY_DEFENSE = "mandatory_defense"
    FRONTIER_DEFENSE = "frontier_defense"
    ORDINARY = "ordinary"


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


def merge_unique(*groups: Iterable[Move]) -> list[Move]:
    return list(
        dict.fromkeys(
            move
            for group in groups
            for move in group
        )
    )


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

    ordered = merge_unique(*ordered_groups)
    required = merge_unique(*required_groups)
    if len(required) >= limit:
        return required[:limit]

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
) -> list[Move]:
    """Build a complete but capped frontier-defense root set."""
    frontier = tuple(frontier_moves)
    ordinary = tuple(ordinary_moves)
    counterattacks = tuple(counterattack_moves)
    return merge_with_required(
        ordered_groups=(frontier, ordinary, counterattacks),
        required_groups=(frontier, counterattacks),
        limit=limit,
    )


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
