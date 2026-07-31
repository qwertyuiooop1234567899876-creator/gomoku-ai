"""Root-result arbitration without search-tree ownership.

PVS, strict proof, and bounded threat risk use different semantics.  This
module is the single place that combines their already-computed results.  It
never searches the board and never writes to a transposition table.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from engine.ai import Move, ProofCandidateAnalysis
from engine.proof_search import ProofState
from engine.search_types import (
    HEURISTIC_SCORE_LIMIT,
    MATE_SCORE,
    RootResult,
    SearchConfig,
)

HeuristicScore = Callable[[Move], int]


def is_mate_like_score(score: int) -> bool:
    return abs(score) >= MATE_SCORE - 10_000


def has_strict_mate_evidence(
    score: int,
    proof_state: str | None,
) -> bool:
    """Return whether a mate-band root score has matching strict proof."""
    if not is_mate_like_score(score):
        return True
    if score > 0:
        return False
    return proof_state == ProofState.PROVEN_WIN.value


def quarantine_unproven_scores(
    result: RootResult,
    *,
    proof_states: dict[Move, str],
    heuristic_score: HeuristicScore,
    preserve_order: bool = False,
) -> tuple[RootResult, bool]:
    """Move selective PVS mate values onto one bounded comparison scale."""
    variations = {
        move: pv
        for move, _, pv in result.ranked_variations
    }
    original_priority = {
        move: len(result.ranked_moves) - index
        for index, (move, _) in enumerate(result.ranked_moves)
    }
    quarantine_required = any(
        not has_strict_mate_evidence(
            score,
            proof_states.get(move),
        )
        for move, score in result.ranked_moves
    )
    if not quarantine_required:
        return result, False

    revised: list[tuple[Move, int, tuple[Move, ...]]] = []
    for move, score in result.ranked_moves:
        has_strict_evidence = has_strict_mate_evidence(
            score,
            proof_states.get(move),
        )
        if not has_strict_evidence:
            if preserve_order:
                distance = MATE_SCORE - abs(score)
                score = (
                    HEURISTIC_SCORE_LIMIT - distance
                    if score > 0
                    else -HEURISTIC_SCORE_LIMIT + distance
                )
            else:
                score = heuristic_score(move)
        elif not preserve_order and not is_mate_like_score(score):
            # Once one move contaminated the result, compare every
            # non-terminal move on the same bounded scale.
            score = heuristic_score(move)
        revised.append(
            (
                move,
                score,
                variations.get(move, (move,)),
            )
        )

    revised.sort(
        key=lambda item: (
            item[1],
            original_priority.get(item[0], 0),
        ),
        reverse=True,
    )
    best_move, best_score, best_pv = revised[0]
    return (
        RootResult(
            move=best_move,
            score=best_score,
            principal_variation=best_pv,
            ranked_moves=tuple(
                (move, score)
                for move, score, _ in revised
            ),
            ranked_variations=tuple(revised),
        ),
        True,
    )


def apply_proof_tiebreak(
    config: SearchConfig,
    result: RootResult,
    proof_candidates: Sequence[ProofCandidateAnalysis],
) -> RootResult:
    """Apply strict safety first, then bounded risk near the PVS best."""
    root_scores = dict(result.ranked_moves)
    available = [
        candidate
        for candidate in proof_candidates
        if candidate.move in root_scores
    ]
    if not available:
        return result

    safety_priority = {
        ProofState.PROVEN_LOSS.value: 2,
        ProofState.UNKNOWN.value: 1,
        ProofState.PROVEN_WIN.value: 0,
    }
    best_safety = max(
        safety_priority.get(candidate.state, 1)
        for candidate in available
    )
    eligible = [
        candidate
        for candidate in available
        if safety_priority.get(candidate.state, 1) == best_safety
    ]

    best_pvs_score = max(
        root_scores[candidate.move]
        for candidate in eligible
    )
    pvs_band = [
        candidate
        for candidate in eligible
        if (
            best_pvs_score - root_scores[candidate.move]
            <= config.proof_risk_pvs_margin
        )
    ]

    if (
        best_safety == safety_priority[ProofState.UNKNOWN.value]
        and len(pvs_band) > 1
        and all(
            candidate.threat_risk is not None
            for candidate in pvs_band
        )
    ):
        lowest_risk = min(
            int(candidate.threat_risk)
            for candidate in pvs_band
            if candidate.threat_risk is not None
        )
        pvs_choice = max(
            pvs_band,
            key=lambda candidate: root_scores[candidate.move],
        )
        pvs_risk = int(pvs_choice.threat_risk)
        material_margin = max(100_000, lowest_risk // 20)
        if pvs_risk - lowest_risk >= material_margin:
            pvs_band = [
                candidate
                for candidate in pvs_band
                if candidate.threat_risk == lowest_risk
            ]

    chosen = max(
        pvs_band,
        key=lambda candidate: root_scores[candidate.move],
    )
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


def is_unknown_risk_override(
    pvs_result: RootResult,
    revised_result: RootResult,
    *,
    proof_states: dict[Move, str],
) -> bool:
    if revised_result.move == pvs_result.move:
        return False
    return (
        proof_states.get(pvs_result.move)
        == ProofState.UNKNOWN.value
        and proof_states.get(revised_result.move)
        == ProofState.UNKNOWN.value
    )
