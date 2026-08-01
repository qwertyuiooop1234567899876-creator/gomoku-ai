from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

from engine.board import (
    BLACK,
    DIRECTIONS,
    EMPTY,
    WHITE,
    Board,
)
from engine.evaluator import (
    ThreatProfile,
    analyze_move_threats,
    evaluate_move,
    find_winning_moves,
    other_side,
)
from engine.native_core import native_core

Move = tuple[int, int]
Direction = tuple[int, int]
StopPredicate = Callable[[], bool]


class ThreatKind(str, Enum):
    """A move's tactical shape; it is not itself a proof result."""

    FIVE = "five"
    DOUBLE_FOUR = "double_four"
    OPEN_FOUR = "open_four"
    FOUR_THREE = "four_three"
    DOUBLE_THREE = "double_three"
    FOUR = "four"
    OPEN_THREE = "open_three"
    QUIET = "quiet"


@dataclass(frozen=True, slots=True)
class ThreatContinuation:
    """One concrete move that keeps an exact forcing threat alive.

    A continuation is exact only when it wins immediately, or leaves at
    least two immediate winning points while the defender has no faster
    immediate win.
    """

    move: Move
    immediate_win: bool
    winning_points: tuple[Move, ...] = ()
    source_lines: tuple[Direction, ...] = ()

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            self.move,
            self.immediate_win,
            self.winning_points,
            self.source_lines,
        )


@dataclass(frozen=True, slots=True)
class DefenseRefutation:
    """A concrete witness showing why one defender reply still loses."""

    defense_move: Move
    continuation_move: Move
    continuation_is_immediate: bool
    winning_points: tuple[Move, ...] = ()

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            self.defense_move,
            self.continuation_move,
            self.continuation_is_immediate,
            self.winning_points,
        )


@dataclass(frozen=True, slots=True)
class DefenseSet:
    """Exhaustive classification of the replies to one gain move.

    ``required_defenses`` contains replies for which no exact continuation
    remains. ``counter_wins`` end the game immediately for the defender.
    Every other classified legal reply is refuted by at least one concrete
    continuation and therefore does not need to become an AND child.
    """

    required_defenses: tuple[Move, ...] = ()
    counter_wins: tuple[Move, ...] = ()
    refutations: tuple[DefenseRefutation, ...] = ()
    unclassified_replies: tuple[Move, ...] = ()
    legal_reply_count: int = 0
    refuted_reply_count: int = 0
    coverage_complete: bool = False
    analysis_completed: bool = True

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            self.required_defenses,
            self.counter_wins,
            tuple(item.signature for item in self.refutations),
            self.unclassified_replies,
            self.legal_reply_count,
            self.refuted_reply_count,
            self.coverage_complete,
            self.analysis_completed,
        )


@dataclass(frozen=True, slots=True)
class Threat:
    """A simulated gain move with exact continuations and defense coverage.

    ``coverage_complete`` is never inferred from ``ThreatProfile``. It is
    true only after every legal defender reply has been classified by real
    board simulation.
    """

    gain_move: Move
    kind: ThreatKind
    attacker: int
    winning_continuations: tuple[Move, ...] = ()
    required_defenses: tuple[Move, ...] = ()
    counter_wins: tuple[Move, ...] = ()
    rest_squares: tuple[Move, ...] = ()
    source_lines: tuple[Direction, ...] = ()
    coverage_complete: bool = False
    profile: ThreatProfile | None = None
    continuations: tuple[ThreatContinuation, ...] = ()
    frontier_continuations: tuple[Move, ...] = ()
    defense_refutations: tuple[DefenseRefutation, ...] = ()
    unclassified_defenses: tuple[Move, ...] = ()
    legal_reply_count: int = 0
    refuted_reply_count: int = 0
    analysis_completed: bool = True

    @property
    def defense_set(self) -> DefenseSet:
        return DefenseSet(
            required_defenses=self.required_defenses,
            counter_wins=self.counter_wins,
            refutations=self.defense_refutations,
            unclassified_replies=self.unclassified_defenses,
            legal_reply_count=self.legal_reply_count,
            refuted_reply_count=self.refuted_reply_count,
            coverage_complete=self.coverage_complete,
            analysis_completed=self.analysis_completed,
        )

    @property
    def signature(self) -> tuple[object, ...]:
        """Return the proof-relevant part used by the proof TT key."""
        return (
            self.gain_move,
            self.kind.value,
            self.attacker,
            tuple(item.signature for item in self.continuations),
            self.frontier_continuations,
            self.required_defenses,
            self.counter_wins,
            self.rest_squares,
            self.source_lines,
            tuple(
                item.signature for item in self.defense_refutations
            ),
            self.unclassified_defenses,
            self.coverage_complete,
            self.analysis_completed,
        )


@dataclass(frozen=True, slots=True)
class ThreatBatch:
    """A bounded set of forcing candidates generated for one OR node."""

    threats: tuple[Threat, ...]
    coverage_complete: bool
    generation_completed: bool = True


@dataclass(frozen=True, slots=True)
class ThreatCandidate:
    """One profiled OR candidate that has not been described yet.

    Candidate profiling is intentionally cheaper than exact defense
    classification.  ProofSearch can therefore describe candidates lazily
    and stop as soon as one strict winning witness has been found.
    """

    move: Move
    profile: ThreatProfile
    kind: ThreatKind
    dependency_score: int = 0


@dataclass(frozen=True, slots=True)
class ThreatCandidateBatch:
    """A bounded, ordered candidate list without proof semantics."""

    candidates: tuple[ThreatCandidate, ...]
    coverage_complete: bool
    generation_completed: bool = True


@dataclass(frozen=True, slots=True)
class ThreatAnalyzerStats:
    candidate_batches: int = 0
    exact_descriptions: int = 0
    frontier_batches: int = 0
    frontier_descriptions: int = 0
    cache_queries: int = 0
    cache_hits: int = 0
    cache_stores: int = 0
    cache_skips: int = 0


@dataclass(frozen=True, slots=True)
class ThreatFrontier:
    """A gain move whose placement creates stronger future threats.

    Frontiers are candidate-generation evidence only.  In particular,
    ``coverage_complete`` is always false: finding several future threats
    does not classify every legal defender reply and must never be promoted
    directly to a proof result.
    """

    gain_move: Move
    kind: ThreatKind
    continuations: tuple[Move, ...]
    continuation_kinds: tuple[ThreatKind, ...]
    continuation_ranks: tuple[int, ...]
    coverage_complete: bool = False

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            self.gain_move,
            self.kind.value,
            self.continuations,
            tuple(kind.value for kind in self.continuation_kinds),
            self.continuation_ranks,
            self.coverage_complete,
        )


_THREAT_PRIORITY = {
    ThreatKind.FIVE: 100,
    ThreatKind.DOUBLE_FOUR: 95,
    ThreatKind.OPEN_FOUR: 90,
    ThreatKind.FOUR_THREE: 85,
    ThreatKind.DOUBLE_THREE: 80,
    ThreatKind.FOUR: 60,
    ThreatKind.OPEN_THREE: 40,
    ThreatKind.QUIET: 0,
}

_FORCING_KINDS = frozenset(
    {
        ThreatKind.FIVE,
        ThreatKind.DOUBLE_FOUR,
        ThreatKind.OPEN_FOUR,
        ThreatKind.FOUR_THREE,
        ThreatKind.DOUBLE_THREE,
        ThreatKind.FOUR,
        ThreatKind.OPEN_THREE,
    }
)


def classify_threat(profile: ThreatProfile) -> ThreatKind:
    """Map an evaluator profile to a candidate class, not a proof state."""
    if profile.immediate_win:
        return ThreatKind.FIVE
    if profile.double_four:
        return ThreatKind.DOUBLE_FOUR
    if profile.open_four_directions >= 1:
        return ThreatKind.OPEN_FOUR
    if profile.four_three:
        return ThreatKind.FOUR_THREE
    if profile.double_three:
        return ThreatKind.DOUBLE_THREE
    if profile.four_directions >= 1:
        return ThreatKind.FOUR
    if profile.open_three_directions >= 1:
        return ThreatKind.OPEN_THREE
    return ThreatKind.QUIET


class ThreatAnalyzer:
    """Build conservative threat descriptions by simulating real moves."""

    def __init__(
        self,
        *,
        candidate_limit: int = 24,
        frontier_scan_limit: int | None = None,
        cache_enabled: bool = True,
        cache_max_entries: int = 50_000,
        profile_cache: (
            dict[tuple[int, int, int, int], ThreatProfile] | None
        ) = None,
        exact_cache: (
            dict[tuple[int, int, int, int], Threat] | None
        ) = None,
        candidate_cache: (
            dict[
                tuple[int, int, int, int | None],
                ThreatCandidateBatch,
            ]
            | None
        ) = None,
    ) -> None:
        if candidate_limit < 1:
            raise ValueError("candidate_limit 必须大于 0。")
        if frontier_scan_limit is not None and frontier_scan_limit < 1:
            raise ValueError("frontier_scan_limit 必须大于 0。")
        if cache_max_entries < 1:
            raise ValueError("cache_max_entries 必须大于 0。")
        self.candidate_limit = candidate_limit
        self.frontier_scan_limit = frontier_scan_limit
        self.cache_enabled = cache_enabled
        self.cache_max_entries = cache_max_entries
        self._profile_cache = (
            {} if profile_cache is None else profile_cache
        )
        self._exact_cache = (
            {} if exact_cache is None else exact_cache
        )
        self._candidate_cache = (
            {} if candidate_cache is None else candidate_cache
        )
        self._candidate_batches = 0
        self._exact_descriptions = 0
        self._frontier_batches = 0
        self._frontier_descriptions = 0
        self._cache_queries = 0
        self._cache_hits = 0
        self._cache_stores = 0
        self._cache_skips = 0

    def stats(self) -> ThreatAnalyzerStats:
        return ThreatAnalyzerStats(
            candidate_batches=self._candidate_batches,
            exact_descriptions=self._exact_descriptions,
            frontier_batches=self._frontier_batches,
            frontier_descriptions=self._frontier_descriptions,
            cache_queries=self._cache_queries,
            cache_hits=self._cache_hits,
            cache_stores=self._cache_stores,
            cache_skips=self._cache_skips,
        )

    def _cache_get(self, cache, key):  # type: ignore[no-untyped-def]
        self._cache_queries += 1
        if not self.cache_enabled:
            return None
        cached = cache.get(key)
        if cached is not None:
            self._cache_hits += 1
        return cached

    def _cache_store(
        self,
        cache,  # type: ignore[no-untyped-def]
        key,  # type: ignore[no-untyped-def]
        value,  # type: ignore[no-untyped-def]
    ) -> None:
        if not self.cache_enabled:
            return
        if key not in cache and len(cache) >= self.cache_max_entries:
            del cache[next(iter(cache))]
        cache[key] = value
        self._cache_stores += 1

    def analyze_profile(
        self,
        board: Board,
        move: Move,
        player: int,
    ) -> ThreatProfile:
        """Return one exact move profile, reusing only the same position."""
        if player not in (BLACK, WHITE):
            raise ValueError("玩家只能是 BLACK 或 WHITE。")
        if not board.is_empty(*move):
            raise ValueError("只能分析空位置。")

        key = (
            board.zobrist_hash,
            move[0],
            move[1],
            player,
        )
        cached = self._cache_get(self._profile_cache, key)
        if cached is not None:
            return cached

        profile = analyze_move_threats(board, *move, player)
        self._cache_store(self._profile_cache, key, profile)
        return profile

    def analyze_profiles(
        self,
        board: Board,
        moves: Iterable[Move],
        player: int,
    ) -> dict[Move, ThreatProfile]:
        """Analyze a move batch while preserving the per-position cache."""
        ordered = tuple(moves)
        results: dict[Move, ThreatProfile] = {}
        missing: list[Move] = []
        for move in ordered:
            key = (board.zobrist_hash, move[0], move[1], player)
            cached = self._cache_get(self._profile_cache, key)
            if cached is None:
                missing.append(move)
            else:
                results[move] = cached

        native_profiles = None
        if missing:
            try:
                native_profiles = native_core.analyze_moves(
                    board,
                    missing,
                    player,
                )
            except RuntimeError:
                native_profiles = None
        if native_profiles is None:
            profiles = [
                analyze_move_threats(board, *move, player)
                for move in missing
            ]
        else:
            profiles = [
                ThreatProfile(
                    immediate_win=profile.immediate_win,
                    open_four_directions=profile.open_four_directions,
                    four_directions=profile.four_directions,
                    open_three_directions=profile.open_three_directions,
                    winning_moves=profile.winning_moves,
                )
                for profile in native_profiles
            ]
        for move, profile in zip(missing, profiles, strict=True):
            key = (board.zobrist_hash, move[0], move[1], player)
            self._cache_store(self._profile_cache, key, profile)
            results[move] = profile
        return {move: results[move] for move in ordered}

    def describe_move(
        self,
        board: Board,
        move: Move,
        player: int,
        *,
        stop_requested: StopPredicate | None = None,
    ) -> Threat:
        """Describe one legal move without changing the caller's board."""
        if player not in (BLACK, WHITE):
            raise ValueError("玩家只能是 BLACK 或 WHITE。")
        if not board.is_empty(*move):
            raise ValueError("只能分析空位置。")

        profile = self.analyze_profile(board, move, player)
        return self._describe_cached_profiled_move(
            board,
            move,
            player,
            profile,
            stop_requested=stop_requested,
        )

    def _describe_cached_profiled_move(
        self,
        board: Board,
        move: Move,
        player: int,
        profile: ThreatProfile,
        *,
        stop_requested: StopPredicate | None,
    ) -> Threat:
        key = (
            board.zobrist_hash,
            move[0],
            move[1],
            player,
        )
        if not self._stopped(stop_requested):
            cached = self._cache_get(self._exact_cache, key)
            if cached is not None:
                return cached

        threat = self._describe_profiled_move(
            board,
            move,
            player,
            profile,
            stop_requested=stop_requested,
        )
        if threat.analysis_completed:
            self._cache_store(self._exact_cache, key, threat)
        else:
            self._cache_skips += 1
        return threat

    def _describe_profiled_move(
        self,
        board: Board,
        move: Move,
        player: int,
        profile: ThreatProfile,
        *,
        stop_requested: StopPredicate | None,
    ) -> Threat:
        self._exact_descriptions += 1
        kind = classify_threat(profile)
        defender = other_side(player)

        board.place(*move, player)
        try:
            if board.check_win(*move):
                return Threat(
                    gain_move=move,
                    kind=ThreatKind.FIVE,
                    attacker=player,
                    coverage_complete=True,
                    profile=profile,
                )

            counter_wins = tuple(
                sorted(find_winning_moves(board, defender))
            )
            continuations, completed = self._exact_continuations(
                board,
                gain_move=move,
                attacker=player,
                stop_requested=stop_requested,
            )
            if not completed:
                return Threat(
                    gain_move=move,
                    kind=kind,
                    attacker=player,
                    counter_wins=counter_wins,
                    profile=profile,
                    continuations=continuations,
                    analysis_completed=False,
                )

            defense_set = self._classify_defenses(
                board,
                attacker=player,
                continuations=continuations,
                counter_wins=counter_wins,
                stop_requested=stop_requested,
            )
            source_lines = tuple(
                sorted(
                    {
                        direction
                        for continuation in continuations
                        for direction in continuation.source_lines
                    }
                )
            )
            rest_squares = tuple(
                sorted(
                    {
                        rest
                        for continuation in continuations
                        for rest in continuation.winning_points
                    }
                )
            )
            return Threat(
                gain_move=move,
                kind=kind,
                attacker=player,
                winning_continuations=tuple(
                    continuation.move
                    for continuation in continuations
                ),
                required_defenses=defense_set.required_defenses,
                counter_wins=defense_set.counter_wins,
                rest_squares=rest_squares,
                source_lines=source_lines,
                coverage_complete=defense_set.coverage_complete,
                profile=profile,
                continuations=continuations,
                defense_refutations=defense_set.refutations,
                unclassified_defenses=(
                    defense_set.unclassified_replies
                ),
                legal_reply_count=defense_set.legal_reply_count,
                refuted_reply_count=defense_set.refuted_reply_count,
                analysis_completed=(
                    defense_set.analysis_completed
                ),
            )
        finally:
            board.undo()

    def _exact_continuations(
        self,
        board: Board,
        *,
        gain_move: Move,
        attacker: int,
        stop_requested: StopPredicate | None,
    ) -> tuple[tuple[ThreatContinuation, ...], bool]:
        immediate = tuple(sorted(find_winning_moves(board, attacker)))
        if immediate:
            continuations: list[ThreatContinuation] = []
            for move in immediate:
                if self._stopped(stop_requested):
                    return tuple(continuations), False
                continuation = self._describe_continuation(
                    board,
                    gain_move=gain_move,
                    move=move,
                    attacker=attacker,
                    require_gain_dependency=False,
                )
                if continuation is not None:
                    continuations.append(continuation)
            return tuple(continuations), True

        continuations = []
        for move in self._line_candidates(board, gain_move):
            if self._stopped(stop_requested):
                return tuple(continuations), False
            continuation = self._describe_continuation(
                board,
                gain_move=gain_move,
                move=move,
                attacker=attacker,
                require_gain_dependency=True,
            )
            if continuation is not None:
                continuations.append(continuation)

        continuations.sort(key=lambda item: item.move)
        return tuple(continuations), True

    def _describe_continuation(
        self,
        board: Board,
        *,
        gain_move: Move,
        move: Move,
        attacker: int,
        require_gain_dependency: bool,
    ) -> ThreatContinuation | None:
        if not board.is_empty(*move):
            return None

        defender = other_side(attacker)
        board.place(*move, attacker)
        try:
            if board.check_win(*move):
                source_lines = self._winning_directions(
                    board,
                    move,
                    attacker,
                    anchor=(
                        gain_move if require_gain_dependency else None
                    ),
                )
                if require_gain_dependency and not source_lines:
                    return None
                return ThreatContinuation(
                    move=move,
                    immediate_win=True,
                    winning_points=(move,),
                    source_lines=source_lines,
                )

            # A double winning point is not a proof if the defender can end
            # the game first on the very next move.
            if find_winning_moves(board, defender):
                return None

            winning_points = tuple(
                sorted(find_winning_moves(board, attacker))
            )
            if len(winning_points) < 2:
                return None

            source_lines = tuple(
                sorted(
                    {
                        direction
                        for winning_point in winning_points
                        for direction in (
                            self._winning_directions_for_empty(
                                board,
                                winning_point,
                                attacker,
                                anchor=gain_move,
                            )
                        )
                    }
                )
            )
            if require_gain_dependency and not source_lines:
                return None
            return ThreatContinuation(
                move=move,
                immediate_win=False,
                winning_points=winning_points,
                source_lines=source_lines,
            )
        finally:
            board.undo()

    def _classify_defenses(
        self,
        board: Board,
        *,
        attacker: int,
        continuations: tuple[ThreatContinuation, ...],
        counter_wins: tuple[Move, ...],
        stop_requested: StopPredicate | None,
    ) -> DefenseSet:
        legal_replies = tuple(board.get_legal_moves())
        if not continuations:
            return DefenseSet(
                counter_wins=counter_wins,
                unclassified_replies=legal_replies,
                legal_reply_count=len(legal_replies),
                coverage_complete=False,
            )

        defender = other_side(attacker)
        counter_win_set = set(counter_wins)
        continuation_blocks = tuple(
            (
                continuation,
                frozenset(
                    (
                        continuation.move,
                        *continuation.winning_points,
                    )
                ),
            )
            for continuation in continuations
        )
        required: list[Move] = []
        refutations: list[DefenseRefutation] = []
        try:
            counter_support = native_core.counter_support_mask(
                board,
                legal_replies,
                defender,
                minimum=3,
            )
        except RuntimeError:
            counter_support = None

        for index, move in enumerate(legal_replies):
            if self._stopped(stop_requested):
                return DefenseSet(
                    required_defenses=tuple(sorted(required)),
                    counter_wins=counter_wins,
                    refutations=tuple(refutations),
                    unclassified_replies=tuple(
                        sorted(legal_replies[index:])
                    ),
                    legal_reply_count=len(legal_replies),
                    refuted_reply_count=len(refutations),
                    coverage_complete=False,
                    analysis_completed=False,
                )
            if move in counter_win_set:
                continue

            direct_refutation = None
            could_create_counter = (
                self._could_create_immediate_counter(
                    board,
                    move,
                    defender,
                )
                if counter_support is None
                else counter_support[index]
            )
            if (
                not counter_win_set
                and not could_create_counter
            ):
                direct_refutation = next(
                    (
                        DefenseRefutation(
                            defense_move=move,
                            continuation_move=continuation.move,
                            continuation_is_immediate=(
                                continuation.immediate_win
                            ),
                            winning_points=(
                                continuation.winning_points
                            ),
                        )
                        for continuation, blocked_moves in continuation_blocks
                        if move not in blocked_moves
                    ),
                    None,
                )
            if direct_refutation is not None:
                refutations.append(direct_refutation)
                continue

            board.place(*move, defender)
            try:
                if board.check_win(*move):
                    continue
                refutation = next(
                    (
                        witness
                        for continuation in continuations
                        if (
                            witness
                            := self._continuation_refutation(
                                board,
                                defense_move=move,
                                continuation=continuation,
                                attacker=attacker,
                            )
                        )
                        is not None
                    ),
                    None,
                )
            finally:
                board.undo()

            if refutation is not None:
                refutations.append(refutation)
            else:
                required.append(move)

        return DefenseSet(
            required_defenses=tuple(sorted(required)),
            counter_wins=counter_wins,
            refutations=tuple(refutations),
            legal_reply_count=len(legal_replies),
            refuted_reply_count=len(refutations),
            coverage_complete=True,
        )

    @staticmethod
    def _could_create_immediate_counter(
        board: Board,
        move: Move,
        defender: int,
    ) -> bool:
        """Cheap necessary test for a newly-created one-move win.

        With no pre-existing immediate win, a defender reply needs at least
        three friendly stones on one length-nine local line before it can
        leave a winning point for the following turn.  False positives are
        allowed and fall back to exact simulation; false negatives would be
        unsafe.
        """
        for row_step, column_step in DIRECTIONS:
            friendly = 0
            for offset in range(-4, 5):
                if offset == 0:
                    continue
                row = move[0] + offset * row_step
                column = move[1] + offset * column_step
                if (
                    board.is_inside(row, column)
                    and board.grid[row][column] == defender
                ):
                    friendly += 1
            if friendly >= 3:
                return True
        return False

    @staticmethod
    def _continuation_refutation(
        board: Board,
        *,
        defense_move: Move,
        continuation: ThreatContinuation,
        attacker: int,
    ) -> DefenseRefutation | None:
        move = continuation.move
        if not board.is_empty(*move):
            return None

        defender = other_side(attacker)
        board.place(*move, attacker)
        try:
            if board.check_win(*move):
                return DefenseRefutation(
                    defense_move=defense_move,
                    continuation_move=move,
                    continuation_is_immediate=True,
                    winning_points=(move,),
                )
            if find_winning_moves(board, defender):
                return None
            winning_points = tuple(
                sorted(find_winning_moves(board, attacker))
            )
            if len(winning_points) < 2:
                return None
            return DefenseRefutation(
                defense_move=defense_move,
                continuation_move=move,
                continuation_is_immediate=False,
                winning_points=winning_points,
            )
        finally:
            board.undo()

    def generate_attack_threats(
        self,
        board: Board,
        player: int,
        *,
        stop_requested: StopPredicate | None = None,
    ) -> ThreatBatch:
        """Generate bounded forcing OR candidates without global coverage.

        Exact defense calculation is only performed for the strongest
        profiled candidates. Exhausting this bounded batch may prove a win
        through a witness, but it cannot prove that the attacker has no win.
        """
        if player not in (BLACK, WHITE):
            raise ValueError("玩家只能是 BLACK 或 WHITE。")

        candidates = self.generate_attack_candidates(
            board,
            player,
            stop_requested=stop_requested,
        )
        if not candidates.generation_completed:
            return ThreatBatch(
                threats=(),
                coverage_complete=False,
                generation_completed=False,
            )

        threats: list[Threat] = []
        for candidate in candidates.candidates:
            if self._stopped(stop_requested):
                return ThreatBatch(
                    threats=tuple(self._sort_threats(board, threats)),
                    coverage_complete=False,
                    generation_completed=False,
                )
            threat = self._describe_cached_profiled_move(
                board,
                candidate.move,
                player,
                candidate.profile,
                stop_requested=stop_requested,
            )
            if not threat.analysis_completed:
                return ThreatBatch(
                    threats=tuple(self._sort_threats(board, threats)),
                    coverage_complete=False,
                    generation_completed=False,
                )
            threats.append(threat)

        ordered = self._sort_threats(board, threats)
        return ThreatBatch(
            threats=tuple(ordered),
            coverage_complete=False,
            generation_completed=True,
        )

    def generate_attack_candidates(
        self,
        board: Board,
        player: int,
        *,
        stop_requested: StopPredicate | None = None,
    ) -> ThreatCandidateBatch:
        """Profile forcing candidates without classifying their defenses."""
        if player not in (BLACK, WHITE):
            raise ValueError("玩家只能是 BLACK 或 WHITE。")
        self._candidate_batches += 1
        cache_key = (
            board.zobrist_hash,
            player,
            self.candidate_limit,
            self.frontier_scan_limit,
        )
        if not self._stopped(stop_requested):
            cached = self._cache_get(
                self._candidate_cache,
                cache_key,
            )
            if cached is not None:
                return cached

        immediate = set(find_winning_moves(board, player))
        ordered_moves = self._relevant_moves(board, immediate)
        profiled: list[ThreatCandidate] = []
        supported_moves: list[Move] = []

        for move in ordered_moves:
            if self._stopped(stop_requested):
                self._cache_skips += 1
                return ThreatCandidateBatch(
                    candidates=(),
                    coverage_complete=False,
                    generation_completed=False,
                )
            if (
                move not in immediate
                and not self._has_local_stone_support(
                    board,
                    move,
                    player,
                    minimum=2,
                )
            ):
                continue
            supported_moves.append(move)

        profiles = self.analyze_profiles(board, supported_moves, player)
        for move in supported_moves:
            if self._stopped(stop_requested):
                self._cache_skips += 1
                return ThreatCandidateBatch(
                    candidates=(),
                    coverage_complete=False,
                    generation_completed=False,
                )
            profile = profiles[move]
            kind = classify_threat(profile)
            if kind in _FORCING_KINDS:
                dependency_score = self._future_threat_score(
                    board,
                    move,
                    player,
                )
                profiled.append(
                    ThreatCandidate(
                        move=move,
                        profile=profile,
                        kind=kind,
                        dependency_score=dependency_score,
                    )
                )

        profiled.sort(
            key=lambda item: (
                -(
                    _THREAT_PRIORITY[item.kind]
                    + item.dependency_score
                ),
                -item.dependency_score,
                -_THREAT_PRIORITY[item.kind],
                -len(item.profile.winning_moves),
                (
                    (item.move[0] - (board.size - 1) / 2) ** 2
                    + (item.move[1] - (board.size - 1) / 2) ** 2
                ),
                item.move,
            )
        )
        result = ThreatCandidateBatch(
            candidates=tuple(profiled[: self.candidate_limit]),
            coverage_complete=False,
            generation_completed=True,
        )
        self._cache_store(
            self._candidate_cache,
            cache_key,
            result,
        )
        return result

    @staticmethod
    def _has_local_stone_support(
        board: Board,
        move: Move,
        player: int,
        *,
        minimum: int,
    ) -> bool:
        """Return whether a forcing pattern is locally possible at all."""
        for row_step, column_step in DIRECTIONS:
            friendly = 0
            for offset in range(-4, 5):
                if offset == 0:
                    continue
                row = move[0] + offset * row_step
                column = move[1] + offset * column_step
                if (
                    board.is_inside(row, column)
                    and board.grid[row][column] == player
                ):
                    friendly += 1
            if friendly >= minimum:
                return True
        return False

    def describe_candidate(
        self,
        board: Board,
        candidate: ThreatCandidate,
        player: int,
        *,
        stop_requested: StopPredicate | None = None,
    ) -> Threat:
        """Build the exact description for one already-profiled candidate."""
        if player not in (BLACK, WHITE):
            raise ValueError("玩家只能是 BLACK 或 WHITE。")
        if not board.is_empty(*candidate.move):
            raise ValueError("只能分析空位置。")
        return self._describe_cached_profiled_move(
            board,
            candidate.move,
            player,
            candidate.profile,
            stop_requested=stop_requested,
        )

    def describe_frontier(
        self,
        board: Board,
        frontier: ThreatFrontier,
        player: int,
    ) -> Threat:
        """Convert a quiet frontier into an exhaustive defender obligation.

        A frontier is still only candidate evidence.  It becomes safe input
        to an AND node by retaining *every* legal defender reply, rather than
        pretending that the future continuation list is already a proof.
        ProofSearch must recursively prove all of those replies.
        """
        if player not in (BLACK, WHITE):
            raise ValueError("玩家只能是 BLACK 或 WHITE。")
        if not board.is_empty(*frontier.gain_move):
            raise ValueError("只能分析空位置。")
        self._frontier_descriptions += 1

        board.place(*frontier.gain_move, player)
        try:
            defenses = tuple(sorted(board.get_legal_moves()))
            return Threat(
                gain_move=frontier.gain_move,
                kind=frontier.kind,
                attacker=player,
                required_defenses=defenses,
                frontier_continuations=frontier.continuations,
                coverage_complete=True,
                legal_reply_count=len(defenses),
            )
        finally:
            board.undo()

    def generate_attack_frontiers(
        self,
        board: Board,
        player: int,
        *,
        frontier_limit: int = 12,
        continuation_limit: int = 12,
        stop_requested: StopPredicate | None = None,
    ) -> tuple[ThreatFrontier, ...]:
        """Find quiet or single-threat moves that create future threats.

        A future continuation is retained only when placing the gain move
        increases that continuation's tactical rank.  This dependency check
        prevents an unrelated pre-existing threat elsewhere on the board
        from making an arbitrary quiet move look like a frontier.
        """
        if player not in (BLACK, WHITE):
            raise ValueError("玩家只能是 BLACK 或 WHITE。")
        if frontier_limit < 1:
            raise ValueError("frontier_limit 必须大于 0。")
        if continuation_limit < 1:
            raise ValueError("continuation_limit 必须大于 0。")
        self._frontier_batches += 1

        relevant_moves = self._relevant_moves(board, ())
        if (
            self.frontier_scan_limit is not None
            and len(relevant_moves) > self.frontier_scan_limit
        ):
            center = (board.size - 1) / 2
            relevant_moves.sort(
                key=lambda move: (
                    -evaluate_move(board, *move, player),
                    (
                        (move[0] - center) ** 2
                        + (move[1] - center) ** 2
                    ),
                    move,
                )
            )
            relevant_moves = relevant_moves[
                : self.frontier_scan_limit
            ]

        frontiers: list[ThreatFrontier] = []
        for move in relevant_moves:
            if self._stopped(stop_requested):
                break

            base_profile = self.analyze_profile(board, move, player)
            base_kind = classify_threat(base_profile)
            if base_profile.forced_win:
                continue

            before_ranks: dict[Move, int] = {}
            line_candidates = self._line_candidates(board, move)
            for continuation in line_candidates:
                if self._stopped(stop_requested):
                    break
                before_ranks[continuation] = self.analyze_profile(
                    board,
                    continuation,
                    player,
                ).tactical_rank
            if self._stopped(stop_requested):
                break

            board.place(*move, player)
            try:
                improved: list[
                    tuple[Move, ThreatKind, int]
                ] = []
                for continuation in line_candidates:
                    if self._stopped(stop_requested):
                        break
                    if not board.is_empty(*continuation):
                        continue
                    profile = self.analyze_profile(
                        board,
                        continuation,
                        player,
                    )
                    if (
                        profile.tactical_rank >= 40
                        and profile.tactical_rank
                        > before_ranks.get(continuation, 0)
                    ):
                        improved.append(
                            (
                                continuation,
                                classify_threat(profile),
                                profile.tactical_rank,
                            )
                        )
            finally:
                board.undo()

            if self._stopped(stop_requested):
                break
            if len(improved) < 2:
                continue

            improved.sort(
                key=lambda item: (
                    -item[2],
                    item[0],
                )
            )
            improved = improved[:continuation_limit]
            frontiers.append(
                ThreatFrontier(
                    gain_move=move,
                    kind=base_kind,
                    continuations=tuple(
                        item[0] for item in improved
                    ),
                    continuation_kinds=tuple(
                        item[1] for item in improved
                    ),
                    continuation_ranks=tuple(
                        item[2] for item in improved
                    ),
                )
            )

        center = (board.size - 1) / 2
        frontiers.sort(
            key=lambda frontier: (
                -max(frontier.continuation_ranks),
                -sum(frontier.continuation_ranks),
                -len(frontier.continuations),
                -_THREAT_PRIORITY[frontier.kind],
                (
                    (frontier.gain_move[0] - center) ** 2
                    + (frontier.gain_move[1] - center) ** 2
                ),
                frontier.gain_move,
            )
        )
        return tuple(frontiers[:frontier_limit])

    @staticmethod
    def _line_candidates(
        board: Board,
        anchor: Move,
    ) -> tuple[Move, ...]:
        candidates: set[Move] = set()
        for row_step, column_step in DIRECTIONS:
            for offset in range(-4, 5):
                if offset == 0:
                    continue
                move = (
                    anchor[0] + offset * row_step,
                    anchor[1] + offset * column_step,
                )
                if board.is_empty(*move):
                    candidates.add(move)
        return tuple(sorted(candidates))

    @staticmethod
    def _winning_directions(
        board: Board,
        move: Move,
        player: int,
        *,
        anchor: Move | None,
    ) -> tuple[Direction, ...]:
        directions: list[Direction] = []
        for direction in DIRECTIONS:
            segment = ThreatAnalyzer._winning_segment(
                board,
                move,
                player,
                direction,
            )
            if len(segment) >= 5 and (
                anchor is None or anchor in segment
            ):
                directions.append(direction)
        return tuple(directions)

    @staticmethod
    def _winning_directions_for_empty(
        board: Board,
        move: Move,
        player: int,
        *,
        anchor: Move | None,
    ) -> tuple[Direction, ...]:
        if not board.is_empty(*move):
            return ()
        board.place(*move, player)
        try:
            return ThreatAnalyzer._winning_directions(
                board,
                move,
                player,
                anchor=anchor,
            )
        finally:
            board.undo()

    @staticmethod
    def _winning_segment(
        board: Board,
        move: Move,
        player: int,
        direction: Direction,
    ) -> set[Move]:
        row_step, column_step = direction
        segment = {move}

        for sign in (-1, 1):
            row = move[0] + sign * row_step
            column = move[1] + sign * column_step
            while (
                board.is_inside(row, column)
                and board.grid[row][column] == player
            ):
                segment.add((row, column))
                row += sign * row_step
                column += sign * column_step
        return segment

    @staticmethod
    def _relevant_moves(
        board: Board,
        mandatory: Iterable[Move],
    ) -> list[Move]:
        """Return empty points close enough to participate in a five-line."""
        relevant = set(mandatory)
        occupied: list[Move] = []

        for row in range(board.size):
            for column in range(board.size):
                if board.grid[row][column] != EMPTY:
                    occupied.append((row, column))

        if not occupied:
            center = board.size // 2
            return [(center, center)]

        for row, column in occupied:
            for row_offset in range(-4, 5):
                for column_offset in range(-4, 5):
                    candidate = (
                        row + row_offset,
                        column + column_offset,
                    )
                    if board.is_empty(*candidate):
                        relevant.add(candidate)

        return sorted(relevant)

    def _profile_sort_key(
        self,
        board: Board,
        move: Move,
        profile: ThreatProfile,
        kind: ThreatKind,
        *,
        player: int,
    ) -> tuple[float, ...]:
        center = (board.size - 1) / 2
        distance = (
            (move[0] - center) ** 2
            + (move[1] - center) ** 2
        )
        dependency_score = self._future_threat_score(
            board,
            move,
            player,
        )
        return (
            -_THREAT_PRIORITY[kind],
            -dependency_score,
            -len(profile.winning_moves),
            distance,
            move[0],
            move[1],
        )

    def _future_threat_score(
        self,
        board: Board,
        move: Move,
        player: int,
    ) -> int:
        """Rank equal tactical shapes by gain-dependent future pressure.

        The value is only a move-ordering heuristic.  It cannot create a
        proof state and does not change defense coverage.  Counting only
        tactical-rank improvements caused by the gain move avoids rewarding
        unrelated threats that were already present elsewhere.
        """
        line_candidates = self._line_candidates(board, move)
        before_profiles = self.analyze_profiles(
            board,
            line_candidates,
            player,
        )
        before_ranks = {
            continuation: profile.tactical_rank
            for continuation, profile in before_profiles.items()
        }

        board.place(*move, player)
        try:
            improvements: list[int] = []
            remaining = tuple(
                continuation
                for continuation in line_candidates
                if board.is_empty(*continuation)
            )
            after_profiles = self.analyze_profiles(
                board,
                remaining,
                player,
            )
            for continuation, profile in after_profiles.items():
                rank = profile.tactical_rank
                if rank > before_ranks.get(continuation, 0):
                    improvements.append(rank)
        finally:
            board.undo()

        return sum(sorted(improvements, reverse=True)[:8])

    @staticmethod
    def _sort_threats(
        board: Board,
        threats: Iterable[Threat],
    ) -> list[Threat]:
        center = (board.size - 1) / 2

        return sorted(
            threats,
            key=lambda threat: (
                -_THREAT_PRIORITY[threat.kind],
                -len(threat.winning_continuations),
                (threat.gain_move[0] - center) ** 2
                + (threat.gain_move[1] - center) ** 2,
                threat.gain_move,
            ),
        )

    @staticmethod
    def _stopped(
        stop_requested: StopPredicate | None,
    ) -> bool:
        return stop_requested is not None and stop_requested()
