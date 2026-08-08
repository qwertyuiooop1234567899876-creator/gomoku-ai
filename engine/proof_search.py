from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

from engine.board import BLACK, WHITE, Board
from engine.evaluator import find_winning_moves, other_side
from engine.native_core import native_core
from engine.threats import (
    DefenseRefutation,
    Move,
    Threat,
    ThreatAnalyzer,
    ThreatCandidate,
    ThreatFrontier,
    ThreatKind,
)
from engine.vcf import VCFSearch, validate_vcf_certificate

Clock = Callable[[], float]


class ProofState(str, Enum):
    """Strict result relative to one fixed attacker."""

    PROVEN_WIN = "proven_win"
    PROVEN_LOSS = "proven_loss"
    UNKNOWN = "unknown"


class _ProofCutoff(RuntimeError):
    """Internal control flow for a VCF oracle stopped by the proof budget."""


def combine_or_states(
    states: Iterable[ProofState],
    *,
    coverage_complete: bool,
) -> ProofState:
    """Combine attacker choices: one winning witness is sufficient."""
    results = tuple(states)
    if ProofState.PROVEN_WIN in results:
        return ProofState.PROVEN_WIN
    if (
        coverage_complete
        and all(state is ProofState.PROVEN_LOSS for state in results)
    ):
        return ProofState.PROVEN_LOSS
    return ProofState.UNKNOWN


def combine_and_states(
    states: Iterable[ProofState],
    *,
    coverage_complete: bool,
) -> ProofState:
    """Combine defender replies: every legal defense must still lose."""
    results = tuple(states)
    if ProofState.PROVEN_LOSS in results:
        return ProofState.PROVEN_LOSS
    if (
        coverage_complete
        and all(state is ProofState.PROVEN_WIN for state in results)
    ):
        return ProofState.PROVEN_WIN
    return ProofState.UNKNOWN


@dataclass(frozen=True, slots=True)
class ProofBudget:
    """Hard limits shared by every node of one proof request."""

    max_nodes: int = 10_000
    max_attacker_moves: int = 4
    max_quiet_frontiers: int = 0
    max_quiet_attacker_moves: int = 1
    vcf_max_attacker_moves: int = 5
    use_vcf_oracle: bool = False
    max_nodes_per_candidate: int | None = None
    max_seconds_per_candidate: float | None = None
    deadline: float | None = None

    def __post_init__(self) -> None:
        if self.max_nodes < 0:
            raise ValueError("max_nodes 不能小于 0。")
        if self.max_attacker_moves < 0:
            raise ValueError("max_attacker_moves 不能小于 0。")
        if self.max_quiet_frontiers < 0:
            raise ValueError("max_quiet_frontiers 不能小于 0。")
        if self.max_quiet_attacker_moves < 0:
            raise ValueError("max_quiet_attacker_moves 不能小于 0。")
        if self.vcf_max_attacker_moves < 0:
            raise ValueError("vcf_max_attacker_moves 不能小于 0。")
        if (
            self.max_nodes_per_candidate is not None
            and self.max_nodes_per_candidate < 1
        ):
            raise ValueError("max_nodes_per_candidate 必须大于 0。")
        if (
            self.max_seconds_per_candidate is not None
            and self.max_seconds_per_candidate <= 0
        ):
            raise ValueError("max_seconds_per_candidate 必须大于 0。")

    @classmethod
    def from_now(
        cls,
        seconds: float,
        *,
        max_nodes: int = 10_000,
        max_attacker_moves: int = 4,
        max_quiet_frontiers: int = 0,
        max_quiet_attacker_moves: int = 1,
        vcf_max_attacker_moves: int = 5,
        use_vcf_oracle: bool = False,
        max_nodes_per_candidate: int | None = None,
        max_seconds_per_candidate: float | None = None,
        clock: Clock = time.monotonic,
    ) -> ProofBudget:
        if seconds <= 0:
            raise ValueError("seconds 必须大于 0。")
        return cls(
            max_nodes=max_nodes,
            max_attacker_moves=max_attacker_moves,
            max_quiet_frontiers=max_quiet_frontiers,
            max_quiet_attacker_moves=max_quiet_attacker_moves,
            vcf_max_attacker_moves=vcf_max_attacker_moves,
            use_vcf_oracle=use_vcf_oracle,
            max_nodes_per_candidate=max_nodes_per_candidate,
            max_seconds_per_candidate=max_seconds_per_candidate,
            deadline=clock() + seconds,
        )


@dataclass(frozen=True, slots=True)
class ProofKey:
    board_hash: int
    attacker: int
    side_to_move: int
    obligation_signature: tuple[object, ...]
    remaining_attacker_moves: int
    remaining_quiet_moves: int


@dataclass(frozen=True, slots=True)
class ProofTTEntry:
    state: ProofState
    complete: bool
    best_move: Move | None
    principal_variation: tuple[Move, ...]
    required_defenses: tuple[Move, ...]
    linear_plan: tuple[Move, ...] | None
    generation: int


@dataclass(frozen=True, slots=True)
class ProofTableStats:
    queries: int = 0
    hits: int = 0
    compatible_hits: int = 0
    stores: int = 0
    skipped_stores: int = 0
    evictions: int = 0
    size: int = 0

    def delta(self, earlier: ProofTableStats) -> ProofTableStats:
        return ProofTableStats(
            queries=self.queries - earlier.queries,
            hits=self.hits - earlier.hits,
            compatible_hits=(
                self.compatible_hits - earlier.compatible_hits
            ),
            stores=self.stores - earlier.stores,
            skipped_stores=(
                self.skipped_stores - earlier.skipped_stores
            ),
            evictions=self.evictions - earlier.evictions,
            size=self.size,
        )


class ProofTable:
    """Proof-only transposition table; heuristic scores never enter it."""

    def __init__(self, *, max_entries: int = 50_000) -> None:
        if max_entries < 1:
            raise ValueError("max_entries 必须大于 0。")
        self.max_entries = max_entries
        self._entries: dict[ProofKey, ProofTTEntry] = {}
        self._depth_index: dict[
            tuple[int, int, int, tuple[object, ...]],
            set[ProofKey],
        ] = {}
        self._queries = 0
        self._hits = 0
        self._compatible_hits = 0
        self._stores = 0
        self._skipped_stores = 0
        self._evictions = 0

    def get(self, key: ProofKey) -> ProofTTEntry | None:
        self._queries += 1
        entry = self._entries.get(key)
        if entry is not None:
            self._hits += 1
            return entry

        indexed = self._depth_index.get(self._base_key(key), ())
        winning: list[tuple[ProofKey, ProofTTEntry]] = []
        losing: list[tuple[ProofKey, ProofTTEntry]] = []
        for stored_key in indexed:
            stored = self._entries.get(stored_key)
            if stored is None:
                continue
            if (
                stored.state is ProofState.PROVEN_WIN
                and stored_key.remaining_attacker_moves
                <= key.remaining_attacker_moves
                and stored_key.remaining_quiet_moves
                <= key.remaining_quiet_moves
            ):
                winning.append((stored_key, stored))
            elif (
                stored.state is ProofState.PROVEN_LOSS
                and stored_key.remaining_attacker_moves
                >= key.remaining_attacker_moves
                and stored_key.remaining_quiet_moves
                >= key.remaining_quiet_moves
            ):
                losing.append((stored_key, stored))

        compatible: ProofTTEntry | None = None
        if winning:
            compatible = min(
                winning,
                key=lambda item: (
                    item[0].remaining_attacker_moves,
                    item[0].remaining_quiet_moves,
                    -item[1].generation,
                ),
            )[1]
        elif losing:
            compatible = max(
                losing,
                key=lambda item: (
                    item[0].remaining_attacker_moves,
                    item[0].remaining_quiet_moves,
                    item[1].generation,
                ),
            )[1]

        if compatible is not None:
            self._hits += 1
            self._compatible_hits += 1
        return compatible

    def store(self, key: ProofKey, entry: ProofTTEntry) -> None:
        # UNKNOWN may describe a timeout, depth limit, or incomplete move
        # set. It must never become a reusable exact result.
        if entry.state is ProofState.UNKNOWN or not entry.complete:
            self._skipped_stores += 1
            return

        if key not in self._entries and len(self._entries) >= self.max_entries:
            oldest_key = min(
                self._entries,
                key=lambda item: self._entries[item].generation,
            )
            del self._entries[oldest_key]
            base = self._base_key(oldest_key)
            indexed = self._depth_index.get(base)
            if indexed is not None:
                indexed.discard(oldest_key)
                if not indexed:
                    del self._depth_index[base]
            self._evictions += 1
        is_new = key not in self._entries
        self._entries[key] = entry
        if is_new:
            self._depth_index.setdefault(
                self._base_key(key),
                set(),
            ).add(key)
        self._stores += 1

    @staticmethod
    def _base_key(
        key: ProofKey,
    ) -> tuple[int, int, int, tuple[object, ...]]:
        return (
            key.board_hash,
            key.attacker,
            key.side_to_move,
            key.obligation_signature,
        )

    def stats(self) -> ProofTableStats:
        return ProofTableStats(
            queries=self._queries,
            hits=self._hits,
            compatible_hits=self._compatible_hits,
            stores=self._stores,
            skipped_stores=self._skipped_stores,
            evictions=self._evictions,
            size=len(self._entries),
        )

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True, slots=True)
class ProofResult:
    state: ProofState
    attacker: int
    side_to_move: int
    best_move: Move | None
    principal_variation: tuple[Move, ...]
    required_defenses: tuple[Move, ...]
    nodes: int
    transposition_hits: int
    searched_attacker_moves: int
    completed: bool
    cutoff_reason: str | None
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class _NodeResult:
    state: ProofState
    complete: bool
    best_move: Move | None = None
    principal_variation: tuple[Move, ...] = ()
    required_defenses: tuple[Move, ...] = ()
    cutoff_reason: str | None = None
    linear_plan: tuple[Move, ...] | None = None


class ProofSearch:
    """Conservative AND/OR threat-space proof search.

    Exact open-three and multi-threat defense sets may become AND nodes;
    incomplete quiet-launch sets still return UNKNOWN. The class has no
    reference to SearchAI's PVS TT, killer moves, history scores, or
    heuristic mate-like scores.
    """

    def __init__(
        self,
        *,
        budget: ProofBudget | None = None,
        analyzer: ThreatAnalyzer | None = None,
        table: ProofTable | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        self.budget = budget or ProofBudget()
        self.analyzer = analyzer or ThreatAnalyzer()
        self.table = table if table is not None else ProofTable()
        self._clock = clock
        self._generation = 0
        self._nodes = 0
        self._transposition_hits = 0
        self._max_attacker_ply = 0
        self._started_at = 0.0
        self._candidate_node_limits: list[int] = []
        self._candidate_deadlines: list[float] = []

    def search(
        self,
        board: Board,
        *,
        attacker: int,
        side_to_move: int | None = None,
    ) -> ProofResult:
        """Search from the current board and always leave it unchanged."""
        if attacker not in (BLACK, WHITE):
            raise ValueError("attacker 只能是 BLACK 或 WHITE。")
        if side_to_move is None:
            side_to_move = (
                BLACK if len(board.move_history) % 2 == 0 else WHITE
            )
        if side_to_move not in (BLACK, WHITE):
            raise ValueError("side_to_move 只能是 BLACK 或 WHITE。")

        self._generation += 1
        self._nodes = 0
        self._transposition_hits = 0
        self._max_attacker_ply = 0
        self._started_at = self._clock()
        self._candidate_node_limits.clear()
        self._candidate_deadlines.clear()

        node_result = self._search_node(
            board,
            attacker=attacker,
            side_to_move=side_to_move,
            remaining_attacker_moves=self.budget.max_attacker_moves,
            remaining_quiet_moves=self.budget.max_quiet_attacker_moves,
            obligation=None,
        )

        return ProofResult(
            state=node_result.state,
            attacker=attacker,
            side_to_move=side_to_move,
            best_move=node_result.best_move,
            principal_variation=node_result.principal_variation,
            required_defenses=node_result.required_defenses,
            nodes=self._nodes,
            transposition_hits=self._transposition_hits,
            searched_attacker_moves=self._max_attacker_ply,
            completed=node_result.complete,
            cutoff_reason=node_result.cutoff_reason,
            elapsed_seconds=max(0.0, self._clock() - self._started_at),
        )

    def search_after_move(
        self,
        board: Board,
        *,
        move: Move,
        mover: int,
        attacker: int,
        side_to_move: int | None = None,
    ) -> ProofResult:
        """Probe a candidate move and always restore the caller's board.

        This is the root-integration boundary used by later SearchAI work:
        the candidate belongs to ``mover`` while the returned state remains
        relative to the fixed ``attacker``.  UNKNOWN is returned unchanged
        and must not be interpreted as a safe candidate.
        """
        if mover not in (BLACK, WHITE):
            raise ValueError("mover 只能是 BLACK 或 WHITE。")
        if attacker not in (BLACK, WHITE):
            raise ValueError("attacker 只能是 BLACK 或 WHITE。")
        if not board.is_empty(*move):
            raise ValueError("只能探测空位置。")
        if side_to_move is None:
            side_to_move = other_side(mover)
        if side_to_move not in (BLACK, WHITE):
            raise ValueError("side_to_move 只能是 BLACK 或 WHITE。")

        board.place(*move, mover)
        try:
            return self.search(
                board,
                attacker=attacker,
                side_to_move=side_to_move,
            )
        finally:
            board.undo()

    def _search_node(
        self,
        board: Board,
        *,
        attacker: int,
        side_to_move: int,
        remaining_attacker_moves: int,
        remaining_quiet_moves: int,
        obligation: Threat | None,
    ) -> _NodeResult:
        cutoff_reason = self._budget_cutoff_reason()
        if cutoff_reason is not None:
            return _NodeResult(
                state=ProofState.UNKNOWN,
                complete=False,
                cutoff_reason=cutoff_reason,
            )

        self._nodes += 1
        used_attacker_moves = (
            self.budget.max_attacker_moves - remaining_attacker_moves
        )
        self._max_attacker_ply = max(
            self._max_attacker_ply,
            used_attacker_moves,
        )

        terminal = self._terminal_result(board, attacker)
        if terminal is not None:
            return terminal

        key = ProofKey(
            board_hash=board.zobrist_hash,
            attacker=attacker,
            side_to_move=side_to_move,
            obligation_signature=(
                () if obligation is None else obligation.signature
            ),
            remaining_attacker_moves=remaining_attacker_moves,
            remaining_quiet_moves=remaining_quiet_moves,
        )
        cached = self.table.get(key)
        if cached is not None:
            self._transposition_hits += 1
            return _NodeResult(
                state=cached.state,
                complete=cached.complete,
                best_move=cached.best_move,
                principal_variation=cached.principal_variation,
                required_defenses=cached.required_defenses,
                linear_plan=cached.linear_plan,
            )

        if side_to_move == attacker:
            result = self._search_or_node(
                board,
                attacker=attacker,
                remaining_attacker_moves=remaining_attacker_moves,
                remaining_quiet_moves=remaining_quiet_moves,
            )
        else:
            result = self._search_and_node(
                board,
                attacker=attacker,
                defender=side_to_move,
                remaining_attacker_moves=remaining_attacker_moves,
                remaining_quiet_moves=remaining_quiet_moves,
                obligation=obligation,
            )

        self.table.store(
            key,
            ProofTTEntry(
                state=result.state,
                complete=result.complete,
                best_move=result.best_move,
                principal_variation=result.principal_variation,
                required_defenses=result.required_defenses,
                linear_plan=result.linear_plan,
                generation=self._generation,
            ),
        )
        return result

    def _search_or_node(
        self,
        board: Board,
        *,
        attacker: int,
        remaining_attacker_moves: int,
        remaining_quiet_moves: int,
    ) -> _NodeResult:
        if remaining_attacker_moves <= 0:
            return _NodeResult(
                state=ProofState.UNKNOWN,
                complete=False,
                cutoff_reason="attacker_depth_limit",
            )

        attacker_wins = tuple(
            sorted(find_winning_moves(board, attacker))
        )
        if attacker_wins:
            move = attacker_wins[0]
            return _NodeResult(
                state=ProofState.PROVEN_WIN,
                complete=True,
                best_move=move,
                principal_variation=(move,),
                linear_plan=(move,),
            )

        defender = other_side(attacker)
        defender_wins = tuple(
            sorted(find_winning_moves(board, defender))
        )
        if len(defender_wins) >= 2:
            return _NodeResult(
                state=ProofState.PROVEN_LOSS,
                complete=True,
                required_defenses=defender_wins,
            )

        # A concrete VCF line is already a strict winning certificate.  Use
        # it before the much more expensive exact threat descriptions.  A
        # miss is deliberately *not* proof of safety: normal AND/OR search
        # continues, while a budget interruption remains UNKNOWN.
        if (
            not defender_wins
            and self.budget.use_vcf_oracle
            and self.budget.vcf_max_attacker_moves > 0
        ):
            try:
                vcf_line = self._find_vcf_witness(board, attacker)
            except _ProofCutoff:
                return _NodeResult(
                    state=ProofState.UNKNOWN,
                    complete=False,
                    cutoff_reason=(
                        self._budget_cutoff_reason() or "vcf_interrupted"
                    ),
                )
            if vcf_line:
                used_attacker_moves = (len(vcf_line) + 1) // 2
                self._max_attacker_ply = max(
                    self._max_attacker_ply,
                    self.budget.max_attacker_moves
                    - remaining_attacker_moves
                    + used_attacker_moves,
                )
                return _NodeResult(
                    state=ProofState.PROVEN_WIN,
                    complete=True,
                    best_move=vcf_line[0],
                    principal_variation=vcf_line,
                    linear_plan=vcf_line,
                )

        if len(defender_wins) == 1:
            # Every other attacker move loses immediately, so this singleton
            # block is a complete OR set for the current node. The block is
            # quiet, so every legal defender reply after it must remain an
            # explicit AND obligation.
            forced_block = defender_wins[0]
            forced_block_defenses = tuple(
                sorted(
                    move
                    for move in board.get_legal_moves()
                    if move != forced_block
                )
            )
            candidate_items: tuple[Threat | ThreatCandidate, ...] = (
                (
                    Threat(
                        gain_move=forced_block,
                        kind=ThreatKind.QUIET,
                        attacker=attacker,
                        required_defenses=forced_block_defenses,
                        coverage_complete=True,
                        legal_reply_count=len(forced_block_defenses),
                    )
                ),
            )
            candidate_coverage_complete = True
        else:
            batch = self.analyzer.generate_attack_candidates(
                board,
                attacker,
                stop_requested=self._deadline_reached,
            )
            if not batch.generation_completed:
                return _NodeResult(
                    state=ProofState.UNKNOWN,
                    complete=False,
                    cutoff_reason=(
                        self._budget_cutoff_reason()
                        or "candidate_generation_interrupted"
                    ),
                )
            candidate_items: tuple[
                Threat | ThreatCandidate | ThreatFrontier,
                ...,
            ] = batch.candidates
            candidate_coverage_complete = batch.coverage_complete

        if (
            not candidate_items
            and self.budget.max_quiet_frontiers <= 0
        ):
            return _NodeResult(
                state=ProofState.UNKNOWN,
                complete=False,
                cutoff_reason="incomplete_attack_set",
            )

        child_results: list[tuple[Threat, _NodeResult]] = []
        winning_result = self._search_candidate_stage(
            board,
            attacker=attacker,
            defender=defender,
            remaining_attacker_moves=remaining_attacker_moves,
            remaining_quiet_moves=remaining_quiet_moves,
            candidate_items=candidate_items,
            child_results=child_results,
        )
        if winning_result is not None:
            return winning_result

        quiet_frontiers: tuple[ThreatFrontier, ...] = ()
        if (
            len(defender_wins) == 0
            and self.budget.max_quiet_frontiers > 0
            and remaining_quiet_moves > 0
            and self._budget_cutoff_reason() is None
        ):
            forcing_moves = {
                candidate.move
                for candidate in batch.candidates
            }
            frontiers = self.analyzer.generate_attack_frontiers(
                board,
                attacker,
                frontier_limit=self.budget.max_quiet_frontiers,
                stop_requested=self._deadline_reached,
            )
            quiet_frontiers = tuple(
                frontier
                for frontier in frontiers
                if (
                    frontier.kind is ThreatKind.QUIET
                    and frontier.gain_move not in forcing_moves
                )
            )
            winning_result = self._search_candidate_stage(
                board,
                attacker=attacker,
                defender=defender,
                remaining_attacker_moves=remaining_attacker_moves,
                remaining_quiet_moves=remaining_quiet_moves,
                candidate_items=quiet_frontiers,
                child_results=child_results,
            )
            if winning_result is not None:
                return winning_result

        states = [child.state for _, child in child_results]
        all_candidates_examined = (
            len(child_results)
            == len(candidate_items) + len(quiet_frontiers)
            and candidate_coverage_complete
        )
        state = combine_or_states(
            states,
            coverage_complete=all_candidates_examined,
        )

        first_unknown = next(
            (
                (threat, child)
                for threat, child in child_results
                if child.state is ProofState.UNKNOWN
            ),
            None,
        )
        cutoff_reason = self._budget_cutoff_reason()
        if cutoff_reason is None and first_unknown is not None:
            cutoff_reason = first_unknown[1].cutoff_reason
        if cutoff_reason is None and state is ProofState.UNKNOWN:
            cutoff_reason = "incomplete_attack_set"

        best_move = (
            child_results[0][0].gain_move if child_results else None
        )
        return _NodeResult(
            state=state,
            complete=state is not ProofState.UNKNOWN,
            best_move=best_move,
            cutoff_reason=cutoff_reason,
        )

    def _search_candidate_stage(
        self,
        board: Board,
        *,
        attacker: int,
        defender: int,
        remaining_attacker_moves: int,
        remaining_quiet_moves: int,
        candidate_items: tuple[
            Threat | ThreatCandidate | ThreatFrontier,
            ...,
        ],
        child_results: list[tuple[Threat, _NodeResult]],
    ) -> _NodeResult | None:
        """Search one already-ordered OR stage.

        Forcing candidates are supplied as the first stage. Quiet frontiers
        are generated only if that stage has no proof witness, avoiding the
        expensive frontier scan at every forcing node.
        """
        for candidate_item in candidate_items:
            if self._budget_cutoff_reason() is not None:
                break

            candidate_limits = self._push_candidate_limits()
            try:
                if isinstance(candidate_item, ThreatCandidate):
                    threat = self.analyzer.describe_candidate(
                        board,
                        candidate_item,
                        attacker,
                        stop_requested=self._deadline_reached,
                    )
                    if not threat.analysis_completed:
                        child = _NodeResult(
                            state=ProofState.UNKNOWN,
                            complete=False,
                            cutoff_reason=(
                                self._budget_cutoff_reason()
                                or "threat_description_interrupted"
                            ),
                        )
                        child_results.append((threat, child))
                        continue
                elif isinstance(candidate_item, ThreatFrontier):
                    threat = self.analyzer.describe_frontier(
                        board,
                        candidate_item,
                        attacker,
                    )
                else:
                    threat = candidate_item

                move = threat.gain_move
                board.place(*move, attacker)
                try:
                    if board.check_win(*move):
                        child = _NodeResult(
                            state=ProofState.PROVEN_WIN,
                            complete=True,
                            linear_plan=(),
                        )
                    else:
                        child = self._search_node(
                            board,
                            attacker=attacker,
                            side_to_move=defender,
                            remaining_attacker_moves=(
                                remaining_attacker_moves - 1
                            ),
                            remaining_quiet_moves=(
                                remaining_quiet_moves - 1
                                if isinstance(
                                    candidate_item,
                                    ThreatFrontier,
                                )
                                and candidate_item.kind is ThreatKind.QUIET
                                else remaining_quiet_moves
                            ),
                            obligation=threat,
                        )
                finally:
                    board.undo()
            finally:
                self._pop_candidate_limits(candidate_limits)

            child_results.append((threat, child))
            if child.state is ProofState.PROVEN_WIN:
                linear_plan = (
                    None
                    if child.linear_plan is None
                    else (move, *child.linear_plan)
                )
                return _NodeResult(
                    state=ProofState.PROVEN_WIN,
                    complete=True,
                    best_move=move,
                    principal_variation=(
                        move,
                        *child.principal_variation,
                    ),
                    required_defenses=child.required_defenses,
                    linear_plan=linear_plan,
                )
        return None

    def _find_vcf_witness(
        self,
        board: Board,
        attacker: int,
    ) -> tuple[Move, ...] | None:
        """Return one fully replayable VCF certificate, if cheaply found.

        Candidate generation may be selective because only a returned line is
        used as proof.  Failing to find a line never proves a loss and falls
        through to the conservative threat-space search.
        """

        # Native VCF is a witness accelerator, never a proof authority.  A
        # returned line is independently replayed below.  A selective miss
        # has no proof meaning and returns to the conservative AND/OR search;
        # repeating the same unsuccessful bounded VCF tree in Python only
        # consumes the final-audit deadline.  The Python VCF implementation
        # remains the unchanged fallback whenever NativeCore is unavailable.
        if native_core.available and self._clock in {
            time.monotonic,
            time.perf_counter,
        }:
            node_limits = [self.budget.max_nodes]
            node_limits.extend(self._candidate_node_limits)
            remaining_nodes = max(0, min(node_limits) - self._nodes)
            deadlines = [
                deadline
                for deadline in (
                    self.budget.deadline,
                    *self._candidate_deadlines,
                )
                if deadline is not None
            ]
            timeout_seconds = (
                None
                if not deadlines
                else max(0.0, min(deadlines) - self._clock())
            )
            if remaining_nodes <= 0 or timeout_seconds == 0.0:
                raise _ProofCutoff
            native = native_core.find_vcf(
                board,
                attacker,
                self.budget.vcf_max_attacker_moves,
                max_nodes=remaining_nodes,
                timeout_seconds=timeout_seconds,
                candidate_limit=self.analyzer.candidate_limit,
            )
            if native is not None:
                if native.cutoff:
                    self._nodes += native.nodes
                    raise _ProofCutoff
                if native.found and validate_vcf_certificate(
                    board,
                    attacker,
                    native.line,
                ):
                    self._nodes += native.nodes
                    return native.line
                if not native.found:
                    return None
                # An invalid native witness is never accepted. Fall through
                # to the independent reference implementation so a native
                # defect cannot suppress an otherwise discoverable witness.
                self._check_vcf_budget()

        search = VCFSearch(
            position_key=lambda position, _player: position.zobrist_hash,
            forcing_candidates=self._vcf_candidates,
            check_timeout=self._check_vcf_budget,
            count_node=self._count_vcf_node,
        )
        return search.find(
            board,
            attacker,
            self.budget.vcf_max_attacker_moves,
        )

    def _vcf_candidates(
        self,
        board: Board,
        attacker: int,
    ) -> list[Move]:
        batch = self.analyzer.generate_attack_candidates(
            board,
            attacker,
            stop_requested=self._deadline_reached,
        )
        if not batch.generation_completed:
            raise _ProofCutoff
        vcf_kinds = {
            ThreatKind.FIVE,
            ThreatKind.DOUBLE_FOUR,
            ThreatKind.OPEN_FOUR,
            ThreatKind.FOUR_THREE,
            ThreatKind.FOUR,
        }
        return [
            candidate.move
            for candidate in batch.candidates
            if candidate.kind in vcf_kinds
        ]

    def _check_vcf_budget(self) -> None:
        if self._budget_cutoff_reason() is not None:
            raise _ProofCutoff

    def _count_vcf_node(self) -> None:
        self._nodes += 1

    def _search_and_node(
        self,
        board: Board,
        *,
        attacker: int,
        defender: int,
        remaining_attacker_moves: int,
        remaining_quiet_moves: int,
        obligation: Threat | None,
    ) -> _NodeResult:
        # A real defender win ends the game before any threat obligation.
        defender_wins = tuple(
            sorted(find_winning_moves(board, defender))
        )
        if defender_wins:
            move = defender_wins[0]
            return _NodeResult(
                state=ProofState.PROVEN_LOSS,
                complete=True,
                best_move=move,
                principal_variation=(move,),
            )

        attacker_wins = tuple(
            sorted(find_winning_moves(board, attacker))
        )
        if len(attacker_wins) >= 2:
            return _NodeResult(
                state=ProofState.PROVEN_WIN,
                complete=True,
                principal_variation=(attacker_wins[0],),
                required_defenses=attacker_wins,
                linear_plan=(),
            )

        if len(attacker_wins) == 1:
            defenses = attacker_wins
            coverage_complete = True
            implicit_refutations: tuple[DefenseRefutation, ...] = ()
        elif (
            obligation is not None
            and obligation.attacker == attacker
            and obligation.analysis_completed
            and obligation.coverage_complete
        ):
            defenses = tuple(
                sorted(
                    set(obligation.required_defenses)
                    | set(obligation.counter_wins)
                )
            )
            coverage_complete = True
            implicit_refutations = obligation.defense_refutations
        else:
            return _NodeResult(
                state=ProofState.UNKNOWN,
                complete=False,
                cutoff_reason="incomplete_defense_set",
            )

        implicit_coverage_complete = (
            self._implicit_refutations_fit_depth(
                implicit_refutations,
                remaining_attacker_moves=remaining_attacker_moves,
            )
        )
        if not defenses and coverage_complete:
            if not implicit_coverage_complete:
                return _NodeResult(
                    state=ProofState.UNKNOWN,
                    complete=False,
                    cutoff_reason="attacker_depth_limit",
                )
            variation = self._refutation_variation(
                implicit_refutations
            )
            self._record_implicit_attacker_move(
                implicit_refutations,
                remaining_attacker_moves=remaining_attacker_moves,
            )
            return _NodeResult(
                state=ProofState.PROVEN_WIN,
                complete=True,
                principal_variation=variation,
                required_defenses=(),
                linear_plan=(),
            )
        if not defenses:
            return _NodeResult(
                state=ProofState.UNKNOWN,
                complete=False,
                cutoff_reason="incomplete_defense_set",
            )

        child_results: list[tuple[Move, _NodeResult]] = []
        reusable_plans: list[tuple[Move, ...]] = []
        merge_obligation = (
            obligation is not None
            and obligation.coverage_complete
            and obligation.analysis_completed
        )
        for move in defenses:
            if self._budget_cutoff_reason() is not None:
                break
            if not board.is_empty(*move):
                continue

            board.place(*move, defender)
            try:
                if board.check_win(*move):
                    child = _NodeResult(
                        state=ProofState.PROVEN_LOSS,
                        complete=True,
                    )
                else:
                    child = None
                    if merge_obligation:
                        for plan in reusable_plans:
                            child = self._replay_linear_plan(
                                board,
                                attacker=attacker,
                                remaining_attacker_moves=(
                                    remaining_attacker_moves
                                ),
                                plan=plan,
                            )
                            if child is not None:
                                break
                    if child is None:
                        child = self._search_node(
                            board,
                            attacker=attacker,
                            side_to_move=attacker,
                            remaining_attacker_moves=(
                                remaining_attacker_moves
                            ),
                            remaining_quiet_moves=remaining_quiet_moves,
                            obligation=None,
                        )
            finally:
                board.undo()

            child_results.append((move, child))
            if (
                merge_obligation
                and child.state is ProofState.PROVEN_WIN
                and child.linear_plan is not None
                and child.linear_plan not in reusable_plans
            ):
                reusable_plans.append(child.linear_plan)
            if child.state is ProofState.PROVEN_LOSS:
                return _NodeResult(
                    state=ProofState.PROVEN_LOSS,
                    complete=True,
                    best_move=move,
                    principal_variation=(
                        move,
                        *child.principal_variation,
                    ),
                    required_defenses=defenses,
                )

        all_defenses_examined = (
            len(child_results) == len(defenses)
            and coverage_complete
            and implicit_coverage_complete
        )
        state = combine_and_states(
            (child.state for _, child in child_results),
            coverage_complete=all_defenses_examined,
        )
        cutoff_reason = self._budget_cutoff_reason()
        if cutoff_reason is None and state is ProofState.UNKNOWN:
            if not implicit_coverage_complete:
                cutoff_reason = "attacker_depth_limit"
            else:
                cutoff_reason = next(
                    (
                        child.cutoff_reason
                        for _, child in child_results
                        if child.state is ProofState.UNKNOWN
                    ),
                    "incomplete_defense_set",
                )

        variation: tuple[Move, ...] = ()
        if child_results:
            move, child = child_results[0]
            variation = (move, *child.principal_variation)
        elif implicit_refutations:
            variation = self._refutation_variation(
                implicit_refutations
            )
        if state is ProofState.PROVEN_WIN:
            self._record_implicit_attacker_move(
                implicit_refutations,
                remaining_attacker_moves=remaining_attacker_moves,
            )
        linear_plan = None
        if (
            state is ProofState.PROVEN_WIN
            and len(child_results) == 1
            and child_results[0][1].linear_plan is not None
        ):
            move, child = child_results[0]
            linear_plan = (move, *child.linear_plan)
        return _NodeResult(
            state=state,
            complete=state is not ProofState.UNKNOWN,
            principal_variation=variation,
            required_defenses=defenses,
            cutoff_reason=cutoff_reason,
            linear_plan=linear_plan,
        )

    def _replay_linear_plan(
        self,
        board: Board,
        *,
        attacker: int,
        remaining_attacker_moves: int,
        plan: tuple[Move, ...],
    ) -> _NodeResult | None:
        """Revalidate a learned single-defense proof on another reply.

        No state is reused merely because two boards look similar. Every
        attacker move is described again on the actual board, including
        exact defense coverage and counter-win checks. Reuse succeeds only
        when the proof remains linear (at most one explicit defense at each
        step) and the saved moves still form a complete strict certificate.
        """
        cutoff_reason = self._budget_cutoff_reason()
        if cutoff_reason is not None or remaining_attacker_moves <= 0:
            return None
        if not plan:
            return None

        defender = other_side(attacker)
        defender_wins = tuple(
            sorted(find_winning_moves(board, defender))
        )
        if len(defender_wins) >= 2:
            return None

        attack_move = plan[0]
        if not board.is_empty(*attack_move):
            return None
        if len(defender_wins) == 1 and attack_move != defender_wins[0]:
            return None

        self._nodes += 1
        threat = self.analyzer.describe_move(
            board,
            attack_move,
            attacker,
            stop_requested=self._deadline_reached,
        )
        if (
            not threat.analysis_completed
            or self._budget_cutoff_reason() is not None
        ):
            return None

        board.place(*attack_move, attacker)
        try:
            if board.check_win(*attack_move):
                return _NodeResult(
                    state=ProofState.PROVEN_WIN,
                    complete=True,
                    best_move=attack_move,
                    principal_variation=(attack_move,),
                    linear_plan=(attack_move,),
                )

            if find_winning_moves(board, defender):
                return None

            attacker_wins = tuple(
                sorted(find_winning_moves(board, attacker))
            )
            if len(attacker_wins) >= 2:
                return _NodeResult(
                    state=ProofState.PROVEN_WIN,
                    complete=True,
                    best_move=attack_move,
                    principal_variation=(
                        attack_move,
                        attacker_wins[0],
                    ),
                    required_defenses=attacker_wins,
                    linear_plan=(attack_move,),
                )

            if len(attacker_wins) == 1:
                defenses = attacker_wins
                implicit_refutations: tuple[
                    DefenseRefutation,
                    ...,
                ] = ()
            elif threat.coverage_complete:
                defenses = tuple(
                    sorted(
                        set(threat.required_defenses)
                        | set(threat.counter_wins)
                    )
                )
                implicit_refutations = threat.defense_refutations
            else:
                return None

            remaining_after_attack = remaining_attacker_moves - 1
            if not self._implicit_refutations_fit_depth(
                implicit_refutations,
                remaining_attacker_moves=remaining_after_attack,
            ):
                return None
            if not defenses:
                self._record_implicit_attacker_move(
                    implicit_refutations,
                    remaining_attacker_moves=remaining_after_attack,
                )
                return _NodeResult(
                    state=ProofState.PROVEN_WIN,
                    complete=True,
                    best_move=attack_move,
                    principal_variation=(attack_move,),
                    linear_plan=(attack_move,),
                )
            if len(defenses) != 1 or len(plan) < 2:
                return None

            defense_move = defenses[0]
            if plan[1] != defense_move or not board.is_empty(*defense_move):
                return None
            board.place(*defense_move, defender)
            try:
                if board.check_win(*defense_move):
                    return None
                child = self._replay_linear_plan(
                    board,
                    attacker=attacker,
                    remaining_attacker_moves=remaining_after_attack,
                    plan=plan[2:],
                )
            finally:
                board.undo()
        finally:
            board.undo()

        if child is None:
            return None
        return _NodeResult(
            state=ProofState.PROVEN_WIN,
            complete=True,
            best_move=attack_move,
            principal_variation=(
                attack_move,
                defense_move,
                *child.principal_variation,
            ),
            linear_plan=(
                attack_move,
                defense_move,
                *child.linear_plan,
            ),
        )

    @staticmethod
    def _implicit_refutations_fit_depth(
        refutations: tuple[DefenseRefutation, ...],
        *,
        remaining_attacker_moves: int,
    ) -> bool:
        return (
            remaining_attacker_moves > 0
            or all(
                refutation.continuation_is_immediate
                for refutation in refutations
            )
        )

    @staticmethod
    def _refutation_variation(
        refutations: tuple[DefenseRefutation, ...],
    ) -> tuple[Move, ...]:
        if not refutations:
            return ()
        witness = refutations[0]
        variation = (
            witness.defense_move,
            witness.continuation_move,
        )
        if (
            not witness.continuation_is_immediate
            and len(witness.winning_points) >= 2
        ):
            variation = (
                *variation,
                witness.winning_points[0],
                witness.winning_points[1],
            )
        return variation

    def _record_implicit_attacker_move(
        self,
        refutations: tuple[DefenseRefutation, ...],
        *,
        remaining_attacker_moves: int,
    ) -> None:
        if not any(
            not refutation.continuation_is_immediate
            for refutation in refutations
        ):
            return
        used_attacker_moves = (
            self.budget.max_attacker_moves
            - remaining_attacker_moves
            + 1
        )
        self._max_attacker_ply = max(
            self._max_attacker_ply,
            used_attacker_moves,
        )

    def _terminal_result(
        self,
        board: Board,
        attacker: int,
    ) -> _NodeResult | None:
        if board.move_history:
            row, column, last_player = board.move_history[-1]
            if board.check_win(row, column):
                return _NodeResult(
                    state=(
                        ProofState.PROVEN_WIN
                        if last_player == attacker
                        else ProofState.PROVEN_LOSS
                    ),
                    complete=True,
                    linear_plan=(
                        ()
                        if last_player == attacker
                        else None
                    ),
                )
        if board.is_full():
            return _NodeResult(
                state=ProofState.PROVEN_LOSS,
                complete=True,
            )
        return None

    def _budget_cutoff_reason(self) -> str | None:
        if self._nodes >= self.budget.max_nodes:
            return "node_limit"
        if (
            self._candidate_node_limits
            and self._nodes >= min(self._candidate_node_limits)
        ):
            return "candidate_node_limit"
        if (
            self._candidate_deadlines
            and self._clock() >= min(self._candidate_deadlines)
        ):
            return "candidate_deadline"
        if self._hard_deadline_reached():
            return "deadline"
        return None

    def _push_candidate_limits(
        self,
    ) -> tuple[int | None, float | None]:
        node_limit = self.budget.max_nodes_per_candidate
        absolute_limit = (
            None if node_limit is None else self._nodes + node_limit
        )
        if absolute_limit is not None:
            self._candidate_node_limits.append(absolute_limit)

        seconds = self.budget.max_seconds_per_candidate
        candidate_deadline = (
            None if seconds is None else self._clock() + seconds
        )
        if candidate_deadline is not None:
            self._candidate_deadlines.append(candidate_deadline)
        return absolute_limit, candidate_deadline

    def _pop_candidate_limits(
        self,
        limits: tuple[int | None, float | None],
    ) -> None:
        absolute_limit, candidate_deadline = limits
        if candidate_deadline is not None:
            popped_deadline = self._candidate_deadlines.pop()
            if popped_deadline != candidate_deadline:
                raise RuntimeError("候选时间预算栈损坏。")
        if absolute_limit is not None:
            popped_limit = self._candidate_node_limits.pop()
            if popped_limit != absolute_limit:
                raise RuntimeError("候选节点预算栈损坏。")

    def _deadline_reached(self) -> bool:
        return (
            (
                self._candidate_deadlines
                and self._clock() >= min(self._candidate_deadlines)
            )
            or self._hard_deadline_reached()
        )

    def _hard_deadline_reached(self) -> bool:
        return (
            self.budget.deadline is not None
            and self._clock() >= self.budget.deadline
        )
