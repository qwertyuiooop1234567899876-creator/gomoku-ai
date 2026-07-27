from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from engine.ai import CandidateAnalysis, DecisionAnalysis, Move, ScoringAI
from engine.board import BLACK, DIRECTIONS, EMPTY, WHITE, Board
from engine.evaluator import (
    ThreatProfile,
    analyze_move_threats,
    evaluate_board,
    evaluate_move,
    find_winning_moves,
    other_side,
)

MATE_SCORE = 1_000_000_000
INFINITY = MATE_SCORE * 2


class SearchTimeout(RuntimeError):
    """搜索超过本手时间预算。"""


class BoundType(str, Enum):
    EXACT = "exact"
    LOWER = "lower"
    UPPER = "upper"


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """V0.7 搜索参数。"""

    max_depth: int = 3
    time_limit_seconds: float | None = 2.0
    root_candidate_limit: int = 12
    branch_candidate_limit: int = 8
    preselection_factor: int = 3
    candidate_radius: int = 2
    recent_move_count: int = 4
    threat_extension_depth: int = 2
    use_transposition_table: bool = True

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
        if self.recent_move_count < 1:
            raise ValueError("recent_move_count 必须大于 0。")
        if self.threat_extension_depth < 0:
            raise ValueError("threat_extension_depth 不能小于 0。")


@dataclass(slots=True)
class SearchCounters:
    nodes: int = 0
    cutoffs: int = 0
    transposition_hits: int = 0


@dataclass(frozen=True, slots=True)
class TTEntry:
    depth: int
    extension_depth: int
    score: int
    bound: BoundType
    principal_variation: tuple[Move, ...]


@dataclass(frozen=True, slots=True)
class RootResult:
    move: Move
    score: int
    principal_variation: tuple[Move, ...]
    ranked_moves: tuple[tuple[Move, int], ...]


class SearchAI(ScoringAI):
    """
    使用迭代加深 Negamax、Alpha-Beta 剪枝和威胁延伸的五子棋 AI。

    V0.6.2 的一步胜负、复合威胁与走法评分仍作为搜索的战术保护层
    和着法排序器；V0.7 在其上继续推演双方的最佳回应。
    """

    def __init__(
        self,
        player: int = WHITE,
        *,
        max_depth: int = 3,
        time_limit_seconds: float | None = 2.0,
        root_candidate_limit: int = 12,
        branch_candidate_limit: int = 8,
        threat_extension_depth: int = 2,
        diagnostics: bool = False,
        top_n: int = 5,
    ) -> None:
        super().__init__(
            player=player,
            diagnostics=diagnostics,
            top_n=top_n,
        )
        self.config = SearchConfig(
            max_depth=max_depth,
            time_limit_seconds=time_limit_seconds,
            root_candidate_limit=root_candidate_limit,
            branch_candidate_limit=branch_candidate_limit,
            threat_extension_depth=threat_extension_depth,
        )
        self._deadline: float | None = None
        self._counters = SearchCounters()
        self._transposition_table: dict[tuple[object, ...], TTEntry] = {}
        self._search_started = 0.0

    def choose_move(self, board: Board) -> Move:
        """按硬战术优先级处理，再进行限时迭代加深搜索。"""
        self.last_analysis = None
        legal_moves = board.get_legal_moves()

        if not legal_moves:
            raise ValueError("棋盘已满，电脑无法落子。")

        self._search_started = time.perf_counter()
        # 战术保护层与根候选排序必须完整完成；时间预算从 Negamax
        # 正式开始时计起，避免极短时限导致 AI 无法返回合法着法。
        self._deadline = None
        self._counters = SearchCounters()
        self._transposition_table.clear()

        # 1. 一步五连不需要搜索。
        own_wins = self._find_winning_moves(
            board,
            legal_moves,
            self.player,
        )
        if own_wins:
            selected = own_wins[0]
            self._save_search_analysis(
                selected_move=selected,
                reason="立即五连",
                candidate_count=len(legal_moves),
                ranked_moves=[
                    (move, MATE_SCORE)
                    for move in own_wins
                ],
                completed_depth=0,
                principal_variation=(selected,),
                search_completed=True,
            )
            return selected

        # 2. 对手只有一个胜点时，它是唯一合法防守。
        opponent_wins = self._find_winning_moves(
            board,
            legal_moves,
            self.opponent,
        )
        if len(opponent_wins) == 1:
            selected = opponent_wins[0]
            self._save_search_analysis(
                selected_move=selected,
                reason="封堵唯一胜点",
                candidate_count=len(legal_moves),
                ranked_moves=[
                    (
                        selected,
                        evaluate_move(
                            board,
                            selected[0],
                            selected[1],
                            self.player,
                        ),
                    )
                ],
                completed_depth=0,
                principal_variation=(selected,),
                search_completed=True,
            )
            return selected

        # 3. 多个胜点已经无法全部封住，只选反击价值最高的一点。
        if len(opponent_wins) >= 2:
            selected = self._choose_emergency_block(
                board,
                opponent_wins,
            )
            ranked = sorted(
                (
                    (
                        move,
                        evaluate_move(
                            board,
                            move[0],
                            move[1],
                            self.player,
                        ),
                    )
                    for move in opponent_wins
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            self._save_search_analysis(
                selected_move=selected,
                reason="对手多胜点：局面已属强制败势",
                candidate_count=len(legal_moves),
                ranked_moves=ranked,
                completed_depth=0,
                principal_variation=(selected,),
                search_completed=True,
            )
            return selected

        if not board.move_history:
            center = board.size // 2
            selected = (center, center)
            self._save_search_analysis(
                selected_move=selected,
                reason="空棋盘选择天元",
                candidate_count=len(legal_moves),
                ranked_moves=[(selected, 0)],
                completed_depth=0,
                principal_variation=(selected,),
                search_completed=True,
            )
            return selected

        # 4. V0.6.2 识别出的强制威胁不再直接拍板，而是缩小根分支，
        #    让 Negamax 检查对手是否存在更强反击，并比较后续变化。
        root_pool = self._root_pool(board, legal_moves)
        own_profiles = self._profile_moves(
            board,
            root_pool,
            self.player,
        )
        own_forcing_moves = [
            move
            for move, profile in own_profiles.items()
            if profile.forced_win
        ]

        opponent_profiles = self._profile_moves(
            board,
            root_pool,
            self.opponent,
        )
        opponent_forcing_moves = [
            move
            for move, profile in opponent_profiles.items()
            if profile.forced_win
        ]

        if own_forcing_moves:
            search_candidates = self._order_specific_moves(
                board,
                own_forcing_moves,
                self.player,
            )
            tactical_reason = "搜索自身强制威胁的最佳变化"
        elif opponent_forcing_moves:
            # 5. 对手下一手会制造强制威胁时，只在关键点中搜索最佳防守。
            search_candidates = self._order_specific_moves(
                board,
                opponent_forcing_moves,
                self.player,
            )
            tactical_reason = "搜索对手强制威胁的最佳防守"
        else:
            search_candidates = self._ordered_moves(
                board,
                self.player,
                at_root=True,
            )
            tactical_reason = "Negamax 搜索最佳变化"

        if not search_candidates:
            selected = self._choose_emergency_block(board, legal_moves)
            self._save_search_analysis(
                selected_move=selected,
                reason="搜索候选为空，回退到静态评分",
                candidate_count=len(legal_moves),
                ranked_moves=[
                    (
                        selected,
                        evaluate_move(
                            board,
                            selected[0],
                            selected[1],
                            self.player,
                        ),
                    )
                ],
                completed_depth=0,
                principal_variation=(selected,),
                search_completed=False,
            )
            return selected

        fallback_move = search_candidates[0]
        fallback_score = evaluate_move(
            board,
            fallback_move[0],
            fallback_move[1],
            self.player,
        )
        best_result = RootResult(
            move=fallback_move,
            score=fallback_score,
            principal_variation=(fallback_move,),
            ranked_moves=((fallback_move, fallback_score),),
        )
        completed_depth = 0
        search_completed = True

        self._deadline = (
            None
            if self.config.time_limit_seconds is None
            else time.perf_counter() + self.config.time_limit_seconds
        )

        # 迭代加深：超时只丢弃当前未完成层，保留上一层结果。
        for depth in range(1, self.config.max_depth + 1):
            try:
                result = self._search_root(
                    board,
                    self.player,
                    depth,
                    search_candidates,
                )
            except SearchTimeout:
                search_completed = False
                break

            best_result = result
            completed_depth = depth

            if abs(result.score) >= MATE_SCORE - 100:
                break

        reason = (
            f"{tactical_reason}（完成深度 {completed_depth}）"
            if completed_depth > 0
            else f"{tactical_reason}（时间不足，使用静态回退）"
        )
        self._save_search_analysis(
            selected_move=best_result.move,
            reason=reason,
            candidate_count=len(search_candidates),
            ranked_moves=list(best_result.ranked_moves),
            completed_depth=completed_depth,
            principal_variation=best_result.principal_variation,
            search_completed=search_completed,
            own_profiles=own_profiles,
            opponent_profiles=opponent_profiles,
        )
        return best_result.move

    def _search_root(
        self,
        board: Board,
        player: int,
        depth: int,
        candidates: list[Move],
    ) -> RootResult:
        alpha = -INFINITY
        beta = INFINITY
        ranked: list[tuple[Move, int, tuple[Move, ...]]] = []

        for move in candidates:
            self._check_timeout()
            board.place(move[0], move[1], player)

            try:
                if board.check_win(move[0], move[1]):
                    score = MATE_SCORE
                    child_pv: tuple[Move, ...] = ()
                else:
                    child_score, child_pv = self._negamax(
                        board,
                        other_side(player),
                        depth - 1,
                        -beta,
                        -alpha,
                        ply=1,
                        extension_depth=self.config.threat_extension_depth,
                    )
                    score = -child_score
            finally:
                board.undo()

            pv = (move, *child_pv)
            ranked.append((move, score, pv))
            alpha = max(alpha, score)

        # Python 排序稳定：同分时保留 V0.6.2 着法排序的先后顺序。
        ranked.sort(
            key=lambda item: item[1],
            reverse=True,
        )
        best_move, best_score, best_pv = ranked[0]

        return RootResult(
            move=best_move,
            score=best_score,
            principal_variation=best_pv,
            ranked_moves=tuple(
                (move, score)
                for move, score, _ in ranked
            ),
        )

    def _negamax(
        self,
        board: Board,
        player: int,
        depth: int,
        alpha: int,
        beta: int,
        *,
        ply: int,
        extension_depth: int,
    ) -> tuple[int, tuple[Move, ...]]:
        self._counters.nodes += 1
        self._check_timeout()

        if board.is_full():
            return 0, ()

        alpha_original = alpha
        key = self._position_key(
            board,
            player,
            depth,
            extension_depth,
        )

        if self.config.use_transposition_table:
            entry = self._transposition_table.get(key)
            if (
                entry is not None
                and entry.depth >= depth
                and entry.extension_depth >= extension_depth
            ):
                self._counters.transposition_hits += 1
                if entry.bound == BoundType.EXACT:
                    return entry.score, entry.principal_variation
                if entry.bound == BoundType.LOWER:
                    alpha = max(alpha, entry.score)
                elif entry.bound == BoundType.UPPER:
                    beta = min(beta, entry.score)
                if alpha >= beta:
                    return entry.score, entry.principal_variation

        if depth <= 0:
            score, pv = self._threat_extension(
                board,
                player,
                alpha,
                beta,
                ply=ply,
                extension_depth=extension_depth,
            )
            self._store_tt(
                key,
                depth,
                extension_depth,
                score,
                alpha_original,
                beta,
                pv,
            )
            return score, pv

        moves = self._ordered_moves(
            board,
            player,
            at_root=False,
        )

        if not moves:
            score = self._static_score(board, player)
            return score, ()

        best_score = -INFINITY
        best_pv: tuple[Move, ...] = ()

        for move in moves:
            board.place(move[0], move[1], player)

            try:
                if board.check_win(move[0], move[1]):
                    score = MATE_SCORE - ply
                    child_pv: tuple[Move, ...] = ()
                else:
                    child_score, child_pv = self._negamax(
                        board,
                        other_side(player),
                        depth - 1,
                        -beta,
                        -alpha,
                        ply=ply + 1,
                        extension_depth=extension_depth,
                    )
                    score = -child_score
            finally:
                board.undo()

            if score > best_score:
                best_score = score
                best_pv = (move, *child_pv)

            alpha = max(alpha, score)
            if alpha >= beta:
                self._counters.cutoffs += 1
                break

        self._store_tt(
            key,
            depth,
            extension_depth,
            best_score,
            alpha_original,
            beta,
            best_pv,
        )
        return best_score, best_pv

    def _threat_extension(
        self,
        board: Board,
        player: int,
        alpha: int,
        beta: int,
        *,
        ply: int,
        extension_depth: int,
    ) -> tuple[int, tuple[Move, ...]]:
        """叶子节点继续搜索立即胜负和强制威胁，降低视野效应。"""
        legal_moves = board.get_legal_moves()

        if not legal_moves:
            return 0, ()

        own_wins = find_winning_moves(
            board,
            player,
            legal_moves,
        )
        if own_wins:
            return MATE_SCORE - ply, (own_wins[0],)

        opponent = other_side(player)
        opponent_wins = find_winning_moves(
            board,
            opponent,
            legal_moves,
        )

        if len(opponent_wins) >= 2:
            return -MATE_SCORE + ply, ()

        if extension_depth <= 0:
            return self._static_score(board, player), ()

        if len(opponent_wins) == 1:
            forcing_moves = opponent_wins
        else:
            forcing_moves = self._forcing_candidates(
                board,
                player,
            )

        if not forcing_moves:
            return self._static_score(board, player), ()

        best_score = -INFINITY
        best_pv: tuple[Move, ...] = ()

        for move in forcing_moves:
            self._check_timeout()
            board.place(move[0], move[1], player)

            try:
                if board.check_win(move[0], move[1]):
                    score = MATE_SCORE - ply
                    child_pv: tuple[Move, ...] = ()
                else:
                    child_score, child_pv = self._threat_extension(
                        board,
                        opponent,
                        -beta,
                        -alpha,
                        ply=ply + 1,
                        extension_depth=extension_depth - 1,
                    )
                    score = -child_score
            finally:
                board.undo()

            if score > best_score:
                best_score = score
                best_pv = (move, *child_pv)

            alpha = max(alpha, score)
            if alpha >= beta:
                self._counters.cutoffs += 1
                break

        return best_score, best_pv

    def _forcing_candidates(
        self,
        board: Board,
        player: int,
    ) -> list[Move]:
        """只对局部高关联点做威胁延伸，避免叶子节点重复全盘评分。"""
        legal_moves = board.get_legal_moves()
        raw_candidates = self._raw_candidates(
            board,
            legal_moves,
            at_root=False,
        )
        shortlist = sorted(
            raw_candidates,
            key=lambda move: self._quick_order_score(
                board,
                move,
                player,
            ),
            reverse=True,
        )[: max(8, self.config.branch_candidate_limit)]

        forcing: list[tuple[Move, ThreatProfile, int]] = []

        for move in shortlist:
            self._check_timeout()
            profile = analyze_move_threats(
                board,
                move[0],
                move[1],
                player,
            )
            if (
                profile.forced_win
                or profile.four_directions >= 1
                or profile.open_three_directions >= 1
            ):
                forcing.append(
                    (
                        move,
                        profile,
                        self._quick_order_score(
                            board,
                            move,
                            player,
                        ),
                    )
                )

        forcing.sort(
            key=lambda item: (
                item[1].tactical_rank,
                item[2],
            ),
            reverse=True,
        )
        return [move for move, _, _ in forcing[:4]]

    def _root_pool(
        self,
        board: Board,
        legal_moves: list[Move],
    ) -> list[Move]:
        nearby = self._get_nearby_moves(
            board,
            legal_moves,
            radius=self.config.candidate_radius,
        )
        return nearby if nearby else legal_moves

    def _ordered_moves(
        self,
        board: Board,
        player: int,
        *,
        at_root: bool,
        limit: int | None = None,
    ) -> list[Move]:
        legal_moves = board.get_legal_moves()

        if not legal_moves:
            return []

        own_wins = find_winning_moves(
            board,
            player,
            legal_moves,
        )
        if own_wins:
            return own_wins

        opponent = other_side(player)
        opponent_wins = find_winning_moves(
            board,
            opponent,
            legal_moves,
        )
        if len(opponent_wins) == 1:
            return opponent_wins
        if len(opponent_wins) >= 2:
            return self._order_specific_moves(
                board,
                opponent_wins,
                player,
            )

        if not board.move_history:
            center = board.size // 2
            return [(center, center)]

        raw_candidates = self._raw_candidates(
            board,
            legal_moves,
            at_root=at_root,
        )
        if not raw_candidates:
            raw_candidates = legal_moves

        desired_limit = (
            self.config.root_candidate_limit
            if at_root
            else self.config.branch_candidate_limit
        )
        if limit is not None:
            desired_limit = limit

        # 内层节点先用廉价局部指标预筛，再调用完整 V0.6.2 走法评分。
        if not at_root:
            preselection_limit = max(
                desired_limit,
                desired_limit * self.config.preselection_factor,
            )
            raw_candidates = sorted(
                raw_candidates,
                key=lambda move: self._quick_order_score(
                    board,
                    move,
                    player,
                ),
                reverse=True,
            )[:preselection_limit]

        ranked = self._order_specific_moves(
            board,
            raw_candidates,
            player,
        )
        return ranked[:desired_limit]

    def _raw_candidates(
        self,
        board: Board,
        legal_moves: list[Move],
        *,
        at_root: bool,
    ) -> list[Move]:
        legal_set = set(legal_moves)
        candidate_set: set[Move] = set()

        if at_root:
            for row, column, _ in board.move_history:
                self._add_neighborhood(
                    board,
                    legal_set,
                    candidate_set,
                    row,
                    column,
                    self.config.candidate_radius,
                )
        else:
            recent_moves = board.move_history[
                -self.config.recent_move_count:
            ]
            for row, column, _ in recent_moves:
                self._add_neighborhood(
                    board,
                    legal_set,
                    candidate_set,
                    row,
                    column,
                    self.config.candidate_radius,
                )

            # 额外保留所有棋子紧邻位置，避免完全漏掉远端连接点。
            for row, column, _ in board.move_history:
                self._add_neighborhood(
                    board,
                    legal_set,
                    candidate_set,
                    row,
                    column,
                    1,
                )

        return sorted(candidate_set)

    @staticmethod
    def _add_neighborhood(
        board: Board,
        legal_set: set[Move],
        target: set[Move],
        row: int,
        column: int,
        radius: int,
    ) -> None:
        for row_step in range(-radius, radius + 1):
            for column_step in range(-radius, radius + 1):
                candidate = (
                    row + row_step,
                    column + column_step,
                )
                if (
                    candidate in legal_set
                    and board.is_inside(*candidate)
                ):
                    target.add(candidate)

    def _order_specific_moves(
        self,
        board: Board,
        moves: list[Move],
        player: int,
    ) -> list[Move]:
        center = (board.size - 1) / 2
        scored: list[tuple[Move, int]] = []

        for move in moves:
            self._check_timeout()
            scored.append(
                (
                    move,
                    evaluate_move(
                        board,
                        move[0],
                        move[1],
                        player,
                    ),
                )
            )

        scored.sort(
            key=lambda item: (
                item[1],
                -(
                    (item[0][0] - center) ** 2
                    + (item[0][1] - center) ** 2
                ),
                -item[0][0],
                -item[0][1],
            ),
            reverse=True,
        )
        return [move for move, _ in scored]

    def _quick_order_score(
        self,
        board: Board,
        move: Move,
        player: int,
    ) -> int:
        row, column = move
        opponent = other_side(player)
        score = 0

        # 邻近己方棋子偏向进攻，邻近对方棋子偏向防守。
        distance_weights = (0, 24, 8, 3, 1)

        for row_step, column_step in DIRECTIONS:
            for sign in (-1, 1):
                for distance in range(1, 5):
                    neighbor_row = row + sign * distance * row_step
                    neighbor_column = (
                        column + sign * distance * column_step
                    )

                    if not board.is_inside(
                        neighbor_row,
                        neighbor_column,
                    ):
                        break

                    cell = board.grid[neighbor_row][neighbor_column]
                    weight = distance_weights[distance]

                    if cell == player:
                        score += weight * 3
                    elif cell == opponent:
                        score += weight * 4
                    elif cell == EMPTY:
                        score += weight

        center = (board.size - 1) / 2
        score -= int(
            (row - center) ** 2
            + (column - center) ** 2
        )
        return score

    def _rank_profile_pairs(
        self,
        board: Board,
        profiled_moves: list[tuple[Move, ThreatProfile]],
        player: int,
    ) -> list[tuple[Move, int]]:
        ranked = [
            (
                move,
                profile.tactical_rank * 10_000_000
                + evaluate_move(
                    board,
                    move[0],
                    move[1],
                    player,
                ),
            )
            for move, profile in profiled_moves
        ]
        ranked.sort(
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked

    def _static_score(self, board: Board, player: int) -> int:
        score = evaluate_board(board, player)
        return max(
            -MATE_SCORE + 10_000,
            min(MATE_SCORE - 10_000, score),
        )

    def _position_key(
        self,
        board: Board,
        player: int,
        depth: int,
        extension_depth: int,
    ) -> tuple[object, ...]:
        flattened = tuple(
            cell
            for row in board.grid
            for cell in row
        )
        return (
            board.size,
            flattened,
            player,
        )

    def _store_tt(
        self,
        key: tuple[object, ...],
        depth: int,
        extension_depth: int,
        score: int,
        alpha_original: int,
        beta: int,
        principal_variation: tuple[Move, ...],
    ) -> None:
        if not self.config.use_transposition_table:
            return

        if score <= alpha_original:
            bound = BoundType.UPPER
        elif score >= beta:
            bound = BoundType.LOWER
        else:
            bound = BoundType.EXACT

        self._transposition_table[key] = TTEntry(
            depth=depth,
            extension_depth=extension_depth,
            score=score,
            bound=bound,
            principal_variation=principal_variation,
        )

    def _check_timeout(self) -> None:
        if (
            self._deadline is not None
            and time.perf_counter() >= self._deadline
        ):
            raise SearchTimeout

    def _save_search_analysis(
        self,
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
    ) -> None:
        elapsed = time.perf_counter() - self._search_started
        top_candidates: list[CandidateAnalysis] = []

        if self.diagnostics:
            own_profiles = own_profiles or {}
            opponent_profiles = opponent_profiles or {}

            for move, score in ranked_moves[: self.top_n]:
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

        self.last_analysis = DecisionAnalysis(
            selected_move=selected_move,
            reason=reason,
            candidate_count=candidate_count,
            top_candidates=tuple(top_candidates),
            search_depth=completed_depth,
            nodes=self._counters.nodes,
            cutoffs=self._counters.cutoffs,
            transposition_hits=self._counters.transposition_hits,
            elapsed_seconds=elapsed,
            principal_variation=principal_variation,
            search_completed=search_completed,
        )
