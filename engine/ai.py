from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from engine.board import BLACK, EMPTY, WHITE, Board
from engine.evaluator import (
    ThreatProfile,
    analyze_move_threats,
    evaluate_move,
    find_winning_moves,
)
from engine.game import format_move

Move = tuple[int, int]


@dataclass(frozen=True, slots=True)
class CandidateAnalysis:
    move: Move
    score: int
    own_threat: str = "普通"
    opponent_threat: str = "普通"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["coordinate"] = format_move(*self.move)
        return data


@dataclass(frozen=True, slots=True)
class DecisionAnalysis:
    selected_move: Move
    reason: str
    candidate_count: int
    top_candidates: tuple[CandidateAnalysis, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_move": list(self.selected_move),
            "selected_coordinate": format_move(*self.selected_move),
            "reason": self.reason,
            "candidate_count": self.candidate_count,
            "top_candidates": [
                candidate.to_dict()
                for candidate in self.top_candidates
            ],
        }


class RandomAI:
    """从所有合法位置中随机选择一步。"""

    def choose_move(self, board: Board) -> Move:
        legal_moves = board.get_legal_moves()

        if not legal_moves:
            raise ValueError("棋盘已满，电脑无法落子。")

        return random.choice(legal_moves)


class TacticalAI:
    """能够立即获胜、封堵对手，并优先靠近棋局落子。"""

    def __init__(self, player: int = WHITE) -> None:
        if player not in (BLACK, WHITE):
            raise ValueError("AI 玩家只能是 BLACK 或 WHITE。")

        self.player = player
        self.opponent = WHITE if player == BLACK else BLACK

    def choose_move(self, board: Board) -> Move:
        legal_moves = board.get_legal_moves()

        if not legal_moves:
            raise ValueError("棋盘已满，电脑无法落子。")

        winning_moves = self._find_winning_moves(
            board,
            legal_moves,
            self.player,
        )
        if winning_moves:
            return winning_moves[0]

        blocking_moves = self._find_winning_moves(
            board,
            legal_moves,
            self.opponent,
        )
        if blocking_moves:
            return blocking_moves[0]

        if not board.move_history:
            center = board.size // 2
            if board.is_empty(center, center):
                return center, center

        nearby_moves = self._get_nearby_moves(
            board,
            legal_moves,
        )
        candidates = nearby_moves if nearby_moves else legal_moves
        center = (board.size - 1) / 2

        return min(
            candidates,
            key=lambda move: (
                (move[0] - center) ** 2
                + (move[1] - center) ** 2,
                move[0],
                move[1],
            ),
        )

    @staticmethod
    def _find_winning_moves(
        board: Board,
        legal_moves: list[Move],
        player: int,
    ) -> list[Move]:
        """返回全部一步获胜点，避免漏掉双端活四。"""
        return find_winning_moves(
            board,
            player,
            legal_moves,
        )

    @staticmethod
    def _find_winning_move(
        board: Board,
        legal_moves: list[Move],
        player: int,
    ) -> Move | None:
        """兼容旧接口：返回第一个一步获胜点。"""
        moves = TacticalAI._find_winning_moves(
            board,
            legal_moves,
            player,
        )
        return moves[0] if moves else None

    @staticmethod
    def _get_nearby_moves(
        board: Board,
        legal_moves: list[Move],
        radius: int = 1,
    ) -> list[Move]:
        """只保留附近存在棋子的合法位置。"""
        return [
            move
            for move in legal_moves
            if TacticalAI._has_neighbor(
                board,
                move[0],
                move[1],
                radius,
            )
        ]

    @staticmethod
    def _has_neighbor(
        board: Board,
        row: int,
        column: int,
        radius: int = 1,
    ) -> bool:
        """判断指定位置附近是否已有棋子。"""
        for row_step in range(-radius, radius + 1):
            for column_step in range(-radius, radius + 1):
                if row_step == 0 and column_step == 0:
                    continue

                neighbor_row = row + row_step
                neighbor_column = column + column_step

                if (
                    board.is_inside(neighbor_row, neighbor_column)
                    and board.grid[neighbor_row][neighbor_column] != EMPTY
                ):
                    return True

        return False


class ScoringAI(TacticalAI):
    """结合一步战术、复合威胁和静态棋型评分选择落点。"""

    def __init__(
        self,
        player: int = WHITE,
        *,
        diagnostics: bool = False,
        top_n: int = 5,
    ) -> None:
        super().__init__(player=player)

        if top_n < 1:
            raise ValueError("top_n 必须大于 0。")

        self.diagnostics = diagnostics
        self.top_n = top_n
        self.last_analysis: DecisionAnalysis | None = None

    def _save_analysis(
        self,
        *,
        selected_move: Move,
        reason: str,
        candidate_count: int,
        top_candidates: list[CandidateAnalysis] | None = None,
    ) -> None:
        self.last_analysis = DecisionAnalysis(
            selected_move=selected_move,
            reason=reason,
            candidate_count=candidate_count,
            top_candidates=tuple((top_candidates or [])[: self.top_n]),
        )

    def _candidate_analysis(
        self,
        board: Board,
        move: Move,
        own_profile: ThreatProfile | None = None,
        opponent_profile: ThreatProfile | None = None,
    ) -> CandidateAnalysis:
        return CandidateAnalysis(
            move=move,
            score=evaluate_move(
                board,
                move[0],
                move[1],
                self.player,
            ),
            own_threat=(own_profile.label if own_profile else "普通"),
            opponent_threat=(
                opponent_profile.label
                if opponent_profile
                else "普通"
            ),
        )

    def choose_move(self, board: Board) -> Move:
        self.last_analysis = None
        legal_moves = board.get_legal_moves()

        if not legal_moves:
            raise ValueError("棋盘已满，电脑无法落子。")

        # 1. 自己能直接赢，立即取胜。
        own_wins = self._find_winning_moves(
            board,
            legal_moves,
            self.player,
        )
        if own_wins:
            selected = own_wins[0]
            top = (
                [self._candidate_analysis(board, move) for move in own_wins]
                if self.diagnostics
                else []
            )
            self._save_analysis(
                selected_move=selected,
                reason="立即五连",
                candidate_count=len(legal_moves),
                top_candidates=top,
            )
            return selected

        # 2. 对手已经存在一步胜点，必须先堵。
        opponent_wins = self._find_winning_moves(
            board,
            legal_moves,
            self.opponent,
        )
        if opponent_wins:
            selected = self._choose_emergency_block(
                board,
                opponent_wins,
            )
            top = (
                sorted(
                    [
                        self._candidate_analysis(board, move)
                        for move in opponent_wins
                    ],
                    key=lambda item: item.score,
                    reverse=True,
                )
                if self.diagnostics
                else []
            )
            reason = (
                "封堵唯一胜点"
                if len(opponent_wins) == 1
                else "对手多胜点：选择反击价值最高的封堵"
            )
            self._save_analysis(
                selected_move=selected,
                reason=reason,
                candidate_count=len(legal_moves),
                top_candidates=top,
            )
            return selected

        if not board.move_history:
            center = board.size // 2
            selected = (center, center)
            self._save_analysis(
                selected_move=selected,
                reason="空棋盘选择天元",
                candidate_count=len(legal_moves),
            )
            return selected

        nearby_moves = self._get_nearby_moves(
            board,
            legal_moves,
            radius=2,
        )
        candidates = nearby_moves if nearby_moves else legal_moves
        center = (board.size - 1) / 2

        own_profiles = self._profile_moves(
            board,
            candidates,
            self.player,
        )

        # 3. 自己能制造活四、双四、四三或双活三，优先发动叉攻。
        own_forcing = [
            (move, profile)
            for move, profile in own_profiles.items()
            if profile.forced_win
        ]
        if own_forcing:
            selected = self._best_profile_move(
                board,
                own_forcing,
                center,
                self.player,
            )
            selected_profile = own_profiles[selected]
            top = self._rank_profile_candidates(
                board,
                own_forcing,
                own_profiles,
                None,
            ) if self.diagnostics else []
            self._save_analysis(
                selected_move=selected,
                reason=f"主动制造{selected_profile.label}",
                candidate_count=len(candidates),
                top_candidates=top,
            )
            return selected

        opponent_profiles = self._profile_moves(
            board,
            candidates,
            self.opponent,
        )

        # 4. 对手下一手可制造强制胜势，抢先占住那个交叉点。
        opponent_forcing = [
            (move, profile)
            for move, profile in opponent_profiles.items()
            if profile.forced_win
        ]
        if opponent_forcing:
            selected = self._best_profile_move(
                board,
                opponent_forcing,
                center,
                self.opponent,
            )
            selected_profile = opponent_profiles[selected]
            top = self._rank_profile_candidates(
                board,
                opponent_forcing,
                own_profiles,
                opponent_profiles,
            ) if self.diagnostics else []
            self._save_analysis(
                selected_move=selected,
                reason=f"抢占对手{selected_profile.label}关键点",
                candidate_count=len(candidates),
                top_candidates=top,
            )
            return selected

        # 5. 没有强制战术时，统一计算一次分数并保留前 N 个候选。
        ranked: list[tuple[tuple[int, int, float, int, int], CandidateAnalysis]] = []

        for move in candidates:
            candidate = self._candidate_analysis(
                board,
                move,
                own_profiles[move],
                opponent_profiles[move],
            )
            sort_key = (
                candidate.score,
                own_profiles[move].tactical_rank,
                opponent_profiles[move].tactical_rank,
                -(
                    (move[0] - center) ** 2
                    + (move[1] - center) ** 2
                ),
                -move[0] * board.size - move[1],
            )
            ranked.append((sort_key, candidate))

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[0][1].move
        top = [item[1] for item in ranked[: self.top_n]]

        self._save_analysis(
            selected_move=selected,
            reason="综合棋型评分最高",
            candidate_count=len(candidates),
            top_candidates=top if self.diagnostics else [],
        )
        return selected

    def _rank_profile_candidates(
        self,
        board: Board,
        profiled_moves: list[tuple[Move, ThreatProfile]],
        own_profiles: dict[Move, ThreatProfile],
        opponent_profiles: dict[Move, ThreatProfile] | None,
    ) -> list[CandidateAnalysis]:
        candidates = [
            self._candidate_analysis(
                board,
                move,
                own_profiles.get(move),
                (
                    opponent_profiles.get(move)
                    if opponent_profiles is not None
                    else None
                ),
            )
            for move, _ in profiled_moves
        ]
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def _choose_emergency_block(
        self,
        board: Board,
        opponent_wins: list[Move],
    ) -> Move:
        """多个胜点已无法全堵时，选择兼顾自身反击价值的一点。"""
        return max(
            opponent_wins,
            key=lambda move: evaluate_move(
                board,
                move[0],
                move[1],
                self.player,
            ),
        )

    @staticmethod
    def _profile_moves(
        board: Board,
        candidates: list[Move],
        player: int,
    ) -> dict[Move, ThreatProfile]:
        return {
            move: analyze_move_threats(
                board,
                move[0],
                move[1],
                player,
            )
            for move in candidates
        }

    @staticmethod
    def _best_profile_move(
        board: Board,
        profiled_moves: list[tuple[Move, ThreatProfile]],
        center: float,
        player: int,
    ) -> Move:
        """按复合威胁级别、启发式分数和中心距离选最佳点。"""
        return max(
            profiled_moves,
            key=lambda item: (
                item[1].tactical_rank,
                len(item[1].winning_moves),
                evaluate_move(
                    board,
                    item[0][0],
                    item[0][1],
                    player,
                ),
                -(
                    (item[0][0] - center) ** 2
                    + (item[0][1] - center) ** 2
                ),
                -item[0][0],
                -item[0][1],
            ),
        )[0]
