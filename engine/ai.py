import random

from engine.board import BLACK, EMPTY, WHITE, Board
from engine.evaluator import evaluate_move


class RandomAI:
    """从所有合法位置中随机选择一步。"""

    def choose_move(self, board: Board) -> tuple[int, int]:
        legal_moves = board.get_legal_moves()

        if not legal_moves:
            raise ValueError("棋盘已满，电脑无法落子。")

        return random.choice(legal_moves)


class TacticalAI:
    """能够立即获胜、封堵对手，并优先靠近棋局落子。"""

    def __init__(self, player: int = WHITE) -> None:
        if player not in (BLACK, WHITE):
            raise ValueError(
                "AI 玩家只能是 BLACK 或 WHITE。"
            )

        self.player = player
        self.opponent = (
            WHITE
            if player == BLACK
            else BLACK
        )

    def choose_move(self, board: Board) -> tuple[int, int]:
        legal_moves = board.get_legal_moves()

        if not legal_moves:
            raise ValueError("棋盘已满，电脑无法落子。")

        winning_move = self._find_winning_move(
            board,
            legal_moves,
            self.player,
        )

        if winning_move is not None:
            return winning_move

        blocking_move = self._find_winning_move(
            board,
            legal_moves,
            self.opponent,
        )

        if blocking_move is not None:
            return blocking_move

        if not board.move_history:
            center = board.size // 2

            if board.is_empty(center, center):
                return center, center

        nearby_moves = self._get_nearby_moves(
            board,
            legal_moves,
        )

        candidates = (
            nearby_moves
            if nearby_moves
            else legal_moves
        )

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
    def _find_winning_move(
        board: Board,
        legal_moves: list[tuple[int, int]],
        player: int,
    ) -> tuple[int, int] | None:
        """寻找指定玩家是否存在一步获胜的位置。"""
        for row, column in legal_moves:
            board.place(
                row,
                column,
                player,
            )

            try:
                if board.check_win(row, column):
                    return row, column
            finally:
                board.undo()

        return None

    @staticmethod
    def _get_nearby_moves(
        board: Board,
        legal_moves: list[tuple[int, int]],
        radius: int = 1,
    ) -> list[tuple[int, int]]:
        """只保留附近存在棋子的合法位置。"""
        return [
            (row, column)
            for row, column in legal_moves
            if TacticalAI._has_neighbor(
                board,
                row,
                column,
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
        for row_step in range(
            -radius,
            radius + 1,
        ):
            for column_step in range(
                -radius,
                radius + 1,
            ):
                if (
                    row_step == 0
                    and column_step == 0
                ):
                    continue

                neighbor_row = row + row_step
                neighbor_column = (
                    column + column_step
                )

                if (
                    board.is_inside(
                        neighbor_row,
                        neighbor_column,
                    )
                    and board.grid[
                        neighbor_row
                    ][
                        neighbor_column
                    ] != EMPTY
                ):
                    return True

        return False


class ScoringAI(TacticalAI):
    """
    在一步胜负判断之外，使用棋型评分选择普通落点。
    """

    def choose_move(self, board: Board) -> tuple[int, int]:
        legal_moves = board.get_legal_moves()

        if not legal_moves:
            raise ValueError("棋盘已满，电脑无法落子。")

        # 第一层保险：自己能直接获胜。
        winning_move = self._find_winning_move(
            board,
            legal_moves,
            self.player,
        )

        if winning_move is not None:
            return winning_move

        # 第二层保险：对手能直接获胜，必须封堵。
        blocking_move = self._find_winning_move(
            board,
            legal_moves,
            self.opponent,
        )

        if blocking_move is not None:
            return blocking_move

        # AI 执黑且面对空棋盘时，选择天元。
        if not board.move_history:
            center = board.size // 2
            return center, center

        # 只搜索已有棋子两格范围内的位置。
        # 避免每回合都评价整张棋盘的所有空位。
        nearby_moves = self._get_nearby_moves(
            board,
            legal_moves,
            radius=2,
        )

        candidates = (
            nearby_moves
            if nearby_moves
            else legal_moves
        )

        center = (board.size - 1) / 2

        return max(
            candidates,
            key=lambda move: (
                evaluate_move(
                    board,
                    move[0],
                    move[1],
                    self.player,
                ),

                # 同分时优先靠近中心。
                -(
                    (move[0] - center) ** 2
                    + (move[1] - center) ** 2
                ),

                # 继续同分时保持结果稳定。
                -move[0],
                -move[1],
            ),
        )