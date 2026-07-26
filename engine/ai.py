import random

from engine.board import BLACK, EMPTY, WHITE, Board


class RandomAI:
    """从所有合法位置中随机选择一步。"""

    def choose_move(self, board: Board) -> tuple[int, int]:
        legal_moves = board.get_legal_moves()

        if not legal_moves:
            raise ValueError("棋盘已满，电脑无法落子。")

        return random.choice(legal_moves)


class TacticalAI:
    """可以立即获胜、封堵对方，并优先靠近棋局落子的电脑玩家。"""

    def __init__(self, player: int = WHITE) -> None:
        if player not in (BLACK, WHITE):
            raise ValueError("AI 玩家只能是 BLACK 或 WHITE。")

        self.player = player
        self.opponent = WHITE if player == BLACK else BLACK

    def choose_move(self, board: Board) -> tuple[int, int]:
        """按照战术优先级选择一步。"""
        legal_moves = board.get_legal_moves()

        if not legal_moves:
            raise ValueError("棋盘已满，电脑无法落子。")

        # 第一优先级：AI 自己下一步能否获胜。
        winning_move = self._find_winning_move(
            board,
            legal_moves,
            self.player,
        )

        if winning_move is not None:
            return winning_move

        # 第二优先级：对手下一步能否获胜，需要立即封堵。
        blocking_move = self._find_winning_move(
            board,
            legal_moves,
            self.opponent,
        )

        if blocking_move is not None:
            return blocking_move

        # 空棋盘时优先落在天元。
        if not board.move_history:
            center = board.size // 2

            if board.is_empty(center, center):
                return center, center

        # 普通情况下，只考虑已有棋子附近的位置。
        nearby_moves = self._get_nearby_moves(
            board,
            legal_moves,
        )

        candidates = nearby_moves if nearby_moves else legal_moves

        # 在候选位置中选择最靠近中心的一个。
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
    ) -> list[tuple[int, int]]:
        """只保留周围一格内存在棋子的合法位置。"""
        return [
            (row, column)
            for row, column in legal_moves
            if TacticalAI._has_neighbor(
                board,
                row,
                column,
            )
        ]

    @staticmethod
    def _has_neighbor(
        board: Board,
        row: int,
        column: int,
        radius: int = 1,
    ) -> bool:
        """判断指定空位附近是否已有棋子。"""
        for row_step in range(-radius, radius + 1):
            for column_step in range(-radius, radius + 1):
                if row_step == 0 and column_step == 0:
                    continue

                neighbor_row = row + row_step
                neighbor_column = column + column_step

                if (
                    board.is_inside(
                        neighbor_row,
                        neighbor_column,
                    )
                    and board.grid[neighbor_row][neighbor_column] != EMPTY
                ):
                    return True

        return False