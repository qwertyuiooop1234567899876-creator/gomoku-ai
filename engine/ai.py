import random

from engine.board import Board


class RandomAI:
    """从所有合法位置中随机选择一步的电脑玩家。"""

    def choose_move(self, board: Board) -> tuple[int, int]:
        """选择一个合法落子位置。"""
        legal_moves = board.get_legal_moves()

        if not legal_moves:
            raise ValueError("棋盘已满，电脑无法落子。")

        return random.choice(legal_moves)