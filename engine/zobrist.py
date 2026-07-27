from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

BLACK = 1
WHITE = 2
MASK_64 = (1 << 64) - 1
_BASE_SEED = 0x5A17_2026_0800_CAFE


@dataclass(frozen=True, slots=True)
class ZobristTable:
    size: int
    pieces: tuple[tuple[tuple[int, int], ...], ...]
    side_to_move: tuple[int, int, int]

    def piece_key(
        self,
        row: int,
        column: int,
        player: int,
    ) -> int:
        if player not in (BLACK, WHITE):
            raise ValueError("player 必须是 BLACK 或 WHITE。")
        return self.pieces[row][column][player - 1]

    def side_key(self, player: int) -> int:
        if player not in (BLACK, WHITE):
            raise ValueError("player 必须是 BLACK 或 WHITE。")
        return self.side_to_move[player]


@lru_cache(maxsize=16)
def get_zobrist_table(size: int) -> ZobristTable:
    """按棋盘尺寸返回确定性的 64 位 Zobrist 随机表。"""
    if size < 5:
        raise ValueError("棋盘尺寸不能小于 5。")

    generator = random.Random((_BASE_SEED ^ (size << 17)) & MASK_64)
    pieces = tuple(
        tuple(
            (
                generator.getrandbits(64),
                generator.getrandbits(64),
            )
            for _ in range(size)
        )
        for _ in range(size)
    )
    side_to_move = (
        0,
        generator.getrandbits(64),
        generator.getrandbits(64),
    )
    return ZobristTable(
        size=size,
        pieces=pieces,
        side_to_move=side_to_move,
    )


def compute_grid_hash(grid: Sequence[Sequence[int]]) -> int:
    """从棋盘矩阵完整重算哈希，主要用于校验和测试。"""
    size = len(grid)
    table = get_zobrist_table(size)
    value = 0

    for row, cells in enumerate(grid):
        if len(cells) != size:
            raise ValueError("grid 必须是正方形棋盘。")
        for column, player in enumerate(cells):
            if player in (BLACK, WHITE):
                value ^= table.piece_key(row, column, player)
            elif player != 0:
                raise ValueError("棋盘只能包含 EMPTY、BLACK、WHITE。")

    return value & MASK_64
