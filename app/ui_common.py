from __future__ import annotations

from dataclasses import dataclass

from engine.arena_settings import AISelection
from engine.board import BLACK, Board


Move = tuple[int, int]


@dataclass(slots=True)
class ClickConfirmation:
    pending: Move | None = None

    def register(self, move: Move) -> bool:
        """Return True only when the same intersection is selected twice."""
        if self.pending == move:
            self.pending = None
            return True
        self.pending = move
        return False

    def cancel(self) -> None:
        self.pending = None


def normalized_ai_selection(
    engine_name: str,
    depth_value: float,
    time_value: float,
) -> AISelection:
    depth = min(8, max(1, int(round(depth_value))))
    time_limit = min(60.0, max(0.5, round(time_value * 2) / 2))
    return AISelection(
        engine_name=engine_name,
        max_depth=depth,
        time_limit_seconds=time_limit,
    )


def stone_name(player: int) -> str:
    return "黑棋 ●" if player == BLACK else "白棋 ○"


def clone_board(board: Board) -> Board:
    """Create a stable search snapshot without sharing mutable board state."""
    clone = Board(size=board.size)
    for row, column, player in board.move_history:
        clone.place(row, column, player)
    return clone
