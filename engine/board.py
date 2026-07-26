EMPTY = 0
BLACK = 1
WHITE = 2

SYMBOLS = {
    EMPTY: ".",
    BLACK: "X",
    WHITE: "O",
}


class Board:
    """表示一个五子棋棋盘。"""

    def __init__(self, size: int = 15) -> None:
        if size < 5:
            raise ValueError("棋盘尺寸不能小于 5。")

        self.size = size
        self.grid = [
            [EMPTY for _ in range(size)]
            for _ in range(size)
        ]
        self.move_history: list[tuple[int, int, int]] = []

    def is_inside(self, row: int, column: int) -> bool:
        """判断坐标是否位于棋盘内。"""
        return 0 <= row < self.size and 0 <= column < self.size

    def is_empty(self, row: int, column: int) -> bool:
        """判断指定位置是否为空。"""
        if not self.is_inside(row, column):
            return False

        return self.grid[row][column] == EMPTY

    def place(self, row: int, column: int, player: int) -> None:
        """在指定位置落下一颗棋子。"""
        if player not in (BLACK, WHITE):
            raise ValueError("玩家只能是 BLACK 或 WHITE。")

        if not self.is_inside(row, column):
            raise ValueError("落子位置超出棋盘范围。")

        if not self.is_empty(row, column):
            raise ValueError("该位置已经有棋子。")

        self.grid[row][column] = player
        self.move_history.append((row, column, player))

    def undo(self) -> tuple[int, int, int] | None:
        """撤销最近一步棋。"""
        if not self.move_history:
            return None

        row, column, player = self.move_history.pop()
        self.grid[row][column] = EMPTY

        return row, column, player

    def __str__(self) -> str:
        """生成适合在终端显示的棋盘文本。"""
        column_labels = " ".join(
            chr(ord("A") + column)
            for column in range(self.size)
        )

        lines = [f"   {column_labels}"]

        for row in range(self.size):
            symbols = " ".join(
                SYMBOLS[cell]
                for cell in self.grid[row]
            )
            lines.append(f"{row + 1:2} {symbols}")

        return "\n".join(lines)