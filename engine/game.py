from engine.board import BLACK, WHITE


PLAYER_NAMES = {
    BLACK: "黑棋 X",
    WHITE: "白棋 O",
}


def parse_move(text: str, board_size: int = 15) -> tuple[int, int]:
    """
    将用户输入的棋盘坐标转换为内部坐标。

    例如：
    H8  -> (7, 7)
    A1  -> (0, 0)
    O15 -> (14, 14)
    """
    normalized = text.strip().replace(" ", "").upper()

    if len(normalized) < 2:
        raise ValueError("请输入“列字母 + 行号”，例如 H8。")

    column_text = normalized[0]
    row_text = normalized[1:]

    if not ("A" <= column_text <= "Z"):
        raise ValueError("列必须使用英文字母，例如 A、H、O。")

    if not row_text.isdigit():
        raise ValueError("行必须使用数字，例如 1、8、15。")

    column = ord(column_text) - ord("A")
    row = int(row_text) - 1

    if not (
        0 <= row < board_size
        and 0 <= column < board_size
    ):
        last_column = chr(ord("A") + board_size - 1)

        raise ValueError(
            f"坐标必须位于 A1 到 {last_column}{board_size} 之间。"
        )

    return row, column

def format_move(row: int, column: int) -> str:
    """将内部坐标转换为用户可读的棋盘坐标。"""
    column_text = chr(ord("A") + column)
    row_text = str(row + 1)

    return f"{column_text}{row_text}"

def other_player(player: int) -> int:
    """返回另一方玩家。"""
    if player == BLACK:
        return WHITE

    if player == WHITE:
        return BLACK

    raise ValueError("无效的玩家编号。")


def player_name(player: int) -> str:
    """返回适合显示的玩家名称。"""
    try:
        return PLAYER_NAMES[player]
    except KeyError as error:
        raise ValueError("无效的玩家编号。") from error