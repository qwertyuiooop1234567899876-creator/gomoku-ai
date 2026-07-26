import math
from collections.abc import Iterator

from engine.board import BLACK, EMPTY, WHITE, Board


# X：当前被评估一方的棋子
# O：对手棋子或棋盘边界
# .：空位
#
# 顺序大致代表棋型价值。当前属于第一版启发式权重，
# 后续会根据实战和自动对局继续调整。
PATTERN_SCORES: tuple[tuple[str, int], ...] = (
    ("XXXXX", 100_000_000),  # 五连
    (".XXXX.", 10_000_000),  # 活四

    ("XXXX.", 1_000_000),    # 冲四
    (".XXXX", 1_000_000),
    ("XXX.X", 1_000_000),    # 跳四
    ("XX.XX", 1_000_000),
    ("X.XXX", 1_000_000),

    (".XXX.", 100_000),      # 活三
    (".XX.X.", 70_000),      # 跳三
    (".X.XX.", 70_000),

    ("XXX..", 10_000),       # 眠三
    ("..XXX", 10_000),
    ("XX.X.", 8_000),
    (".X.XX", 8_000),
    ("X.XX.", 8_000),

    (".XX.", 1_000),         # 活二
    (".X.X.", 800),          # 跳二

    ("XX...", 100),          # 眠二
    ("...XX", 100),
)


def other_side(player: int) -> int:
    """返回指定玩家的对手。"""
    if player == BLACK:
        return WHITE

    if player == WHITE:
        return BLACK

    raise ValueError("玩家只能是 BLACK 或 WHITE。")


def _line_to_text(line: list[int], player: int) -> str:
    """
    将一条棋盘线转换为模式识别字符串。

    当前玩家棋子：X
    对手棋子：O
    空位：.
    边界：O
    """
    characters: list[str] = ["O"]

    for cell in line:
        if cell == player:
            characters.append("X")
        elif cell == EMPTY:
            characters.append(".")
        else:
            characters.append("O")

    characters.append("O")

    return "".join(characters)


def _collect_line(
    board: Board,
    start_row: int,
    start_column: int,
    row_step: int,
    column_step: int,
) -> list[int]:
    """从起点沿指定方向收集一整条棋盘线。"""
    line: list[int] = []

    row = start_row
    column = start_column

    while board.is_inside(row, column):
        line.append(board.grid[row][column])

        row += row_step
        column += column_step

    return line


def _iter_lines(board: Board) -> Iterator[list[int]]:
    """依次生成所有横线、竖线和两类斜线。"""

    # 横线
    for row in range(board.size):
        yield board.grid[row]

    # 竖线
    for column in range(board.size):
        yield [
            board.grid[row][column]
            for row in range(board.size)
        ]

    # 左上到右下：从第一行出发
    for start_column in range(board.size):
        line = _collect_line(
            board,
            0,
            start_column,
            1,
            1,
        )

        if len(line) >= 5:
            yield line

    # 左上到右下：从第一列出发
    # start_row 从 1 开始，避免重复左上角主对角线。
    for start_row in range(1, board.size):
        line = _collect_line(
            board,
            start_row,
            0,
            1,
            1,
        )

        if len(line) >= 5:
            yield line

    # 右上到左下：从第一行出发
    for start_column in range(board.size):
        line = _collect_line(
            board,
            0,
            start_column,
            1,
            -1,
        )

        if len(line) >= 5:
            yield line

    # 右上到左下：从最右列出发
    for start_row in range(1, board.size):
        line = _collect_line(
            board,
            start_row,
            board.size - 1,
            1,
            -1,
        )

        if len(line) >= 5:
            yield line


def _count_overlapping(text: str, pattern: str) -> int:
    """统计一个模式在字符串中的出现次数，允许模式重叠。"""
    count = 0
    start = 0

    while True:
        index = text.find(pattern, start)

        if index == -1:
            return count

        count += 1
        start = index + 1


def evaluate_player(board: Board, player: int) -> int:
    """计算指定玩家在当前棋盘上的棋型总分。"""
    if player not in (BLACK, WHITE):
        raise ValueError("玩家只能是 BLACK 或 WHITE。")

    total_score = 0

    for line in _iter_lines(board):
        text = _line_to_text(line, player)

        for pattern, score in PATTERN_SCORES:
            occurrences = _count_overlapping(
                text,
                pattern,
            )
            total_score += occurrences * score

    return total_score


def evaluate_board(board: Board, perspective: int) -> int:
    """
    从指定玩家视角评价整个棋盘。

    正数：指定玩家占优
    负数：对手占优
    零：静态评分大致均衡
    """
    opponent = other_side(perspective)

    own_score = evaluate_player(
        board,
        perspective,
    )
    opponent_score = evaluate_player(
        board,
        opponent,
    )

    return own_score - opponent_score


def _center_bonus(
    board: Board,
    row: int,
    column: int,
) -> int:
    """给靠近棋盘中心的位置少量奖励。"""
    center = (board.size - 1) / 2

    distance_squared = (
        (row - center) ** 2
        + (column - center) ** 2
    )

    return max(
        0,
        int(100 - distance_squared * 2),
    )


def evaluate_move(
    board: Board,
    row: int,
    column: int,
    player: int,
    defense_weight: float = 1.15,
) -> int:
    """
    对一个候选落点进行评分。

    评分组成：
    1. 自己下在这里后新增的棋型价值；
    2. 对手若下在这里可能形成的威胁；
    3. 少量中心位置奖励。
    """
    if not board.is_empty(row, column):
        raise ValueError("只能评价空位置。")

    opponent = other_side(player)

    own_before = evaluate_player(
        board,
        player,
    )
    opponent_before = evaluate_player(
        board,
        opponent,
    )

    board.place(
        row,
        column,
        player,
    )

    try:
        own_after = evaluate_player(
            board,
            player,
        )
    finally:
        board.undo()

    board.place(
        row,
        column,
        opponent,
    )

    try:
        opponent_after = evaluate_player(
            board,
            opponent,
        )
    finally:
        board.undo()

    attack_gain = max(
        0,
        own_after - own_before,
    )
    defense_gain = max(
        0,
        opponent_after - opponent_before,
    )

    return int(
        attack_gain
        + defense_gain * defense_weight
        + _center_bonus(board, row, column)
    )


def score_to_percentage(
    score: int,
    scale: float = 250_000,
) -> float:
    """
    将无边界的局面分数压缩到 0～100。

    这里返回的是“局面倾向百分比”，不是严格胜率。
    """
    if scale <= 0:
        raise ValueError("scale 必须大于 0。")

    percentage = (
        50.0
        + 50.0 * math.tanh(score / scale)
    )

    return max(
        0.0,
        min(100.0, percentage),
    )


def format_evaluation_score(score: int) -> str:
    """把白棋视角评分转换为可读文字。"""
    if score > 0:
        return f"白棋 +{score:,}"

    if score < 0:
        return f"黑棋 +{abs(score):,}"

    return "大致均衡"


def render_evaluation_bar(
    board: Board,
    width: int = 24,
) -> str:
    """生成终端版实时局面评分条。"""
    if width < 10:
        raise ValueError("评分条宽度不能小于 10。")

    # 统一从白棋角度计算：
    # 正数表示白棋占优，负数表示黑棋占优。
    score = evaluate_board(
        board,
        WHITE,
    )

    white_percentage = score_to_percentage(score)
    black_percentage = 100.0 - white_percentage

    black_cells = round(
        width * black_percentage / 100
    )
    black_cells = max(
        0,
        min(width, black_cells),
    )

    white_cells = width - black_cells

    bar = (
        "█" * black_cells
        + "░" * white_cells
    )

    return (
        f"局面评价：{format_evaluation_score(score)}\n"
        f"黑 X {black_percentage:5.1f}% "
        f"[{bar}] "
        f"{white_percentage:5.1f}% 白 O"
    )