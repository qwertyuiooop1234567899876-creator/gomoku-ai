import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from engine.board import BLACK, DIRECTIONS, EMPTY, WHITE, Board

Move = tuple[int, int]
Direction = tuple[int, int]

WIN_SCORE = 100_000_000
FORCED_WIN_SCORE = 50_000_000
DOUBLE_THREE_SCORE = 12_000_000
FOUR_SCORE = 1_000_000
OPEN_THREE_SCORE = 100_000
YIXIN_DECISIVE_SCORE = 10_000
YIXIN_DISPLAY_SCALE = 400.0


@dataclass(frozen=True, slots=True)
class PositionEvaluation:
    """供界面显示的独立局面评价，不参与 SearchAI 选点。"""

    source: str
    score_white: int | None
    raw_score: int | None = None
    depth: int = 0
    selective_depth: int = 0
    elapsed_seconds: float = 0.0
    best_move: Move | None = None
    bestline: tuple[str, ...] = ()
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.score_white is not None and self.error is None


# 只维护一个方向的基础棋型，镜像会自动生成。
_BASE_PATTERN_SCORES: tuple[tuple[str, int], ...] = (
    ("XXXXX", WIN_SCORE),       # 五连
    (".XXXX.", 10_000_000),    # 活四
    ("XXXX.", FOUR_SCORE),     # 冲四
    ("XXX.X", FOUR_SCORE),     # 跳四
    ("XX.XX", FOUR_SCORE),
    (".XXX.", OPEN_THREE_SCORE),
    (".XX.X.", 70_000),        # 跳三
    ("XXX..", 10_000),         # 眠三
    ("XX.X.", 8_000),
    (".XX.", 1_000),           # 活二
    (".X.X.", 800),            # 跳二
    ("XX...", 100),            # 眠二
)


def _build_pattern_scores() -> tuple[tuple[str, int], ...]:
    """自动加入镜像棋型，并按分数和长度从强到弱排序。"""
    scores: dict[str, int] = {}

    for pattern, score in _BASE_PATTERN_SCORES:
        scores[pattern] = max(score, scores.get(pattern, 0))

        mirrored = pattern[::-1]
        scores[mirrored] = max(score, scores.get(mirrored, 0))

    return tuple(
        sorted(
            scores.items(),
            key=lambda item: (-item[1], -len(item[0]), item[0]),
        )
    )


PATTERN_SCORES = _build_pattern_scores()


@dataclass(frozen=True, slots=True)
class ThreatProfile:
    """描述一手棋在四个方向同时制造出的复合威胁。"""

    immediate_win: bool = False
    open_four_directions: int = 0
    four_directions: int = 0
    open_three_directions: int = 0
    winning_moves: tuple[Move, ...] = ()

    @property
    def double_four(self) -> bool:
        return self.four_directions >= 2

    @property
    def four_three(self) -> bool:
        return (
            self.four_directions >= 1
            and self.open_three_directions >= 1
        )

    @property
    def double_three(self) -> bool:
        return self.open_three_directions >= 2

    @property
    def forced_win(self) -> bool:
        """自由五子棋下的强制胜势级威胁。"""
        return (
            self.immediate_win
            or self.open_four_directions >= 1
            or self.double_four
            or self.four_three
            or self.double_three
        )

    @property
    def tactical_rank(self) -> int:
        """供 AI 排序使用；数值越大，战术优先级越高。"""
        if self.immediate_win:
            return 100
        if self.double_four:
            return 95
        if self.open_four_directions >= 1:
            return 90
        if self.four_three:
            return 85
        if self.double_three:
            return 80
        if self.four_directions >= 1:
            return 60
        if self.open_three_directions >= 1:
            return 40
        return 0

    @property
    def label(self) -> str:
        if self.immediate_win:
            return "五连"
        if self.double_four:
            return "双四"
        if self.open_four_directions >= 1:
            return "活四"
        if self.four_three:
            return "四三"
        if self.double_three:
            return "双活三"
        if self.four_directions >= 1:
            return "冲四"
        if self.open_three_directions >= 1:
            return "活三"
        return "普通"


def other_side(player: int) -> int:
    """返回指定玩家的对手。"""
    if player == BLACK:
        return WHITE
    if player == WHITE:
        return BLACK
    raise ValueError("玩家只能是 BLACK 或 WHITE。")


def _line_to_text(line: list[int], player: int) -> str:
    """将棋盘线转换为 X、O、. 组成的模式字符串。"""
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
    """生成横、竖和两种斜线。"""
    for row in range(board.size):
        yield board.grid[row]

    for column in range(board.size):
        yield [board.grid[row][column] for row in range(board.size)]

    for start_column in range(board.size):
        line = _collect_line(board, 0, start_column, 1, 1)
        if len(line) >= 5:
            yield line

    for start_row in range(1, board.size):
        line = _collect_line(board, start_row, 0, 1, 1)
        if len(line) >= 5:
            yield line

    for start_column in range(board.size):
        line = _collect_line(board, 0, start_column, 1, -1)
        if len(line) >= 5:
            yield line

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


def _score_line(text: str) -> int:
    """强棋型优先且不让同一段字符被重复计分。"""
    occupied = [False] * len(text)
    total = 0

    for pattern, score in PATTERN_SCORES:
        start = 0

        while True:
            index = text.find(pattern, start)
            if index == -1:
                break

            end = index + len(pattern)

            if not any(occupied[index:end]):
                total += score
                for position in range(index, end):
                    occupied[position] = True

            start = index + 1

    return total


def evaluate_player(board: Board, player: int) -> int:
    """计算指定玩家在当前棋盘上的静态棋型总分。"""
    if player not in (BLACK, WHITE):
        raise ValueError("玩家只能是 BLACK 或 WHITE。")

    return sum(
        _score_line(_line_to_text(line, player))
        for line in _iter_lines(board)
    )


def evaluate_board(board: Board, perspective: int) -> int:
    """从指定玩家视角评价整个棋盘。"""
    opponent = other_side(perspective)
    return (
        evaluate_player(board, perspective)
        - evaluate_player(board, opponent)
    )


def _center_bonus(board: Board, row: int, column: int) -> int:
    """给靠近棋盘中心的位置少量奖励。"""
    center = (board.size - 1) / 2
    distance_squared = (
        (row - center) ** 2
        + (column - center) ** 2
    )
    return max(0, int(100 - distance_squared * 2))


def _direction_positions(
    board: Board,
    row: int,
    column: int,
    direction: Direction,
    radius: int = 4,
) -> Iterator[Move]:
    """生成同一直线上、可能参与五连的附近位置。"""
    row_step, column_step = direction

    for offset in range(-radius, radius + 1):
        if offset == 0:
            continue

        candidate = (
            row + offset * row_step,
            column + offset * column_step,
        )

        if board.is_inside(*candidate):
            yield candidate


def _winning_segment(
    board: Board,
    row: int,
    column: int,
    player: int,
    direction: Direction,
) -> set[Move]:
    """返回穿过指定棋子的连续同色线段。"""
    row_step, column_step = direction
    segment: set[Move] = {(row, column)}

    current_row = row + row_step
    current_column = column + column_step
    while (
        board.is_inside(current_row, current_column)
        and board.grid[current_row][current_column] == player
    ):
        segment.add((current_row, current_column))
        current_row += row_step
        current_column += column_step

    current_row = row - row_step
    current_column = column - column_step
    while (
        board.is_inside(current_row, current_column)
        and board.grid[current_row][current_column] == player
    ):
        segment.add((current_row, current_column))
        current_row -= row_step
        current_column -= column_step

    return segment


def _winning_moves_in_direction(
    board: Board,
    anchor: Move,
    player: int,
    direction: Direction,
) -> set[Move]:
    """寻找下一手能在指定方向形成、且包含 anchor 的五连点。"""
    winning_moves: set[Move] = set()

    for candidate in _direction_positions(
        board,
        anchor[0],
        anchor[1],
        direction,
    ):
        if not board.is_empty(*candidate):
            continue

        board.place(*candidate, player)

        try:
            segment = _winning_segment(
                board,
                candidate[0],
                candidate[1],
                player,
                direction,
            )

            if len(segment) >= 5 and anchor in segment:
                winning_moves.add(candidate)
        finally:
            board.undo()

    return winning_moves


def _creates_open_three_in_direction(
    board: Board,
    anchor: Move,
    player: int,
    direction: Direction,
) -> bool:
    """判断该方向是否存在一步发展成双胜点活四的走法。"""
    for extension in _direction_positions(
        board,
        anchor[0],
        anchor[1],
        direction,
    ):
        if not board.is_empty(*extension):
            continue

        board.place(*extension, player)

        try:
            winning_moves = _winning_moves_in_direction(
                board,
                anchor,
                player,
                direction,
            )

            if len(winning_moves) >= 2:
                return True
        finally:
            board.undo()

    return False


def _analyze_placed_move(
    board: Board,
    row: int,
    column: int,
    player: int,
) -> ThreatProfile:
    """分析已经临时放在棋盘上的一颗棋。"""
    anchor = (row, column)
    all_winning_moves: set[Move] = set()
    open_four_directions = 0
    four_directions = 0
    open_three_directions = 0

    for direction in DIRECTIONS:
        winning_moves = _winning_moves_in_direction(
            board,
            anchor,
            player,
            direction,
        )

        if winning_moves:
            four_directions += 1
            all_winning_moves.update(winning_moves)

            if len(winning_moves) >= 2:
                open_four_directions += 1

        elif _creates_open_three_in_direction(
            board,
            anchor,
            player,
            direction,
        ):
            open_three_directions += 1

    return ThreatProfile(
        immediate_win=board.check_win(row, column),
        open_four_directions=open_four_directions,
        four_directions=four_directions,
        open_three_directions=open_three_directions,
        winning_moves=tuple(sorted(all_winning_moves)),
    )


def analyze_move_threats(
    board: Board,
    row: int,
    column: int,
    player: int,
) -> ThreatProfile:
    """临时落子并统计这一子同时制造的四方向威胁。"""
    if player not in (BLACK, WHITE):
        raise ValueError("玩家只能是 BLACK 或 WHITE。")
    if not board.is_empty(row, column):
        raise ValueError("只能分析空位置。")

    board.place(row, column, player)

    try:
        return _analyze_placed_move(
            board,
            row,
            column,
            player,
        )
    finally:
        board.undo()


def find_winning_moves(
    board: Board,
    player: int,
    candidates: Sequence[Move] | None = None,
) -> list[Move]:
    """返回指定玩家当前所有一步获胜点，而不是只返回第一个。"""
    moves = board.get_legal_moves() if candidates is None else candidates
    winning_moves: list[Move] = []

    for row, column in moves:
        if not _could_be_winning_move(
            board,
            row,
            column,
            player,
        ):
            continue
        board.place(row, column, player)

        try:
            if board.check_win(row, column):
                winning_moves.append((row, column))
        finally:
            board.undo()

    return winning_moves


def _could_be_winning_move(
    board: Board,
    row: int,
    column: int,
    player: int,
) -> bool:
    """Cheap necessary condition before simulating a winning move."""
    for row_step, column_step in DIRECTIONS:
        friendly = 0
        for offset in range(-4, 5):
            if offset == 0:
                continue
            candidate_row = row + offset * row_step
            candidate_column = column + offset * column_step
            if (
                board.is_inside(candidate_row, candidate_column)
                and board.grid[candidate_row][candidate_column] == player
            ):
                friendly += 1
        if friendly >= 4:
            return True
    return False


def _profile_bonus(profile: ThreatProfile) -> int:
    if profile.immediate_win:
        return WIN_SCORE
    if profile.double_four:
        return FORCED_WIN_SCORE
    if profile.open_four_directions >= 1:
        return FORCED_WIN_SCORE - 1_000_000
    if profile.four_three:
        return FORCED_WIN_SCORE - 2_000_000
    if profile.double_three:
        return DOUBLE_THREE_SCORE

    return (
        profile.four_directions * FOUR_SCORE
        + profile.open_three_directions * OPEN_THREE_SCORE
    )


def evaluate_move(
    board: Board,
    row: int,
    column: int,
    player: int,
    defense_weight: float = 1.15,
    *,
    own_before: int | None = None,
    opponent_before: int | None = None,
    own_profile: ThreatProfile | None = None,
    opponent_profile: ThreatProfile | None = None,
) -> int:
    """综合静态增益、复合威胁、防守价值和中心位置评价落点。

    搜索器可以传入同一局面共享的基准分和已缓存威胁分析，避免
    对每个候选点重复计算；普通调用保持原有行为。
    """
    if not board.is_empty(row, column):
        raise ValueError("只能评价空位置。")

    opponent = other_side(player)
    if own_before is None:
        own_before = evaluate_player(board, player)
    if opponent_before is None:
        opponent_before = evaluate_player(board, opponent)

    if own_profile is None:
        own_profile = analyze_move_threats(
            board,
            row,
            column,
            player,
        )
    if opponent_profile is None:
        opponent_profile = analyze_move_threats(
            board,
            row,
            column,
            opponent,
        )

    board.place(row, column, player)
    try:
        own_after = evaluate_player(board, player)
    finally:
        board.undo()

    board.place(row, column, opponent)
    try:
        opponent_after = evaluate_player(board, opponent)
    finally:
        board.undo()

    attack_gain = max(0, own_after - own_before)
    defense_gain = max(0, opponent_after - opponent_before)

    return int(
        attack_gain
        + _profile_bonus(own_profile)
        + defense_weight * (
            defense_gain
            + _profile_bonus(opponent_profile)
        )
        + _center_bonus(board, row, column)
    )


def score_to_percentage(
    score: int,
    scale: float = 1_000_000,
) -> float:
    """把启发式分数压缩成局面倾向百分比；它不是校准胜率。"""
    if scale <= 0:
        raise ValueError("scale 必须大于 0。")

    percentage = 50.0 + 50.0 * math.tanh(score / scale)
    return max(0.0, min(100.0, percentage))


def yixin_score_to_percentage(
    score_white: int,
    *,
    scale: float = YIXIN_DISPLAY_SCALE,
) -> float:
    """把 YiXin 原始评价映射为形势条位置；结果不是胜率。"""
    if scale <= 0:
        raise ValueError("scale 必须大于 0。")
    if score_white >= YIXIN_DECISIVE_SCORE:
        return 100.0
    if score_white <= -YIXIN_DECISIVE_SCORE:
        return 0.0

    percentage = 50.0 + 50.0 * math.tanh(score_white / scale)
    return max(0.0, min(100.0, percentage))


def format_evaluation_score(score: int) -> str:
    """把白棋视角分数转换为可读文字。"""
    if score > 0:
        return f"白棋 +{score:,}"
    if score < 0:
        return f"黑棋 +{abs(score):,}"
    return "大致均衡"


def _player_text(player: int) -> str:
    return "黑棋" if player == BLACK else "白棋"


def _format_move(move: Move) -> str:
    """把内部坐标转换为棋盘坐标，例如 (7, 7) -> H8。"""
    row, column = move
    column_text = chr(ord("A") + column)
    return f"{column_text}{row + 1}"


def _signed_score_for_player(player: int, magnitude: int) -> int:
    """评分条统一采用白棋为正、黑棋为负。"""
    return magnitude if player == WHITE else -magnitude


def _infer_current_player(board: Board) -> int:
    """标准黑先白后下，根据已走步数推断当前行棋方。"""
    return BLACK if len(board.move_history) % 2 == 0 else WHITE


def _format_external_evaluation(
    evaluation: PositionEvaluation,
) -> str:
    if evaluation.error is not None:
        return f"{evaluation.source}评价不可用：{evaluation.error}"
    if evaluation.score_white is None:
        return f"{evaluation.source}没有返回有效评价"

    if evaluation.score_white >= YIXIN_DECISIVE_SCORE:
        advantage = "白棋已证明胜势"
    elif evaluation.score_white <= -YIXIN_DECISIVE_SCORE:
        advantage = "黑棋已证明胜势"
    elif evaluation.score_white > 0:
        advantage = f"白棋 +{evaluation.score_white:,}"
    elif evaluation.score_white < 0:
        advantage = f"黑棋 +{abs(evaluation.score_white):,}"
    else:
        advantage = "大致均衡"

    details: list[str] = []
    if evaluation.depth > 0:
        depth = str(evaluation.depth)
        if evaluation.selective_depth > 0:
            depth += f"-{evaluation.selective_depth}"
        details.append(f"深度 {depth}")
    if evaluation.elapsed_seconds > 0:
        details.append(f"{evaluation.elapsed_seconds:.3f}s")

    suffix = f"；{'，'.join(details)}" if details else ""
    return (
        f"{evaluation.source}：{advantage}"
        f"（原始评价{suffix}；形势条非胜率）"
    )


def render_evaluation_bar(
    board: Board,
    current_player: int | None = None,
    width: int = 24,
    *,
    position_evaluation: PositionEvaluation | None = None,
) -> str:
    """生成局面评分条；可由独立 YiXin 评价替代静态棋型分。"""
    if width < 10:
        raise ValueError("评分条宽度不能小于 10。")

    if current_player is None:
        current_player = _infer_current_player(board)

    if current_player not in (BLACK, WHITE):
        raise ValueError("current_player 必须是 BLACK 或 WHITE。")

    uses_external_score = False

    # 已经结束的棋局优先于所有静态棋型。
    if board.move_history:
        last_row, last_column, last_player = board.move_history[-1]
        if board.check_win(last_row, last_column):
            score = _signed_score_for_player(last_player, WIN_SCORE)
            status = f"{_player_text(last_player)}已获胜"
        elif board.is_full():
            score = 0
            status = "棋盘已满，平局"
        elif position_evaluation is not None:
            score = position_evaluation.score_white
            status = _format_external_evaluation(position_evaluation)
            uses_external_score = position_evaluation.available
        else:
            score = evaluate_board(board, WHITE)
            status = format_evaluation_score(score)
    else:
        if position_evaluation is not None:
            score = position_evaluation.score_white
            status = _format_external_evaluation(position_evaluation)
            uses_external_score = position_evaluation.available
        else:
            score = 0
            status = "大致均衡（空棋盘）"

    if not (
        board.move_history
        and board.check_win(
            board.move_history[-1][0],
            board.move_history[-1][1],
        )
    ):
        opponent = other_side(current_player)
        current_wins = find_winning_moves(board, current_player)
        opponent_wins = find_winning_moves(board, opponent)

        if current_wins:
            score = _signed_score_for_player(
                current_player,
                WIN_SCORE,
            )
            uses_external_score = False
            status = (
                f"{_player_text(current_player)}一步取胜 "
                f"({len(current_wins)} 个胜点)"
            )
        elif len(opponent_wins) >= 2:
            score = _signed_score_for_player(
                opponent,
                WIN_SCORE,
            )
            uses_external_score = False
            status = (
                f"{_player_text(opponent)}双胜点，当前方无法全堵"
            )
        elif len(opponent_wins) == 1:
            forced_block = opponent_wins[0]

            if position_evaluation is not None:
                status = (
                    f"{_player_text(current_player)}唯一应手："
                    f"{_format_move(forced_block)}；"
                    f"{_format_external_evaluation(position_evaluation)}"
                )
            else:
                # 兼容旧调用：没有外部评价时，继续试算唯一封堵。
                board.place(
                    forced_block[0],
                    forced_block[1],
                    current_player,
                )

                try:
                    score = evaluate_board(board, WHITE)
                finally:
                    board.undo()

                status = (
                    f"{_player_text(current_player)}唯一应手："
                    f"{_format_move(forced_block)}；以下为封堵后评价"
                )

    if score is None:
        black_text = white_text = "  -- "
        black_cells = width // 2
        white_cells = width - black_cells
    else:
        white_percentage = (
            yixin_score_to_percentage(score)
            if uses_external_score
            else score_to_percentage(score)
        )
        black_percentage = 100.0 - white_percentage
        black_text = f"{black_percentage:5.1f}%"
        white_text = f"{white_percentage:5.1f}%"

        black_cells = round(width * black_percentage / 100)
        black_cells = max(0, min(width, black_cells))
        white_cells = width - black_cells

    bar = "█" * black_cells + "░" * white_cells

    return (
        f"局面评价：{status}\n"
        f"黑 X {black_text} "
        f"[{bar}] "
        f"{white_text} 白 O"
    )
