from engine.ai import ScoringAI
from engine.board import BLACK, WHITE, Board
from engine.evaluator import render_evaluation_bar
from engine.game import format_move


DIRECTIONS = {
    "横向": (0, 1),
    "纵向": (1, 0),
    "左上到右下": (1, 1),
    "右上到左下": (1, -1),
}


def create_open_three(
    player: int,
    row_step: int,
    column_step: int,
) -> Board:
    """在棋盘中央建立指定方向的活三。"""
    board = Board()

    center_row = 7
    center_column = 7

    for offset in (-1, 0, 1):
        board.place(
            center_row + offset * row_step,
            center_column + offset * column_step,
            player,
        )

    return board


def expected_endpoints(
    row_step: int,
    column_step: int,
) -> set[tuple[int, int]]:
    """返回活三两端的关键位置。"""
    center_row = 7
    center_column = 7

    return {
        (
            center_row - 2 * row_step,
            center_column - 2 * column_step,
        ),
        (
            center_row + 2 * row_step,
            center_column + 2 * column_step,
        ),
    }


def run_scenario(
    title: str,
    board: Board,
    ai: ScoringAI,
    expected_moves: set[tuple[int, int]],
) -> None:
    """运行并显示一个人工验证场景。"""
    move = ai.choose_move(board)
    result = "PASS" if move in expected_moves else "FAIL"

    print("=" * 52)
    print(title)
    print()
    print(board)
    print()
    print(render_evaluation_bar(board))
    print()
    print(f"电脑选择：{format_move(*move)}")
    print(
        "预期位置："
        + " 或 ".join(
            sorted(format_move(*item) for item in expected_moves)
        )
    )
    print(f"测试结果：[{result}]")
    print()


def main() -> None:
    ai = ScoringAI(player=WHITE)

    for direction_name, (
        row_step,
        column_step,
    ) in DIRECTIONS.items():
        attack_board = create_open_three(
            WHITE,
            row_step,
            column_step,
        )

        run_scenario(
            title=f"进攻测试：白棋{direction_name}活三",
            board=attack_board,
            ai=ai,
            expected_moves=expected_endpoints(
                row_step,
                column_step,
            ),
        )

        defense_board = create_open_three(
            BLACK,
            row_step,
            column_step,
        )

        run_scenario(
            title=f"防守测试：黑棋{direction_name}活三",
            board=defense_board,
            ai=ai,
            expected_moves=expected_endpoints(
                row_step,
                column_step,
            ),
        )


if __name__ == "__main__":
    main()