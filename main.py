from __future__ import annotations

import time

from engine.board import BLACK, WHITE, Board
from engine.evaluator import evaluate_board, render_evaluation_bar
from engine.game import (
    format_move,
    other_player,
    parse_move,
    player_name,
)
from engine.records import GameRecorder, RecordPaths
from engine.search import SearchAI
from engine.settings import (
    SearchSettings,
    load_search_settings,
    save_search_settings,
)

ENGINE_VERSION = "0.7.2"


def choose_human_player() -> int:
    """让玩家选择先后手；直接回车表示玩家执白、AI 先下。"""
    while True:
        choice = input(
            "请选择执棋：B 黑棋先手，W 白棋后手，"
            "直接回车由 AI 执黑先下："
        ).strip().upper()

        if choice in {"B", "BLACK", "X", "黑"}:
            return BLACK

        if choice in {"", "W", "WHITE", "O", "白"}:
            return WHITE

        print("输入无效：请输入 B、W，或直接回车。")


def choose_search_settings(
    settings_path: str = "search_settings.json",
) -> SearchSettings:
    """读取上次参数，并允许玩家在启动时修改。"""
    current = load_search_settings(settings_path)

    print(
        "当前搜索参数："
        f"depth={current.max_depth}，"
        f"time-limit={current.time_limit_seconds:g}s"
    )

    while True:
        raw_depth = input(
            "最大搜索深度 "
            f"[{current.max_depth}]（1～8，回车保持）："
        ).strip()

        try:
            max_depth = (
                current.max_depth
                if raw_depth == ""
                else int(raw_depth)
            )
            if not 1 <= max_depth <= 8:
                raise ValueError
            break
        except ValueError:
            print("输入无效：搜索深度必须是 1～8 的整数。")

    while True:
        raw_time_limit = input(
            "每步思考上限秒 "
            f"[{current.time_limit_seconds:g}]"
            "（0.1～60，回车保持）："
        ).strip()

        try:
            time_limit_seconds = (
                current.time_limit_seconds
                if raw_time_limit == ""
                else float(raw_time_limit)
            )
            if not 0.1 <= time_limit_seconds <= 60.0:
                raise ValueError
            break
        except ValueError:
            print("输入无效：时间限制必须在 0.1～60 秒之间。")

    selected = SearchSettings(
        max_depth=max_depth,
        time_limit_seconds=time_limit_seconds,
    )
    save_search_settings(
        selected,
        settings_path,
    )

    print(
        "本局搜索参数："
        f"depth={selected.max_depth}，"
        f"time-limit={selected.time_limit_seconds:g}s；"
        "已记忆为下次默认值。"
    )

    return selected


def create_computer(
    player: int,
    settings: SearchSettings | None = None,
) -> SearchAI:
    """按指定颜色和搜索参数创建 V0.7.2 搜索 AI。"""
    selected = settings or SearchSettings()

    return SearchAI(
        player=player,
        max_depth=selected.max_depth,
        time_limit_seconds=selected.time_limit_seconds,
        root_candidate_limit=12,
        branch_candidate_limit=8,
        threat_extension_depth=2,
        diagnostics=True,
        top_n=5,
    )


def create_recorder(human_player: int) -> GameRecorder:
    """按实际先后手创建棋谱记录器。"""
    if human_player == BLACK:
        black_name = "Human"
        white_name = "SearchAI"
    elif human_player == WHITE:
        black_name = "SearchAI"
        white_name = "Human"
    else:
        raise ValueError("human_player 必须是 BLACK 或 WHITE。")

    return GameRecorder(
        mode="PVC",
        black_name=black_name,
        white_name=white_name,
    )


def save_and_report(
    recorder: GameRecorder,
    board: Board,
    result: str,
    game_started: float,
) -> RecordPaths | None:
    if not recorder.moves:
        return None

    paths = recorder.save(
        board=board,
        result=result,
        duration_seconds=time.perf_counter() - game_started,
        prefix="pvc-v072",
    )
    print(f"TXT 棋谱：{paths.txt}")
    print(f"JSON 诊断：{paths.json}")
    return paths


def print_search_summary(computer: SearchAI) -> None:
    analysis = computer.last_analysis
    if analysis is None:
        return

    print(
        "电脑决策："
        f"{analysis.reason}；"
        f"候选点 {analysis.candidate_count} 个"
    )

    if analysis.search_depth > 0:
        completed_text = "完整" if analysis.search_completed else "限时截断"
        print(
            "搜索统计："
            f"深度 {analysis.search_depth}；"
            f"节点 {analysis.nodes:,}；"
            f"剪枝 {analysis.cutoffs:,}；"
            f"置换命中 {analysis.transposition_hits:,}；"
            f"{analysis.elapsed_seconds:.3f}s；"
            f"{completed_text}"
        )

    if analysis.principal_variation:
        pv = " → ".join(
            format_move(*move)
            for move in analysis.principal_variation
        )
        print(f"最佳变化：{pv}")


def main() -> None:
    search_settings = choose_search_settings()
    human_player = choose_human_player()
    computer_player = other_player(human_player)

    board = Board()
    computer = create_computer(
        computer_player,
        search_settings,
    )
    recorder = create_recorder(human_player)
    current_player = BLACK
    game_started = time.perf_counter()

    print(f"Gomoku Engine V{ENGINE_VERSION}")
    print(
        f"玩家执{player_name(human_player)}，"
        f"电脑执{player_name(computer_player)}。"
    )
    if computer_player == BLACK:
        print("电脑执黑棋，将自动先下。")
    print(
        "搜索参数："
        f"depth={search_settings.max_depth}，"
        f"time-limit={search_settings.time_limit_seconds:g}s。"
    )
    print("电脑使用复合威胁、Negamax、Alpha-Beta 和威胁延伸搜索。")
    print("输入 H 查看指令，U 悔棋，R 重开，M 棋谱，Q 退出。")

    while True:
        print()
        print(board)
        print()
        print(render_evaluation_bar(board, current_player))
        print()
        print(recorder.render_score_sheet(last_rounds=8))
        print()

        if current_player == human_player:
            input_started = time.perf_counter()
            raw_move = input(
                f"{player_name(current_player)}，请输入落子位置："
            )
            human_think_seconds = time.perf_counter() - input_started
            command = raw_move.strip().upper()

            if command == "Q":
                recorder.add_event("quit", "玩家主动退出")
                print("游戏已退出。")
                save_and_report(
                    recorder,
                    board,
                    "玩家退出",
                    game_started,
                )
                break

            if command == "H":
                print("可用指令：")
                print("  H8  在 H8 落子")
                print("  U   悔棋（撤销玩家和电脑各一手）")
                print("  R   保存当前棋谱并以相同颜色重新开局")
                print("  M   显示完整着法记录")
                print("  Q   保存棋谱并退出游戏")
                continue

            if command == "M":
                print()
                print(recorder.render_score_sheet(full=True))
                continue

            if command == "R":
                recorder.add_event("restart", "玩家要求重新开局")
                paths = save_and_report(
                    recorder,
                    board,
                    "重新开局",
                    game_started,
                )
                if paths is not None:
                    print("当前对局已保存。")

                board = Board()
                computer = create_computer(
                    computer_player,
                    search_settings,
                )
                recorder = create_recorder(human_player)
                current_player = BLACK
                game_started = time.perf_counter()
                print("棋盘已清空，以相同执棋颜色重新开局。")
                continue

            if command == "U":
                if len(board.move_history) < 2:
                    print("当前没有完整的一轮棋可以撤销。")
                    continue

                computer_move = board.undo()
                player_move = board.undo()

                if computer_move is None or player_move is None:
                    print("悔棋失败：落子历史不完整。")
                    continue

                recorder.undo_last_moves(2)

                computer_row, computer_column, computer_stone = computer_move
                player_row, player_column, player_stone = player_move

                # 正常情况下，最近一手属于电脑、前一手属于玩家。
                # 额外校验可尽早暴露轮次错乱。
                if (
                    computer_stone != computer_player
                    or player_stone != human_player
                ):
                    raise RuntimeError("悔棋时检测到玩家与电脑轮次不一致。")

                print(
                    "已悔棋：撤销玩家 "
                    f"{format_move(player_row, player_column)}，"
                    "以及电脑 "
                    f"{format_move(computer_row, computer_column)}。"
                )
                continue

            try:
                row, column = parse_move(
                    command,
                    board.size,
                )

                evaluation_before = evaluate_board(board, WHITE)
                board.place(
                    row,
                    column,
                    human_player,
                )
                evaluation_after = evaluate_board(board, WHITE)

                recorder.record_move(
                    player=human_player,
                    row=row,
                    column=column,
                    actor="Human",
                    think_seconds=human_think_seconds,
                    evaluation_before=evaluation_before,
                    evaluation_after=evaluation_after,
                )

            except ValueError as error:
                print(f"输入无效：{error}")
                continue

        else:
            evaluation_before = evaluate_board(board, WHITE)
            think_started = time.perf_counter()
            row, column = computer.choose_move(board)
            think_seconds = time.perf_counter() - think_started

            board.place(
                row,
                column,
                computer_player,
            )
            evaluation_after = evaluate_board(board, WHITE)
            analysis = (
                computer.last_analysis.to_dict()
                if computer.last_analysis is not None
                else None
            )

            recorder.record_move(
                player=computer_player,
                row=row,
                column=column,
                actor="SearchAI",
                think_seconds=think_seconds,
                evaluation_before=evaluation_before,
                evaluation_after=evaluation_after,
                analysis=analysis,
            )

            print(
                f"电脑{player_name(computer_player)}落子："
                f"{format_move(row, column)} "
                f"（{think_seconds:.3f}s）"
            )
            print_search_summary(computer)

        if board.check_win(row, column):
            print()
            print(board)
            print()
            print(render_evaluation_bar(board, current_player))
            print()
            result = f"{player_name(current_player)}获胜"
            print(f"{result}！")
            save_and_report(
                recorder,
                board,
                result,
                game_started,
            )
            break

        if board.is_full():
            print()
            print(board)
            print()
            print(render_evaluation_bar(board, current_player))
            print()
            print("棋盘已满，本局平局。")
            save_and_report(
                recorder,
                board,
                "平局",
                game_started,
            )
            break

        current_player = other_player(current_player)


if __name__ == "__main__":
    main()
