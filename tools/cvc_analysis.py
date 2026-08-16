from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from engine.board import BLACK, WHITE, Board
from engine.game import format_move, other_player, parse_move
from engine.version import ENGINE_VERSION
from engine.yixin import (
    DEFAULT_YIXIN_SETTINGS_PATH,
    YixinConfig,
    YixinEngine,
    YixinError,
    YixinSearchReport,
    load_yixin_config,
    yixin_executable_sha256,
)


MATE_EVALUATION = 9_000


class PositionAnalyzer(Protocol):
    player: int
    last_report: YixinSearchReport | None

    def choose_move(self, board: Board) -> tuple[int, int]: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class PositionAssessment:
    ply: int
    side_to_move: int
    recommended_move: str | None
    completed_best_move: str | None
    evaluation_aligned: bool | None
    evaluation_raw: int | None
    evaluation_white: int | None
    depth: int
    selective_depth: int
    elapsed_ms: int
    nodes: int
    bestline: list[str]
    terminal: bool = False


@dataclass(slots=True)
class MoveAssessment:
    number: int
    player: int
    player_name: str
    actual_move: str
    recommended_move: str | None
    completed_best_move_before: str | None
    matches_recommendation: bool
    evaluation_aligned_before: bool | None
    evaluation_aligned_after: bool | None
    evaluation_before_white: int | None
    evaluation_after_white: int | None
    evaluation_after_recommended_white: int | None
    loss_for_mover: int | None
    classification: str
    comparison_basis: str
    paired_comparison_completed: bool
    bestline_before: list[str]
    recommended_child_bestline: list[str]
    depth_before: int
    selective_depth_before: int
    elapsed_ms_before: int
    recommended_child_depth: int
    recommended_child_selective_depth: int
    recommended_child_elapsed_ms: int


def load_cvc_record(path: str | Path) -> dict[str, Any]:
    record_path = Path(path)
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取 CVC JSON 棋谱：{error}") from error
    if not isinstance(payload, dict):
        raise ValueError("CVC JSON 顶层必须是对象。")
    moves = payload.get("moves")
    if not isinstance(moves, list) or not moves:
        raise ValueError("CVC JSON 中没有可分析的 moves。")
    return payload


def _move_from_payload(
    move_payload: Any,
    *,
    number: int,
    board: Board,
) -> tuple[int, int, int]:
    if not isinstance(move_payload, dict):
        raise ValueError(f"第 {number} 手不是有效对象。")
    try:
        row = int(move_payload["row"])
        column = int(move_payload["column"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"第 {number} 手缺少有效 row/column。"
        ) from error
    expected_player = BLACK if number % 2 == 1 else WHITE
    try:
        player = int(
            move_payload.get("player", expected_player)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"第 {number} 手 player 无效。") from error
    if player != expected_player:
        raise ValueError(
            f"第 {number} 手颜色顺序异常："
            f"应为 {expected_player}，实际为 {player}。"
        )
    if not board.is_inside(row, column):
        raise ValueError(f"第 {number} 手坐标越界。")
    if not board.is_empty(row, column):
        raise ValueError(f"第 {number} 手落在已有棋子的位置。")
    return row, column, player


def _evaluate_position(
    analyzer: PositionAnalyzer,
    board: Board,
    *,
    side_to_move: int,
    ply: int,
) -> PositionAssessment:
    analyzer.player = side_to_move
    analyzer.choose_move(board)
    report = analyzer.last_report
    if report is None:
        raise RuntimeError("YiXin 没有产生本局面分析。")
    evaluation_white = None
    if report.evaluation is not None:
        evaluation_white = (
            report.evaluation
            if side_to_move == WHITE
            else -report.evaluation
        )
    return PositionAssessment(
        ply=ply,
        side_to_move=side_to_move,
        recommended_move=report.coordinate,
        completed_best_move=report.completed_best_coordinate,
        evaluation_aligned=report.evaluation_aligned_with_move,
        evaluation_raw=report.evaluation,
        evaluation_white=evaluation_white,
        depth=report.depth,
        selective_depth=report.selective_depth,
        elapsed_ms=report.elapsed_ms,
        nodes=report.nodes,
        bestline=list(report.bestline),
    )


def _terminal_assessment(
    *,
    ply: int,
    side_to_move: int,
    winner: int | None,
) -> PositionAssessment:
    evaluation_white = None
    if winner == WHITE:
        evaluation_white = 10_000
    elif winner == BLACK:
        evaluation_white = -10_000
    elif winner is None:
        evaluation_white = 0
    return PositionAssessment(
        ply=ply,
        side_to_move=side_to_move,
        recommended_move=None,
        completed_best_move=None,
        evaluation_aligned=True,
        evaluation_raw=None,
        evaluation_white=evaluation_white,
        depth=0,
        selective_depth=0,
        elapsed_ms=0,
        nodes=0,
        bestline=[],
        terminal=True,
    )


def _loss_for_mover(
    before_white: int | None,
    after_white: int | None,
    player: int,
) -> int | None:
    if before_white is None or after_white is None:
        return None
    delta_white = after_white - before_white
    return delta_white if player == BLACK else -delta_white


def _is_decisive_loss(
    evaluation_white: int | None,
    player: int,
) -> bool:
    if evaluation_white is None:
        return False
    if player == BLACK:
        return evaluation_white >= MATE_EVALUATION
    return evaluation_white <= -MATE_EVALUATION


def classify_move(
    *,
    matches_recommendation: bool,
    before_white: int | None,
    after_white: int | None,
    player: int,
    recommended_after_white: int | None = None,
) -> tuple[str, int | None]:
    # A move that matches YiXin's returned recommendation cannot honestly be
    # blamed for a score jump between two independently searched positions.
    # Those adjacent searches may complete at different depths or terminate
    # early on a provisional mate score.
    if matches_recommendation:
        return "推荐一致", 0

    reference_white = (
        recommended_after_white
        if recommended_after_white is not None
        else before_white
    )
    loss = _loss_for_mover(reference_white, after_white, player)
    if (
        not _is_decisive_loss(reference_white, player)
        and _is_decisive_loss(after_white, player)
    ):
        return "直接败着", loss
    if loss is None:
        return "无法定量", None
    if loss >= 500:
        return "严重失误", loss
    if loss >= 150:
        return "明显失误", loss
    if loss >= 50:
        return "可疑着法", loss
    if loss > 0:
        return "轻微偏差", loss
    return "可接受变化", loss


def _assessment_after_move(
    analyzer: PositionAnalyzer,
    board: Board,
    *,
    row: int,
    column: int,
    player: int,
    ply: int,
) -> PositionAssessment:
    """Evaluate one already-placed child from the next side's perspective."""
    next_player = other_player(player)
    won = board.check_win(row, column)
    if won or board.is_full():
        return _terminal_assessment(
            ply=ply,
            side_to_move=next_player,
            winner=player if won else None,
        )
    return _evaluate_position(
        analyzer,
        board,
        side_to_move=next_player,
        ply=ply,
    )


def _recommended_child_assessment(
    analyzer: PositionAnalyzer,
    board: Board,
    *,
    recommendation: str | None,
    actual_coordinate: str,
    player: int,
    ply: int,
) -> PositionAssessment | None:
    """Search the recommended sibling before the actual child.

    Both children start from the same board and use the same analyzer
    settings.  The actual child is searched second so the replay can continue
    from the real game without leaving the external engine synchronized to a
    counterfactual branch.
    """
    if recommendation is None or recommendation == actual_coordinate:
        return None
    try:
        row, column = parse_move(recommendation, board.size)
    except ValueError:
        return None
    if not board.is_empty(row, column):
        return None

    board.place(row, column, player)
    try:
        return _assessment_after_move(
            analyzer,
            board,
            row=row,
            column=column,
            player=player,
            ply=ply,
        )
    finally:
        board.undo()


def analyze_cvc_payload(
    payload: dict[str, Any],
    analyzer: PositionAnalyzer,
    *,
    first_move: int = 1,
    last_move: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    moves_payload = payload.get("moves")
    if not isinstance(moves_payload, list) or not moves_payload:
        raise ValueError("CVC JSON 中没有可分析的 moves。")
    move_count = len(moves_payload)
    if not 1 <= first_move <= move_count:
        raise ValueError(
            f"first_move 必须在 1～{move_count} 之间。"
        )
    selected_last = move_count if last_move is None else last_move
    if not first_move <= selected_last <= move_count:
        raise ValueError(
            f"last_move 必须在 {first_move}～{move_count} 之间。"
        )

    board = Board()
    for number, move_payload in enumerate(
        moves_payload[: first_move - 1],
        start=1,
    ):
        row, column, player = _move_from_payload(
            move_payload,
            number=number,
            board=board,
        )
        board.place(row, column, player)
        if board.check_win(row, column):
            raise ValueError(
                f"棋局在第 {number} 手已结束，"
                "后续着法无法分析。"
            )

    side_to_move = BLACK if first_move % 2 == 1 else WHITE
    current = _evaluate_position(
        analyzer,
        board,
        side_to_move=side_to_move,
        ply=first_move - 1,
    )
    assessments: list[MoveAssessment] = []

    for number in range(first_move, selected_last + 1):
        move_payload = moves_payload[number - 1]
        row, column, player = _move_from_payload(
            move_payload,
            number=number,
            board=board,
        )
        actual_coordinate = format_move(row, column)
        if progress is not None:
            progress(
                f"分析第 {number}/{selected_last} 手："
                f"{actual_coordinate}"
            )

        recommended_child = _recommended_child_assessment(
            analyzer,
            board,
            recommendation=current.recommended_move,
            actual_coordinate=actual_coordinate,
            player=player,
            ply=number,
        )

        board.place(row, column, player)
        won = board.check_win(row, column)
        following = _assessment_after_move(
            analyzer,
            board,
            row=row,
            column=column,
            player=player,
            ply=number,
        )

        matches = current.recommended_move == actual_coordinate
        recommended_after_white = (
            following.evaluation_white
            if matches
            else (
                None
                if recommended_child is None
                else recommended_child.evaluation_white
            )
        )
        if not matches and recommended_child is None:
            classification, loss = "无法同根比较", None
        else:
            classification, loss = classify_move(
                matches_recommendation=matches,
                before_white=current.evaluation_white,
                after_white=following.evaluation_white,
                player=player,
                recommended_after_white=recommended_after_white,
            )
        paired_completed = matches or recommended_child is not None
        comparison_basis = (
            "same_recommended_move"
            if matches
            else (
                "paired_child_positions"
                if recommended_child is not None
                else "adjacent_position_fallback"
            )
        )
        assessment = MoveAssessment(
            number=number,
            player=player,
            player_name="BLACK" if player == BLACK else "WHITE",
            actual_move=actual_coordinate,
            recommended_move=current.recommended_move,
            completed_best_move_before=(
                current.completed_best_move
            ),
            matches_recommendation=matches,
            evaluation_aligned_before=(
                current.evaluation_aligned
            ),
            evaluation_aligned_after=(
                following.evaluation_aligned
            ),
            evaluation_before_white=current.evaluation_white,
            evaluation_after_white=following.evaluation_white,
            evaluation_after_recommended_white=(
                recommended_after_white
            ),
            loss_for_mover=loss,
            classification=classification,
            comparison_basis=comparison_basis,
            paired_comparison_completed=paired_completed,
            bestline_before=list(current.bestline),
            recommended_child_bestline=(
                []
                if recommended_child is None
                else list(recommended_child.bestline)
            ),
            depth_before=current.depth,
            selective_depth_before=current.selective_depth,
            elapsed_ms_before=current.elapsed_ms,
            recommended_child_depth=(
                0
                if recommended_child is None
                else recommended_child.depth
            ),
            recommended_child_selective_depth=(
                0
                if recommended_child is None
                else recommended_child.selective_depth
            ),
            recommended_child_elapsed_ms=(
                0
                if recommended_child is None
                else recommended_child.elapsed_ms
            ),
        )
        assessments.append(assessment)

        current = following
        if won or board.is_full():
            if number < selected_last:
                raise ValueError(
                    f"棋局在第 {number} 手已经结束，"
                    "但棋谱仍有后续着法。"
                )
            break

    first_decisive = next(
        (
            item
            for item in assessments
            if item.classification == "直接败着"
        ),
        None,
    )
    first_decisive_by_player = {
        player_name: next(
            (
                asdict(item)
                for item in assessments
                if item.player == player
                and item.classification == "直接败着"
            ),
            None,
        )
        for player_name, player in (
            ("BLACK", BLACK),
            ("WHITE", WHITE),
        )
    }

    sorted_errors = sorted(
        (
            item
            for item in assessments
            if item.loss_for_mover is not None
            and item.loss_for_mover > 0
        ),
        key=lambda item: item.loss_for_mover or 0,
        reverse=True,
    )
    return {
        "analysis_format_version": "1.4",
        "analyzer_engine_version": ENGINE_VERSION,
        "source": {
            "mode": payload.get("mode"),
            "black": payload.get("black"),
            "white": payload.get("white"),
            "result": payload.get("result"),
            "move_count": move_count,
        },
        "range": {
            "first_move": first_move,
            "last_move": selected_last,
        },
        "first_decisive_blunder": (
            asdict(first_decisive)
            if first_decisive is not None
            else None
        ),
        "first_decisive_blunder_by_player": (
            first_decisive_by_player
        ),
        "largest_losses": [
            asdict(item) for item in sorted_errors[:10]
        ],
        "moves": [asdict(item) for item in assessments],
    }


def _evaluation_text(value: int | None) -> str:
    return "?" if value is None else f"{value:+d}"


def render_analysis_text(
    result: dict[str, Any],
    *,
    source_path: Path,
    config: YixinConfig,
    executable_sha256: str | None,
) -> str:
    source = result["source"]
    lines = [
        "Gomoku CVC YiXin Analysis",
        f"Program version: V{ENGINE_VERSION}",
        f"Source: {source_path}",
        f"Black: {source.get('black')}",
        f"White: {source.get('white')}",
        f"Result: {source.get('result')}",
        (
            "YiXin settings: "
            f"time={config.timeout_turn_seconds:g}s, "
            f"threads={config.thread_num}, "
            f"split_depth={config.thread_split_depth}, "
            f"hash={config.hash_size}, "
            f"caution={config.caution_factor}, "
            f"checkmate={config.checkmate}"
        ),
        f"YiXin SHA256: {executable_sha256 or 'unavailable'}",
        "",
        "逐手分析（评价统一为白棋视角，正数利白、负数利黑）：",
        (
            "“推荐”是 YiXin 最终返回着。实战着与推荐着不同"
            "时，分别从同一根局面落下两着，再用相同 YiXin 设置"
            "搜索两个子局面；损失只比较这两个子局面，不再把"
            "相邻且深度不同的独立搜索直接相减。"
        ),
        "手数  方  实战  推荐  实战后/推荐后  损失  判定",
    ]
    for move in result["moves"]:
        after = _evaluation_text(
            move["evaluation_after_white"]
        )
        recommended_after = _evaluation_text(
            move.get("evaluation_after_recommended_white")
        )
        loss = move["loss_for_mover"]
        loss_text = "?" if loss is None else f"{loss:+d}"
        lines.append(
            f"{move['number']:3d}  "
            f"{'黑' if move['player'] == BLACK else '白'}  "
            f"{move['actual_move']:4}  "
            f"{(move['recommended_move'] or '?'):4}  "
            f"{after:>7}/{recommended_after:<7}  "
            f"{loss_text:>6}  "
            f"{move['classification']}"
        )
        bestline = move.get("bestline_before", [])
        if bestline:
            lines.append(
                "     YiXin线："
                + " → ".join(str(item) for item in bestline)
            )
        if move.get("evaluation_aligned_before") is False:
            lines.append(
                "     完成层与返回着不同：最终返回 "
                f"{move.get('recommended_move') or '?'}；"
                "完成层首选 "
                f"{move.get('completed_best_move_before') or '?'}；"
                "仍按完成层局面分数统计。"
            )
        if (
            move.get("comparison_basis")
            == "paired_child_positions"
        ):
            lines.append(
                "     同根对照：实战着后 "
                f"{after}；推荐着后 {recommended_after}；"
                f"推荐子局面 depth="
                f"{move.get('recommended_child_depth', 0)}/"
                f"{move.get('recommended_child_selective_depth', 0)}，"
                f"elapsed="
                f"{move.get('recommended_child_elapsed_ms', 0)}ms。"
            )

    decisive = result.get("first_decisive_blunder")
    unaligned_count = sum(
        1
        for move in result["moves"]
        if move.get("evaluation_aligned_before") is False
    )
    lines.extend(["", "关键结论："])
    if unaligned_count:
        lines.append(
            f"- 有 {unaligned_count} 个局面的最终返回着与完成层"
            "首选不一致；推荐着仍记录最终返回，损失使用完成层"
            "局面分数计算。"
        )
    if decisive is None:
        lines.append("- 本次分析范围内未发现评价直接进入必败带的一手。")
    else:
        lines.append(
            "- 首个断崖败着："
            f"第 {decisive['number']} 手 "
            f"{decisive['actual_move']}；"
            f"YiXin 推荐 {decisive['recommended_move'] or '?'}；"
            "同根子局面对照 "
            f"实战后 "
            f"{_evaluation_text(decisive['evaluation_after_white'])}；"
            "推荐后 "
            f"{_evaluation_text(decisive.get('evaluation_after_recommended_white'))}。"
        )

    decisive_by_player = result.get(
        "first_decisive_blunder_by_player",
        {},
    )
    for player_name, player_label in (
        ("BLACK", "黑方"),
        ("WHITE", "白方"),
    ):
        item = decisive_by_player.get(player_name)
        if item is None:
            continue
        lines.append(
            f"- {player_label}首个断崖败着："
            f"第 {item['number']} 手 {item['actual_move']}；"
            f"YiXin 推荐 {item['recommended_move'] or '?'}；"
            f"损失 {item['loss_for_mover']:+d}。"
        )

    largest_losses = result.get("largest_losses", [])
    if largest_losses:
        lines.append("- 最大评价损失：")
        for item in largest_losses[:5]:
            lines.append(
                "  "
                f"第 {item['number']} 手 {item['actual_move']}："
                f"{item['loss_for_mover']:+d}，"
                f"{item['classification']}。"
            )
    else:
        lines.append("- 没有可量化的正向评价损失。")
    lines.append("")
    return "\n".join(lines)


def save_analysis(
    result: dict[str, Any],
    *,
    source_path: str | Path,
    output_directory: str | Path,
    config: YixinConfig,
    executable_sha256: str | None,
) -> tuple[Path, Path]:
    source = Path(source_path)
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    base = output_dir / f"{source.stem}-yixin-analysis-{timestamp}"
    json_path = base.with_suffix(".json")
    txt_path = base.with_suffix(".txt")

    payload = dict(result)
    payload["generated_at"] = datetime.now().isoformat(
        timespec="seconds"
    )
    payload["source_path"] = str(source)
    payload["yixin_config"] = config.to_dict()
    payload["yixin_executable_sha256"] = executable_sha256
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    txt_path.write_text(
        render_analysis_text(
            result,
            source_path=source,
            config=config,
            executable_sha256=executable_sha256,
        ),
        encoding="utf-8",
    )
    return txt_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "用 YiXin 核心自动重放并分析 Gomoku CVC JSON 棋谱。"
        )
    )
    parser.add_argument("record", help="CVC JSON 棋谱路径。")
    parser.add_argument(
        "--settings",
        default=str(DEFAULT_YIXIN_SETTINGS_PATH),
        help="YiXin 设置 JSON。",
    )
    parser.add_argument(
        "--yixin-path",
        default=None,
        help="临时覆盖 YiXin executable_path。",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="每个局面的 YiXin 分析时间上限（秒）。",
    )
    parser.add_argument(
        "--from-move",
        type=int,
        default=1,
        help="从第几手开始输出分析。",
    )
    parser.add_argument(
        "--to-move",
        type=int,
        default=None,
        help="分析到第几手；默认到棋谱结束。",
    )
    parser.add_argument(
        "--output-dir",
        default="records/analysis",
        help="分析报告输出目录。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = Path(args.record)
    try:
        payload = load_cvc_record(source_path)
        config = load_yixin_config(args.settings)
        if args.yixin_path is not None:
            config = replace(
                config,
                executable_path=args.yixin_path,
            )
        if args.time_limit is not None:
            config = config.with_time_limit(args.time_limit)

        executable_sha256 = yixin_executable_sha256(config)
        analyzer = YixinEngine(
            player=BLACK,
            config=config,
        )
        try:
            result = analyze_cvc_payload(
                payload,
                analyzer,
                first_move=args.from_move,
                last_move=args.to_move,
                progress=print,
            )
        finally:
            analyzer.close()

        txt_path, json_path = save_analysis(
            result,
            source_path=source_path,
            output_directory=args.output_dir,
            config=config,
            executable_sha256=executable_sha256,
        )
    except (ValueError, OSError, RuntimeError, YixinError) as error:
        print(f"分析失败：{error}", file=sys.stderr)
        return 2

    print()
    print(f"TXT 分析：{txt_path}")
    print(f"JSON 分析：{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
