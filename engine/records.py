from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from engine.board import BLACK, WHITE, Board
from engine.game import format_move


@dataclass(slots=True)
class MoveRecord:
    """一手棋的可复盘信息。"""

    number: int
    player: int
    row: int
    column: int
    actor: str
    think_seconds: float
    evaluation_before: int
    evaluation_after: int
    analysis: dict[str, Any] | None = None

    @property
    def coordinate(self) -> str:
        return format_move(self.row, self.column)


@dataclass(slots=True)
class EventRecord:
    """悔棋、重开、退出等非落子事件。"""

    event_type: str
    detail: str
    move_count: int
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )


@dataclass(frozen=True, slots=True)
class RecordPaths:
    txt: Path
    json: Path


class GameRecorder:
    """统一记录 PVC/CVC 棋谱、评价变化和 AI 决策依据。"""

    def __init__(
        self,
        *,
        mode: str,
        black_name: str,
        white_name: str,
        record_dir: str | Path = "records",
    ) -> None:
        self.mode = mode
        self.black_name = black_name
        self.white_name = white_name
        self.record_dir = Path(record_dir)
        self.started_at = datetime.now()
        self.moves: list[MoveRecord] = []
        self.events: list[EventRecord] = []

    def record_move(
        self,
        *,
        player: int,
        row: int,
        column: int,
        actor: str,
        think_seconds: float,
        evaluation_before: int,
        evaluation_after: int,
        analysis: dict[str, Any] | None = None,
    ) -> MoveRecord:
        if player not in (BLACK, WHITE):
            raise ValueError("player 必须是 BLACK 或 WHITE。")
        if think_seconds < 0:
            raise ValueError("think_seconds 不能小于 0。")

        move = MoveRecord(
            number=len(self.moves) + 1,
            player=player,
            row=row,
            column=column,
            actor=actor,
            think_seconds=think_seconds,
            evaluation_before=evaluation_before,
            evaluation_after=evaluation_after,
            analysis=analysis,
        )
        self.moves.append(move)
        return move

    def add_event(self, event_type: str, detail: str) -> None:
        self.events.append(
            EventRecord(
                event_type=event_type,
                detail=detail,
                move_count=len(self.moves),
            )
        )

    def undo_last_moves(self, count: int = 2) -> list[MoveRecord]:
        if count < 1:
            raise ValueError("count 必须大于 0。")
        if len(self.moves) < count:
            raise ValueError("棋谱中没有足够的着法可撤销。")

        removed = self.moves[-count:]
        del self.moves[-count:]

        detail = "、".join(move.coordinate for move in removed)
        self.add_event("undo", f"撤销 {detail}")
        return removed

    def render_score_sheet(
        self,
        *,
        last_rounds: int = 8,
        full: bool = False,
    ) -> str:
        """以类似国际象棋记谱表的形式显示黑白着法。"""
        if not self.moves:
            return "着法记录：暂无"

        rounds: list[tuple[int, str, str]] = []

        for index in range(0, len(self.moves), 2):
            black_move = self.moves[index]
            white_move = (
                self.moves[index + 1]
                if index + 1 < len(self.moves)
                else None
            )
            rounds.append(
                (
                    index // 2 + 1,
                    black_move.coordinate,
                    white_move.coordinate if white_move else "—",
                )
            )

        if not full:
            if last_rounds < 1:
                raise ValueError("last_rounds 必须大于 0。")
            rounds = rounds[-last_rounds:]

        header = "完整着法记录：" if full else "着法记录（最近回合）："
        lines = [header, " 回合   黑 X   白 O"]
        lines.extend(
            f"{round_no:4}.  {black:4}   {white:4}"
            for round_no, black, white in rounds
        )
        return "\n".join(lines)

    def _move_to_dict(self, move: MoveRecord) -> dict[str, Any]:
        data = asdict(move)
        data["coordinate"] = move.coordinate
        data["player_name"] = "BLACK" if move.player == BLACK else "WHITE"
        return data

    def _render_move_details(self) -> list[str]:
        lines = ["Move details:"]

        for move in self.moves:
            score_delta = move.evaluation_after - move.evaluation_before
            lines.append(
                f"{move.number:3}. "
                f"{'黑 X' if move.player == BLACK else '白 O':4} "
                f"{move.coordinate:4} "
                f"actor={move.actor} "
                f"think={move.think_seconds:.3f}s "
                f"eval={move.evaluation_before:+,}->{move.evaluation_after:+,} "
                f"delta={score_delta:+,}"
            )

            if move.analysis:
                reason = move.analysis.get("reason", "未说明")
                candidate_count = move.analysis.get("candidate_count", 0)
                lines.append(
                    f"     reason={reason}; candidates={candidate_count}"
                )

                search_depth = move.analysis.get("search_depth", 0)
                if search_depth > 0:
                    lines.append(
                        "     "
                        f"search=depth:{search_depth} "
                        f"nodes:{move.analysis.get('nodes', 0):,} "
                        f"cutoffs:{move.analysis.get('cutoffs', 0):,} "
                        f"tt_hits:{move.analysis.get('transposition_hits', 0):,} "
                        f"elapsed:{move.analysis.get('elapsed_seconds', 0.0):.3f}s "
                        f"completed:{move.analysis.get('search_completed', True)}"
                    )

                    principal_variation = move.analysis.get(
                        "principal_variation",
                        [],
                    )
                    if principal_variation:
                        pv_text = " -> ".join(
                            item.get("coordinate", "?")
                            for item in principal_variation
                        )
                        lines.append(f"     pv={pv_text}")

                top_candidates = move.analysis.get("top_candidates", [])
                for rank, candidate in enumerate(top_candidates, start=1):
                    lines.append(
                        "     "
                        f"#{rank} {candidate.get('coordinate', '?'):4} "
                        f"score={candidate.get('score', 0):+,} "
                        f"own={candidate.get('own_threat', '普通')} "
                        f"opp={candidate.get('opponent_threat', '普通')}"
                    )

        return lines

    def save(
        self,
        *,
        board: Board,
        result: str,
        duration_seconds: float,
        prefix: str | None = None,
    ) -> RecordPaths:
        if duration_seconds < 0:
            raise ValueError("duration_seconds 不能小于 0。")

        self.record_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        safe_prefix = prefix or self.mode.lower().replace(" ", "-")
        base = self.record_dir / f"{safe_prefix}-{timestamp}"
        txt_path = base.with_suffix(".txt")
        json_path = base.with_suffix(".json")

        finished_at = datetime.now()
        payload = {
            "format_version": "1.0",
            "engine_version": "0.7.2",
            "mode": self.mode,
            "black": self.black_name,
            "white": self.white_name,
            "result": result,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_seconds": round(duration_seconds, 6),
            "move_count": len(self.moves),
            "moves": [self._move_to_dict(move) for move in self.moves],
            "events": [asdict(event) for event in self.events],
            "final_grid": board.grid,
            "final_board": str(board),
        }

        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        txt_lines = [
            "Gomoku AI Record",
            "Engine version: V0.7.2",
            f"Mode: {self.mode}",
            f"Black: {self.black_name}",
            f"White: {self.white_name}",
            f"Result: {result}",
            f"Moves: {len(self.moves)}",
            f"Duration: {duration_seconds:.3f}s",
            "",
            self.render_score_sheet(full=True),
            "",
            *self._render_move_details(),
            "",
            "Events:",
        ]

        if self.events:
            txt_lines.extend(
                f"- [{event.created_at}] {event.event_type}: {event.detail}"
                for event in self.events
            )
        else:
            txt_lines.append("- 无")

        txt_lines.extend(
            [
                "",
                "Final board:",
                str(board),
                "",
            ]
        )

        txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
        return RecordPaths(txt=txt_path, json=json_path)
