from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from engine.board import BLACK, WHITE, Board
from engine.game import format_move
from engine.version import ENGINE_VERSION, RECORD_FORMAT_VERSION


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

    @staticmethod
    def _format_time_used_ratio(analysis: dict[str, Any]) -> str:
        ratio = analysis.get("time_used_ratio")
        if ratio is None:
            return "n/a"
        try:
            return f"{float(ratio) * 100:.1f}%"
        except (TypeError, ValueError):
            return "n/a"

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
                root_sources = move.analysis.get(
                    "root_candidate_sources",
                    [],
                )
                if root_sources:
                    lines.append(
                        "     root_sources="
                        + " | ".join(
                            f"{item.get('coordinate', '?')}:"
                            f"{','.join(item.get('sources', []))}"
                            for item in root_sources
                        )
                    )

                if move.analysis.get("engine_name") == "yixin":
                    bestline = move.analysis.get("bestline", [])
                    bestline_text = " -> ".join(
                        str(item) for item in bestline
                    )
                    lines.append(
                        "     "
                        f"yixin=eval:"
                        f"{move.analysis.get('evaluation', None)} "
                        f"white_eval:"
                        f"{move.analysis.get('evaluation_white', None)} "
                        f"depth:"
                        f"{move.analysis.get('search_depth', 0)}-"
                        f"{move.analysis.get('selective_depth', 0)} "
                        f"bestline:{bestline_text or '?'}"
                    )

                if move.analysis.get("vcf_found", False):
                    lines.append(
                        "     "
                        f"vcf=found depth:{move.analysis.get('vcf_depth', 0)} "
                        f"nodes:{move.analysis.get('vcf_nodes', 0):,}"
                    )

                if move.analysis.get("proof_checked", False):
                    lines.append(
                        "     "
                        f"proof={move.analysis.get('proof_state', 'unknown')} "
                        f"nodes:{move.analysis.get('proof_nodes', 0):,} "
                        f"elapsed:"
                        f"{move.analysis.get('proof_elapsed_seconds', 0.0):.3f}s "
                        f"best:{move.analysis.get('proof_best_coordinate', '?')} "
                        f"cutoff:"
                        f"{move.analysis.get('proof_cutoff_reason', None)}"
                    )
                    proof_tt_queries = move.analysis.get(
                        "proof_tt_queries",
                        0,
                    )
                    proof_tt_hits = move.analysis.get(
                        "proof_tt_hits",
                        0,
                    )
                    threat_cache_queries = move.analysis.get(
                        "threat_cache_queries",
                        0,
                    )
                    threat_cache_hits = move.analysis.get(
                        "threat_cache_hits",
                        0,
                    )
                    lines.append(
                        "     "
                        f"proof_tt=hits:{proof_tt_hits:,}/"
                        f"queries:{proof_tt_queries:,} "
                        f"compatible:"
                        f"{move.analysis.get('proof_tt_compatible_hits', 0):,} "
                        f"stores:{move.analysis.get('proof_tt_stores', 0):,} "
                        f"skipped:"
                        f"{move.analysis.get('proof_tt_skipped_stores', 0):,} "
                        f"size:{move.analysis.get('proof_tt_size', 0):,}; "
                        f"proof_hint=hits:"
                        f"{move.analysis.get('proof_hint_hits', 0):,}/"
                        f"queries:"
                        f"{move.analysis.get('proof_hint_queries', 0):,} "
                        f"stores:"
                        f"{move.analysis.get('proof_hint_stores', 0):,} "
                        f"size:"
                        f"{move.analysis.get('proof_hint_size', 0):,}; "
                        f"threat_cache=hits:{threat_cache_hits:,}/"
                        f"queries:{threat_cache_queries:,} "
                        f"stores:"
                        f"{move.analysis.get('threat_cache_stores', 0):,} "
                        f"skipped:"
                        f"{move.analysis.get('threat_cache_skips', 0):,}"
                    )
                    lines.append(
                        "     "
                        f"threat_work=candidate_batches:"
                        f"{move.analysis.get('threat_candidate_batches', 0):,} "
                        f"exact_descriptions:"
                        f"{move.analysis.get('threat_exact_descriptions', 0):,} "
                        f"frontier_batches:"
                        f"{move.analysis.get('threat_frontier_batches', 0):,} "
                        f"frontier_descriptions:"
                        f"{move.analysis.get('threat_frontier_descriptions', 0):,}"
                    )
                    for rank, candidate in enumerate(
                        move.analysis.get("proof_candidates", []),
                        start=1,
                    ):
                        pv = candidate.get("principal_variation", [])
                        pv_text = " -> ".join(
                            item.get("coordinate", "?")
                            for item in pv
                        )
                        suffix = f" pv={pv_text}" if pv_text else ""
                        lines.append(
                            "     "
                            f"proof#{rank} "
                            f"{candidate.get('coordinate', '?'):4} "
                            f"phase={candidate.get('phase', 'initial')} "
                            f"state={candidate.get('state', 'unknown')} "
                            f"complete={candidate.get('completed', False)} "
                            f"nodes={candidate.get('nodes', 0):,} "
                            f"risk={candidate.get('threat_risk', None)} "
                            f"cutoff={candidate.get('cutoff_reason', None)}"
                            f"{suffix}"
                        )

                    if move.analysis.get("final_proof_checked", False):
                        rejected = move.analysis.get(
                            "final_proof_rejected_moves",
                            [],
                        )
                        rejected_text = ",".join(
                            item.get("coordinate", "?")
                            for item in rejected
                        )
                        lines.append(
                            "     "
                            "final_proof="
                            f"{move.analysis.get('final_proof_state', 'unknown')} "
                            f"complete:"
                            f"{move.analysis.get('final_proof_completed', False)} "
                            f"selected:"
                            f"{move.analysis.get('final_proof_selected_coordinate', '?')} "
                            f"basis:"
                            f"{move.analysis.get('final_proof_selection_basis', 'not_checked')} "
                            f"rejected:{rejected_text or '?'}"
                        )

                if move.analysis.get("defense_vct_checked", False):
                    lines.append(
                        "     "
                        f"defense_vct=checked "
                        f"depth:{move.analysis.get('defense_vct_depth', 0)} "
                        f"nodes:{move.analysis.get('defense_vct_nodes', 0):,} "
                        f"best:{move.analysis.get('defense_vct_best_coordinate', '?')}"
                    )
                    for rank, candidate in enumerate(
                        move.analysis.get("defense_vct_candidates", []),
                        start=1,
                    ):
                        pv = candidate.get("principal_variation", [])
                        pv_text = " -> ".join(
                            item.get("coordinate", "?")
                            for item in pv
                        )
                        suffix = f" pv={pv_text}" if pv_text else ""
                        lines.append(
                            "     "
                            f"defense#{rank} "
                            f"{candidate.get('coordinate', '?'):4} "
                            f"status={candidate.get('status', 'unknown')} "
                            f"score={candidate.get('score', 0):+,}"
                            f"{suffix}"
                        )

                if move.analysis.get("root_safety_checked", False):
                    leaders = move.analysis.get(
                        "root_safety_leaders",
                        [],
                    )
                    leader_text = " -> ".join(
                        item.get("coordinate", "?")
                        for item in leaders
                    )
                    lines.append(
                        "     "
                        f"root_safety=checked "
                        f"applied:"
                        f"{move.analysis.get('root_safety_applied', False)} "
                        f"trigger:"
                        f"{move.analysis.get('root_safety_trigger', None)} "
                        f"pvs_gap:"
                        f"{move.analysis.get('root_safety_pvs_gap', None)} "
                        f"main_stable:"
                        f"{move.analysis.get('root_safety_main_rank_stable', True)} "
                        f"depth:"
                        f"{move.analysis.get('root_safety_depth', 0)} "
                        f"nodes:"
                        f"{move.analysis.get('root_safety_nodes', 0):,} "
                        f"arbitration:"
                        f"{move.analysis.get('review_arbitration_state', 'not_checked')} "
                        f"review_depth:"
                        f"{move.analysis.get('review_completed_depth', 0)} "
                        f"review_stable:"
                        f"{move.analysis.get('review_rank_stable', False)} "
                        f"boundary_tie:"
                        f"{move.analysis.get('review_boundary_tie_detected', False)} "
                        f"budget:"
                        f"{move.analysis.get('review_budget_seconds', 0.0):.3f}s "
                        f"escalation:"
                        f"{move.analysis.get('review_escalation_budget_seconds', 0.0):.3f}s "
                        f"best:"
                        f"{move.analysis.get('root_safety_best_coordinate', '?')} "
                        f"leaders:{leader_text or '?'}"
                    )
                    for rank, candidate in enumerate(
                        move.analysis.get(
                            "root_safety_candidates",
                            [],
                        ),
                        start=1,
                    ):
                        pv = candidate.get("principal_variation", [])
                        pv_text = " -> ".join(
                            item.get("coordinate", "?")
                            for item in pv
                        )
                        suffix = f" pv={pv_text}" if pv_text else ""
                        lines.append(
                            "     "
                            f"safety#{rank} "
                            f"{candidate.get('coordinate', '?'):4} "
                            f"score={candidate.get('score', 0):+,}"
                            f"{suffix}"
                        )
                    review_pairs = move.analysis.get(
                        "root_review_pairs",
                        [],
                    )
                    if review_pairs:
                        finalists = move.analysis.get(
                            "root_review_finalists",
                            [],
                        )
                        finalist_text = ",".join(
                            item.get("coordinate", "?")
                            for item in finalists
                        )
                        channels = ",".join(
                            pair.get("channel", "?")
                            for pair in review_pairs
                        )
                        lines.append(
                            "     "
                            f"root_review_audit=pairs:{len(review_pairs)} "
                            f"finalists:{finalist_text or '?'} "
                            f"channels:{channels}"
                        )

                if move.analysis.get("root_vcf_checked", False):
                    baseline = move.analysis.get(
                        "root_vcf_baseline_line",
                        [],
                    )
                    baseline_text = " -> ".join(
                        item.get("coordinate", "?")
                        for item in baseline
                    )
                    lines.append(
                        "     "
                        f"root_vcf=checked "
                        f"complete:"
                        f"{move.analysis.get('root_vcf_complete', False)} "
                        f"nodes:"
                        f"{move.analysis.get('root_vcf_nodes', 0):,} "
                        f"rescue_scan:"
                        f"{move.analysis.get('root_vcf_exhaustive_rescue_scanned', False)} "
                        f"rescue_checked:"
                        f"{move.analysis.get('root_vcf_rescue_candidates_checked', 0):,} "
                        f"baseline:{baseline_text or '?'}"
                    )
                    for rank, candidate in enumerate(
                        move.analysis.get("root_vcf_candidates", []),
                        start=1,
                    ):
                        pv = candidate.get("principal_variation", [])
                        pv_text = " -> ".join(
                            item.get("coordinate", "?")
                            for item in pv
                        )
                        suffix = f" pv={pv_text}" if pv_text else ""
                        lines.append(
                            "     "
                            f"root_vcf#{rank} "
                            f"{candidate.get('coordinate', '?'):4} "
                            f"status:"
                            f"{candidate.get('status', 'unknown')} "
                            f"complete:"
                            f"{candidate.get('completed', False)} "
                            f"nodes:{candidate.get('nodes', 0):,}"
                            f"{suffix}"
                        )

                search_depth = move.analysis.get("search_depth", 0)
                if search_depth > 0:
                    lines.append(
                        "     "
                        f"search=depth:{search_depth}/"
                        f"{move.analysis.get('requested_depth', search_depth)} "
                        f"interrupted:{move.analysis.get('interrupted_depth', 0)} "
                        f"nodes:{move.analysis.get('nodes', 0):,} "
                        f"nps:{move.analysis.get('nps', 0):,} "
                        f"cutoffs:{move.analysis.get('cutoffs', 0):,} "
                        f"tt_hits:{move.analysis.get('transposition_hits', 0):,} "
                        f"tt_cutoffs:{move.analysis.get('transposition_cutoffs', 0):,} "
                        f"tt_size:{move.analysis.get('transposition_size', 0):,} "
                        f"elapsed:{move.analysis.get('elapsed_seconds', 0.0):.3f}s "
                        f"completed:{move.analysis.get('search_completed', True)} "
                        f"stop:{move.analysis.get('stop_reason', 'unspecified')} "
                        f"time_used:{self._format_time_used_ratio(move.analysis)}"
                    )
                    lines.append(
                        "     "
                        f"ordering=killer:{move.analysis.get('killer_hits', 0):,} "
                        f"history:{move.analysis.get('history_hits', 0):,} "
                        f"extensions:{move.analysis.get('extensions', 0):,} "
                        f"pvs_research:{move.analysis.get('pvs_researches', 0):,} "
                        f"aspiration_research:"
                        f"{move.analysis.get('aspiration_researches', 0):,}"
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
            "format_version": RECORD_FORMAT_VERSION,
            "engine_version": ENGINE_VERSION,
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
            f"Engine version: V{ENGINE_VERSION}",
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
