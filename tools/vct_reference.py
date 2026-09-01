from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

from engine.board import BLACK, WHITE, Board
from engine.evaluator import other_side
from engine.game import format_move, parse_move
from engine.proof_search import ProofBudget, ProofSearch, ProofState, ProofTable
from engine.threats import ThreatAnalyzer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = (
    ROOT / "tests" / "positions" / "v0175_reverse_move10_vct.json"
)


@dataclass(frozen=True, slots=True)
class VCTReferenceCase:
    name: str
    board_size: int
    player: int
    history: tuple[str, ...]
    candidates: tuple[str, ...]
    expected_hash: int


@dataclass(frozen=True, slots=True)
class VCTCandidateResult:
    coordinate: str
    attacker_state: str
    completed: bool
    cutoff_reason: str | None
    best_coordinate: str | None
    principal_variation: tuple[str, ...]
    required_defenses: tuple[str, ...]
    nodes: int
    searched_attacker_moves: int
    elapsed_seconds: float
    threat_candidate_batches: int
    threat_exact_descriptions: int
    threat_frontier_batches: int
    threat_frontier_descriptions: int


@dataclass(frozen=True, slots=True)
class VCTReferenceRun:
    case_name: str
    player: int
    attacker: int
    candidates: tuple[VCTCandidateResult, ...]


def load_case(path: Path = DEFAULT_FIXTURE) -> VCTReferenceCase:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "gomoku-search-baseline-v1":
        raise ValueError(f"不支持的基线格式：{payload.get('format')!r}")
    player = int(payload["player"])
    if player not in (BLACK, WHITE):
        raise ValueError("基线行棋方必须是 BLACK 或 WHITE。")
    history = tuple(map(str, payload["history"]))
    expected_player = BLACK if len(history) % 2 == 0 else WHITE
    if player != expected_player:
        raise ValueError("基线行棋方与有序历史奇偶不一致。")
    candidates = tuple(map(str, payload["candidates"]))
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("VCT 基线候选不能为空或重复。")
    return VCTReferenceCase(
        name=str(payload["name"]),
        board_size=int(payload["board_size"]),
        player=player,
        history=history,
        candidates=candidates,
        expected_hash=int(payload["zobrist_hash"]),
    )


def build_board(case: VCTReferenceCase) -> Board:
    board = Board(case.board_size)
    for index, coordinate in enumerate(case.history):
        board.place(
            *parse_move(coordinate, board.size),
            BLACK if index % 2 == 0 else WHITE,
        )
    if board.zobrist_hash != case.expected_hash:
        raise RuntimeError(
            f"基线哈希不一致：{board.zobrist_hash} != {case.expected_hash}"
        )
    return board


def board_state(board: Board) -> tuple[object, ...]:
    return (
        tuple(tuple(row) for row in board.grid),
        tuple(board.move_history),
        board.zobrist_hash,
        board.empty_count,
    )


def run_reference(
    case: VCTReferenceCase,
    *,
    coordinates: tuple[str, ...] | None = None,
    seconds_per_candidate: float = 10.0,
    max_nodes: int = 100_000,
    max_attacker_moves: int = 6,
    max_quiet_frontiers: int = 16,
    max_quiet_attacker_moves: int = 2,
    vcf_max_attacker_moves: int = 6,
    candidate_limit: int = 24,
    frontier_scan_limit: int | None = 48,
) -> VCTReferenceRun:
    """Run isolated conservative proofs relative to the opponent attacker."""
    selected = case.candidates if coordinates is None else coordinates
    if not selected:
        raise ValueError("VCT 参考验证至少需要一个候选。")
    unknown = tuple(move for move in selected if move not in case.candidates)
    if unknown:
        raise ValueError(
            "VCT 参考候选必须来自夹具：" + ", ".join(unknown)
        )
    if seconds_per_candidate <= 0:
        raise ValueError("单候选秒数必须大于 0。")

    board = build_board(case)
    before = board_state(board)
    attacker = other_side(case.player)
    results: list[VCTCandidateResult] = []
    for coordinate in selected:
        move = parse_move(coordinate, board.size)
        if not board.is_empty(*move):
            raise ValueError(f"候选点不是空位：{coordinate}")
        analyzer = ThreatAnalyzer(
            candidate_limit=candidate_limit,
            frontier_scan_limit=frontier_scan_limit,
        )
        proof = ProofSearch(
            budget=ProofBudget.from_now(
                seconds_per_candidate,
                max_nodes=max_nodes,
                max_attacker_moves=max_attacker_moves,
                max_quiet_frontiers=max_quiet_frontiers,
                max_quiet_attacker_moves=max_quiet_attacker_moves,
                vcf_max_attacker_moves=vcf_max_attacker_moves,
                use_vcf_oracle=True,
                clock=time.perf_counter,
            ),
            analyzer=analyzer,
            table=ProofTable(),
            clock=time.perf_counter,
        ).search_after_move(
            board,
            move=move,
            mover=case.player,
            attacker=attacker,
            side_to_move=attacker,
        )
        if proof.completed == (proof.state is ProofState.UNKNOWN):
            raise RuntimeError("VCT 三态与完成标记不一致。")
        stats = analyzer.stats()
        results.append(
            VCTCandidateResult(
                coordinate=coordinate,
                attacker_state=proof.state.value,
                completed=proof.completed,
                cutoff_reason=proof.cutoff_reason,
                best_coordinate=(
                    None
                    if proof.best_move is None
                    else format_move(*proof.best_move)
                ),
                principal_variation=tuple(
                    format_move(*item) for item in proof.principal_variation
                ),
                required_defenses=tuple(
                    format_move(*item) for item in proof.required_defenses
                ),
                nodes=proof.nodes,
                searched_attacker_moves=proof.searched_attacker_moves,
                elapsed_seconds=proof.elapsed_seconds,
                threat_candidate_batches=stats.candidate_batches,
                threat_exact_descriptions=stats.exact_descriptions,
                threat_frontier_batches=stats.frontier_batches,
                threat_frontier_descriptions=stats.frontier_descriptions,
            )
        )
        if board_state(board) != before:
            raise RuntimeError("VCT 参考验证污染了棋盘或有序历史。")

    return VCTReferenceRun(
        case_name=case.name,
        player=case.player,
        attacker=attacker,
        candidates=tuple(results),
    )


def _player_name(player: int) -> str:
    return "BLACK" if player == BLACK else "WHITE"


def print_run(run: VCTReferenceRun) -> None:
    print(
        f"VCT reference | case={run.case_name} | "
        f"player={_player_name(run.player)} | "
        f"attacker={_player_name(run.attacker)}"
    )
    print(
        f"{'Candidate':<10} {'Attacker state':<15} {'Complete':<9} "
        f"{'Nodes':>10} {'Time':>9} {'Cutoff'}"
    )
    print("-" * 78)
    for candidate in run.candidates:
        print(
            f"{candidate.coordinate:<10} "
            f"{candidate.attacker_state:<15} "
            f"{str(candidate.completed):<9} "
            f"{candidate.nodes:>10,} "
            f"{candidate.elapsed_seconds:>8.3f}s "
            f"{candidate.cutoff_reason or '-'}"
        )
        if candidate.principal_variation:
            print("  PV: " + " ".join(candidate.principal_variation))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="独立、保守的候选后 VCT 三态参考验证器。"
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--max-nodes", type=int, default=100_000)
    parser.add_argument("--max-attacker-moves", type=int, default=6)
    parser.add_argument("--max-quiet-frontiers", type=int, default=16)
    parser.add_argument("--max-quiet-attacker-moves", type=int, default=2)
    parser.add_argument("--vcf-max-attacker-moves", type=int, default=6)
    parser.add_argument("--candidate-limit", type=int, default=24)
    parser.add_argument("--frontier-scan-limit", type=int, default=48)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.max_nodes < 0:
        parser.error("--max-nodes 不能小于 0")
    case = load_case(args.fixture)
    run = run_reference(
        case,
        coordinates=(
            None if args.candidate is None else tuple(args.candidate)
        ),
        seconds_per_candidate=args.seconds,
        max_nodes=args.max_nodes,
        max_attacker_moves=args.max_attacker_moves,
        max_quiet_frontiers=args.max_quiet_frontiers,
        max_quiet_attacker_moves=args.max_quiet_attacker_moves,
        vcf_max_attacker_moves=args.vcf_max_attacker_moves,
        candidate_limit=args.candidate_limit,
        frontier_scan_limit=args.frontier_scan_limit,
    )
    print_run(run)
    if args.json is not None:
        args.json.write_text(
            json.dumps(asdict(run), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
