from __future__ import annotations

from pathlib import Path
import random
import unittest

from engine.board import BLACK, DIRECTIONS, WHITE, Board
from engine.game import parse_move
from engine.native_core import (
    DEFENSE_CLASSIFICATION_COMPLETE,
    DEFENSE_CLASSIFICATION_CUTOFF,
    DEFENSE_CUTOFF_REPLY_LIMIT,
    DEFENSE_CUTOFF_TIMEOUT,
    native_core,
)
from engine.threats import DefenseSet, ThreatAnalyzer
from tools.native_defense_baseline import (
    DEFAULT_FIXTURES,
    board_state,
    run_benchmark,
)
from tools.vct_reference import build_board, load_case


POSITIONS = Path(__file__).resolve().parent / "positions"


class _StopAfterReplies:
    def __init__(self, completed_replies: int) -> None:
        self._remaining = completed_replies

    def __call__(self) -> bool:
        if self._remaining <= 0:
            return True
        self._remaining -= 1
        return False


@unittest.skipUnless(native_core.available, "NativeCore尚未编译")
class TestNativeDefenseClassification(unittest.TestCase):
    def _prepared_fixture(self, fixture_name: str, coordinate: str):
        case = load_case(POSITIONS / fixture_name)
        board = build_board(case)
        analyzer = ThreatAnalyzer(candidate_limit=24, frontier_scan_limit=48)
        move = parse_move(coordinate, board.size)
        threat = analyzer.describe_move(board, move, case.player)
        board.place(*move, case.player)
        return case, board, analyzer, threat

    def _python_result(
        self,
        analyzer: ThreatAnalyzer,
        board: Board,
        attacker: int,
        threat,
        *,
        completed_replies: int | None = None,
    ) -> DefenseSet:
        return analyzer._classify_defenses(
            board,
            attacker=attacker,
            continuations=threat.continuations,
            counter_wins=threat.counter_wins,
            stop_requested=(
                None
                if completed_replies is None
                else _StopAfterReplies(completed_replies)
            ),
        )

    def test_three_vct_fixtures_match_python_exactly(self) -> None:
        for fixture in DEFAULT_FIXTURES:
            case = load_case(fixture)
            board = build_board(case)
            analyzer = ThreatAnalyzer(
                candidate_limit=24,
                frontier_scan_limit=48,
            )
            initial = board_state(board)
            for coordinate in case.candidates:
                with self.subTest(fixture=fixture.name, move=coordinate):
                    move = parse_move(coordinate, board.size)
                    threat = analyzer.describe_move(
                        board,
                        move,
                        case.player,
                    )
                    self.assertEqual(initial, board_state(board))
                    board.place(*move, case.player)
                    placed = board_state(board)
                    reference = self._python_result(
                        analyzer,
                        board,
                        case.player,
                        threat,
                    )
                    native = native_core.classify_defenses(
                        board,
                        case.player,
                        threat.continuations,
                        threat.counter_wins,
                    )
                    assert native is not None
                    self.assertEqual(reference.signature, native.signature)
                    self.assertEqual(
                        DEFENSE_CLASSIFICATION_COMPLETE,
                        native.status,
                    )
                    self.assertEqual(placed, board_state(board))
                    board.undo()
                    self.assertEqual(initial, board_state(board))

    def test_reply_limit_matches_python_partial_classification(self) -> None:
        case, board, analyzer, threat = self._prepared_fixture(
            "v0175_selfplay_move24_vct.json",
            "J10",
        )
        before = board_state(board)
        reference = self._python_result(
            analyzer,
            board,
            case.player,
            threat,
            completed_replies=7,
        )
        native = native_core.classify_defenses(
            board,
            case.player,
            threat.continuations,
            threat.counter_wins,
            reply_limit=7,
        )
        assert native is not None
        self.assertEqual(reference.signature, native.signature)
        self.assertEqual(DEFENSE_CLASSIFICATION_CUTOFF, native.status)
        self.assertEqual(DEFENSE_CUTOFF_REPLY_LIMIT, native.cutoff_reason)
        self.assertEqual(7, native.processed_reply_count)
        self.assertEqual(before, board_state(board))

    def test_zero_timeout_matches_immediate_python_interruption(self) -> None:
        case, board, analyzer, threat = self._prepared_fixture(
            "v0175_selfplay_move24_vct.json",
            "J10",
        )
        before = board_state(board)
        reference = self._python_result(
            analyzer,
            board,
            case.player,
            threat,
            completed_replies=0,
        )
        native = native_core.classify_defenses(
            board,
            case.player,
            threat.continuations,
            threat.counter_wins,
            timeout_seconds=0.0,
        )
        assert native is not None
        self.assertEqual(reference.signature, native.signature)
        self.assertEqual(DEFENSE_CLASSIFICATION_CUTOFF, native.status)
        self.assertEqual(DEFENSE_CUTOFF_TIMEOUT, native.cutoff_reason)
        self.assertEqual(0, native.processed_reply_count)
        self.assertEqual(before, board_state(board))

    def test_random_planted_threats_match_python(self) -> None:
        generator = random.Random(20260902)
        bases = {
            (0, 1): (7, 4),
            (1, 0): (4, 7),
            (1, 1): (4, 4),
            (1, -1): (4, 10),
        }
        for sample in range(16):
            direction = DIRECTIONS[sample % len(DIRECTIONS)]
            row, column = bases[direction]
            row_step, column_step = direction
            attacker = BLACK if sample % 2 == 0 else WHITE
            defender = WHITE if attacker == BLACK else BLACK
            board = Board(15)
            protected = {
                (
                    row + offset * row_step,
                    column + offset * column_step,
                )
                for offset in range(-1, 5)
            }
            for offset in (0, 1):
                move = (
                    row + offset * row_step,
                    column + offset * column_step,
                )
                board.place(*move, attacker)
            noise = [
                move
                for move in board.get_legal_moves()
                if move not in protected
            ]
            generator.shuffle(noise)
            for index, move in enumerate(noise[: sample % 7]):
                board.place(*move, attacker if index % 2 else defender)
            gain = (
                row + 2 * row_step,
                column + 2 * column_step,
            )
            analyzer = ThreatAnalyzer(
                candidate_limit=24,
                frontier_scan_limit=48,
            )
            initial = board_state(board)
            threat = analyzer.describe_move(board, gain, attacker)
            self.assertTrue(threat.continuations)
            self.assertEqual(initial, board_state(board))
            board.place(*gain, attacker)
            placed = board_state(board)
            reference = self._python_result(
                analyzer,
                board,
                attacker,
                threat,
            )
            native = native_core.classify_defenses(
                board,
                attacker,
                threat.continuations,
                threat.counter_wins,
            )
            assert native is not None
            with self.subTest(sample=sample, direction=direction):
                self.assertEqual(reference.signature, native.signature)
                self.assertEqual(placed, board_state(board))
            board.undo()
            self.assertEqual(initial, board_state(board))

    def test_read_only_benchmark_reports_all_fixture_candidates(self) -> None:
        run = run_benchmark(repeats=1)
        self.assertEqual(6, len(run.candidates))
        self.assertTrue(all(item.equivalent for item in run.candidates))
        self.assertTrue(any(item.continuation_count for item in run.candidates))


if __name__ == "__main__":
    unittest.main()
