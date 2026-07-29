import unittest

from engine.board import BLACK, WHITE, Board
from engine.evaluator import analyze_move_threats
from engine.game import format_move, parse_move
from engine.proof_search import ProofBudget, ProofSearch, ProofState
from engine.threats import (
    Threat,
    ThreatAnalyzer,
    ThreatCandidate,
    ThreatCandidateBatch,
    ThreatFrontier,
    ThreatKind,
)
from tests.test_v090_real_regressions import (
    I4_POSITION,
    board_state,
    build_position,
)


class TestExhaustiveFrontierObligations(unittest.TestCase):
    def test_quiet_frontier_keeps_every_legal_defender_reply(self) -> None:
        board = build_position(I4_POSITION)
        board.place(*parse_move("I4", board.size), BLACK)
        before = board_state(board)
        analyzer = ThreatAnalyzer()
        frontiers = analyzer.generate_attack_frontiers(
            board,
            WHITE,
            frontier_limit=64,
        )
        e7 = next(
            frontier
            for frontier in frontiers
            if format_move(*frontier.gain_move) == "E7"
        )

        obligation = analyzer.describe_frontier(
            board,
            e7,
            WHITE,
        )

        self.assertIs(ThreatKind.QUIET, obligation.kind)
        self.assertTrue(obligation.coverage_complete)
        self.assertEqual(
            board.empty_count - 1,
            len(obligation.required_defenses),
        )
        self.assertEqual(
            set(board.get_legal_moves()) - {e7.gain_move},
            set(obligation.required_defenses),
        )
        self.assertEqual(
            e7.continuations,
            obligation.frontier_continuations,
        )
        self.assertEqual(before, board_state(board))

    def test_low_budget_frontier_search_stays_unknown_and_restores(self) -> None:
        board = build_position(I4_POSITION)
        before = board_state(board)

        result = ProofSearch(
            budget=ProofBudget(
                max_nodes=1,
                max_attacker_moves=7,
                max_quiet_frontiers=64,
            )
        ).search_after_move(
            board,
            move=parse_move("I4", board.size),
            mover=BLACK,
            attacker=WHITE,
        )

        self.assertIs(ProofState.UNKNOWN, result.state)
        self.assertFalse(result.completed)
        self.assertEqual("node_limit", result.cutoff_reason)
        self.assertEqual(before, board_state(board))


class _SlicedCandidateAnalyzer(ThreatAnalyzer):
    def __init__(self, *, include_witness: bool) -> None:
        super().__init__(candidate_limit=2)
        self.first = (0, 0)
        self.second = (0, 1)
        self.include_witness = include_witness
        self.described: list[tuple[int, int]] = []

    def generate_attack_candidates(
        self,
        board: Board,
        player: int,
        *,
        stop_requested=None,  # type: ignore[no-untyped-def]
    ) -> ThreatCandidateBatch:
        candidates = [
            ThreatCandidate(
                move=self.first,
                profile=analyze_move_threats(
                    board,
                    *self.first,
                    player,
                ),
                kind=ThreatKind.QUIET,
            )
        ]
        if self.include_witness:
            candidates.append(
                ThreatCandidate(
                    move=self.second,
                    profile=analyze_move_threats(
                        board,
                        *self.second,
                        player,
                    ),
                    kind=ThreatKind.QUIET,
                )
            )
        return ThreatCandidateBatch(
            candidates=tuple(candidates),
            coverage_complete=False,
        )

    def describe_candidate(
        self,
        board: Board,
        candidate: ThreatCandidate,
        player: int,
        *,
        stop_requested=None,  # type: ignore[no-untyped-def]
    ) -> Threat:
        self.described.append(candidate.move)
        if candidate.move == self.first:
            defenses = tuple(
                move
                for move in board.get_legal_moves()
                if move != candidate.move
            )
            return Threat(
                gain_move=candidate.move,
                kind=ThreatKind.QUIET,
                attacker=player,
                required_defenses=defenses,
                coverage_complete=True,
                legal_reply_count=len(defenses),
            )
        return Threat(
            gain_move=candidate.move,
            kind=ThreatKind.QUIET,
            attacker=player,
            coverage_complete=True,
        )

    def generate_attack_frontiers(
        self,
        board: Board,
        player: int,
        *,
        frontier_limit: int = 12,
        continuation_limit: int = 12,
        stop_requested=None,  # type: ignore[no-untyped-def]
    ) -> tuple[ThreatFrontier, ...]:
        return ()


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _TimedCandidateAnalyzer(_SlicedCandidateAnalyzer):
    def __init__(self, clock: _FakeClock) -> None:
        super().__init__(include_witness=True)
        self.clock = clock

    def describe_candidate(
        self,
        board: Board,
        candidate: ThreatCandidate,
        player: int,
        *,
        stop_requested=None,  # type: ignore[no-untyped-def]
    ) -> Threat:
        if candidate.move == self.first:
            self.described.append(candidate.move)
            self.clock.advance(2.0)
            self.assert_stopped(stop_requested)
            return Threat(
                gain_move=candidate.move,
                kind=ThreatKind.QUIET,
                attacker=player,
                analysis_completed=False,
            )
        return super().describe_candidate(
            board,
            candidate,
            player,
            stop_requested=stop_requested,
        )

    @staticmethod
    def assert_stopped(stop_requested) -> None:  # type: ignore[no-untyped-def]
        if stop_requested is None or not stop_requested():
            raise AssertionError("候选时间截止没有传入威胁描述器。")


class TestCandidateBudgetScheduling(unittest.TestCase):
    def test_slice_moves_from_unknown_branch_to_winning_witness(self) -> None:
        board = Board(size=5)
        before = board_state(board)
        analyzer = _SlicedCandidateAnalyzer(include_witness=True)

        result = ProofSearch(
            budget=ProofBudget(
                max_nodes=20,
                max_attacker_moves=2,
                max_nodes_per_candidate=1,
            ),
            analyzer=analyzer,
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertIs(ProofState.PROVEN_WIN, result.state)
        self.assertEqual(analyzer.second, result.best_move)
        self.assertEqual(
            [analyzer.first, analyzer.second],
            analyzer.described,
        )
        self.assertEqual(before, board_state(board))

    def test_sliced_unknown_is_not_converted_to_proven_loss(self) -> None:
        board = Board(size=5)
        analyzer = _SlicedCandidateAnalyzer(include_witness=False)

        result = ProofSearch(
            budget=ProofBudget(
                max_nodes=20,
                max_attacker_moves=2,
                max_nodes_per_candidate=1,
            ),
            analyzer=analyzer,
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertIs(ProofState.UNKNOWN, result.state)
        self.assertFalse(result.completed)
        self.assertEqual(
            "candidate_node_limit",
            result.cutoff_reason,
        )

    def test_candidate_deadline_moves_to_next_witness(self) -> None:
        board = Board(size=5)
        before = board_state(board)
        clock = _FakeClock()
        analyzer = _TimedCandidateAnalyzer(clock)

        result = ProofSearch(
            budget=ProofBudget(
                max_nodes=20,
                max_attacker_moves=2,
                max_seconds_per_candidate=1.0,
            ),
            analyzer=analyzer,
            clock=clock,
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertIs(ProofState.PROVEN_WIN, result.state)
        self.assertEqual(analyzer.second, result.best_move)
        self.assertEqual(
            [analyzer.first, analyzer.second],
            analyzer.described,
        )
        self.assertEqual(before, board_state(board))

    def test_candidate_budget_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            ProofBudget(max_nodes_per_candidate=0)
        with self.assertRaises(ValueError):
            ProofBudget(max_seconds_per_candidate=0)


if __name__ == "__main__":
    unittest.main()
