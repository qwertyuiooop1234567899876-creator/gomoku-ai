import time
import unittest

from engine.board import BLACK, WHITE, Board
from engine.evaluator import analyze_move_threats
from engine.game import parse_move
from engine.proof_search import (
    ProofBudget,
    ProofSearch,
    ProofState,
)
from engine.threats import (
    Threat,
    ThreatAnalyzer,
    ThreatCandidate,
    ThreatCandidateBatch,
    ThreatKind,
)
from tests.test_v090_real_regressions import (
    I4_POSITION,
    board_state,
    build_position,
)


class _ReplayAnalyzer(ThreatAnalyzer):
    def __init__(self) -> None:
        super().__init__(candidate_limit=1)
        self.root = (2, 2)
        self.follow = (1, 1)
        self.defenses = ((0, 0), (0, 1), (0, 2))
        self.candidate_generations = 0
        self.replay_descriptions = 0
        self.frontier_generations = 0

    def generate_attack_candidates(
        self,
        board: Board,
        player: int,
        *,
        stop_requested=None,  # type: ignore[no-untyped-def]
    ) -> ThreatCandidateBatch:
        self.candidate_generations += 1
        move = self.root if board.is_empty(*self.root) else self.follow
        return ThreatCandidateBatch(
            candidates=(
                ThreatCandidate(
                    move=move,
                    profile=analyze_move_threats(
                        board,
                        *move,
                        player,
                    ),
                    kind=ThreatKind.QUIET,
                ),
            ),
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
        if candidate.move == self.root:
            return Threat(
                gain_move=self.root,
                kind=ThreatKind.QUIET,
                attacker=player,
                required_defenses=self.defenses,
                coverage_complete=True,
                legal_reply_count=len(self.defenses),
            )
        return Threat(
            gain_move=self.follow,
            kind=ThreatKind.QUIET,
            attacker=player,
            coverage_complete=True,
        )

    def describe_move(
        self,
        board: Board,
        move: tuple[int, int],
        player: int,
        *,
        stop_requested=None,  # type: ignore[no-untyped-def]
    ) -> Threat:
        self.replay_descriptions += 1
        if move != self.follow:
            raise AssertionError("复用计划包含了未学习的着法。")
        return Threat(
            gain_move=self.follow,
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
    ) -> tuple[object, ...]:
        self.frontier_generations += 1
        return ()


class _CapturingProofSearch(ProofSearch):
    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self.quiet_obligations: list[Threat] = []

    def _search_and_node(self, board: Board, **kwargs):  # type: ignore[no-untyped-def]
        obligation = kwargs["obligation"]
        if (
            obligation is not None
            and obligation.kind is ThreatKind.QUIET
        ):
            self.quiet_obligations.append(obligation)
        return super()._search_and_node(board, **kwargs)


class TestV090ObligationMerging(unittest.TestCase):
    def test_linear_plan_is_revalidated_for_sibling_defenses(self) -> None:
        board = Board(size=5)
        before = board_state(board)
        analyzer = _ReplayAnalyzer()

        result = ProofSearch(
            budget=ProofBudget(
                max_nodes=50,
                max_attacker_moves=3,
                max_quiet_frontiers=8,
            ),
            analyzer=analyzer,
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertIs(ProofState.PROVEN_WIN, result.state)
        self.assertEqual(2, analyzer.candidate_generations)
        self.assertEqual(2, analyzer.replay_descriptions)
        self.assertEqual(0, analyzer.frontier_generations)
        self.assertEqual(before, board_state(board))

    def test_real_i4_plan_replays_on_another_legal_reply(self) -> None:
        board = build_position(I4_POSITION)
        for coordinate, player in (
            ("I4", BLACK),
            ("E7", WHITE),
            ("B1", BLACK),
        ):
            board.place(
                *parse_move(coordinate, board.size),
                player,
            )
        before = board_state(board)
        search = ProofSearch(
            budget=ProofBudget(
                max_nodes=100,
                max_attacker_moves=8,
            )
        )
        search._started_at = time.monotonic()
        plan = tuple(
            parse_move(coordinate, board.size)
            for coordinate in ("F5", "G5", "E4", "D3", "E6")
        )

        result = search._replay_linear_plan(
            board,
            attacker=WHITE,
            remaining_attacker_moves=8,
            plan=plan,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIs(ProofState.PROVEN_WIN, result.state)
        self.assertEqual(before, board_state(board))

    def test_forcing_proof_skips_quiet_frontier_generation(self) -> None:
        board = Board(size=5)
        analyzer = _ReplayAnalyzer()

        result = ProofSearch(
            budget=ProofBudget(
                max_nodes=50,
                max_attacker_moves=3,
                max_quiet_frontiers=8,
            ),
            analyzer=analyzer,
        ).search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertIs(ProofState.PROVEN_WIN, result.state)
        self.assertEqual(0, analyzer.frontier_generations)

    def test_forced_counter_block_keeps_all_following_replies(self) -> None:
        board = Board(size=5)
        for column in range(4):
            board.place(2, column, WHITE)
        before = board_state(board)
        search = _CapturingProofSearch(
            budget=ProofBudget(
                max_nodes=2,
                max_attacker_moves=2,
            )
        )

        result = search.search(
            board,
            attacker=BLACK,
            side_to_move=BLACK,
        )

        self.assertIs(ProofState.UNKNOWN, result.state)
        self.assertEqual(1, len(search.quiet_obligations))
        obligation = search.quiet_obligations[0]
        self.assertTrue(obligation.coverage_complete)
        self.assertEqual(
            board.empty_count - 1,
            len(obligation.required_defenses),
        )
        self.assertEqual(before, board_state(board))


if __name__ == "__main__":
    unittest.main()
