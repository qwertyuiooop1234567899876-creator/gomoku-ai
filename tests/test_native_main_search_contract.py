from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
import random

from engine.board import BLACK, WHITE, Board
from engine.game import format_move, parse_move
from engine.native_core import (
    MAIN_SEARCH_FLAG_PVS,
    MAIN_SEARCH_FLAG_TT,
    STATUS_CUTOFF,
    STATUS_FOUND,
    native_core,
)
from engine.search import SearchAI
from engine.search_types import BoundType, INFINITY
from tools import native_search_baseline


MASK_64 = (1 << 64) - 1
FNV_OFFSET = 1_469_598_103_934_665_603
FNV_PRIME = 1_099_511_628_211
MOVE_21_FIXTURE = (
    Path(__file__).resolve().parent
    / "positions"
    / "v01618_yixin_move21.json"
)
MOVE_21_REVIEW_FIXTURE = (
    Path(__file__).resolve().parent
    / "positions"
    / "v0172_yixin_move21_native_review.json"
)


def _hash_u64(digest: int, value: int) -> int:
    value &= MASK_64
    for shift in range(0, 64, 8):
        digest ^= (value >> shift) & 0xFF
        digest = (digest * FNV_PRIME) & MASK_64
    return digest


class _NativeKeySearchAI(SearchAI):
    """Python oracle using the ABI's deterministic, portable TT key."""

    def _position_key(self, board, player):  # type: ignore[no-untyped-def]
        digest = _hash_u64(FNV_OFFSET, board.size)
        for index, cell in enumerate(
            (cell for row in board.grid for cell in row),
            start=1,
        ):
            digest = _hash_u64(digest, index)
            digest = _hash_u64(digest, cell)
        digest = _hash_u64(digest, player)
        recent = board.move_history[-self.config.recent_move_count :]
        digest = _hash_u64(digest, len(recent))
        for ordinal, (row, column, stone) in enumerate(recent, start=1):
            digest = _hash_u64(digest, ordinal)
            digest = _hash_u64(digest, row * board.size + column)
            digest = _hash_u64(digest, stone)
        return digest


def _native_tt_digest(table, board_size: int) -> int:  # type: ignore[no-untyped-def]
    bound_codes = {
        BoundType.EXACT: 0,
        BoundType.LOWER: 1,
        BoundType.UPPER: 2,
    }
    digest = FNV_OFFSET
    for key, entry in sorted(table.items()):
        values = (
            key,
            entry.depth,
            entry.extension_depth,
            entry.score,
            bound_codes[entry.bound],
            (
                0
                if entry.best_move is None
                else entry.best_move[0] * board_size
                + entry.best_move[1]
                + 1
            ),
            len(entry.principal_variation),
        )
        for value in values:
            digest = _hash_u64(digest, value)
        for row, column in entry.principal_variation:
            digest = _hash_u64(digest, row * board_size + column)
    return digest


def _python_native_key_oracle(
    case,
    coordinate: str,
    depth: int,
    *,
    use_pvs: bool = True,
):  # type: ignore[no-untyped-def]
    board = native_search_baseline.build_board(case)
    move = parse_move(coordinate, board.size)
    ai = _NativeKeySearchAI(
        case.player,
        max_depth=depth,
        time_limit_seconds=None,
        threat_extension_depth=2,
        branch_candidate_limit=8,
    )
    ai.config = replace(ai.config, use_pvs=use_pvs)
    ai._begin_move_search()
    result = ai._search_root(
        board,
        case.player,
        depth,
        [move],
        alpha=-INFINITY,
        beta=INFINITY,
    )
    return ai, result


class TestNativeMainSearchContract(unittest.TestCase):
    def test_coarse_abi_preserves_board_history_candidates_and_config(self) -> None:
        self.assertTrue(native_core.available, native_core.error)
        case = native_search_baseline.load_case()
        board = native_search_baseline.build_board(case)
        before = native_search_baseline.board_state(board)
        candidates = tuple(
            parse_move(coordinate, board.size)
            for coordinate in case.candidates
        )
        options = dict(
            depth=8,
            node_limit=25_000,
            branch_candidate_limit=8,
            preselection_factor=3,
            candidate_radius=2,
            recent_move_count=4,
            threat_extension_depth=2,
        )

        probe = native_core.probe_main_search_contract(
            board,
            case.player,
            candidates,
            use_pvs=True,
            use_transposition_table=True,
            **options,
        )

        expected_digest = native_core.main_search_input_digest(
            board,
            case.player,
            candidates,
            flags=MAIN_SEARCH_FLAG_PVS | MAIN_SEARCH_FLAG_TT,
            **options,
        )
        self.assertIsNotNone(probe)
        assert probe is not None
        self.assertEqual(STATUS_FOUND, probe.status)
        self.assertEqual(8, probe.completed_depth)
        self.assertEqual(expected_digest, probe.input_digest)
        self.assertIn(probe.best_move, candidates)
        self.assertEqual(len(candidates), len(probe.root_scores))
        self.assertGreater(probe.nodes, 0)
        self.assertGreater(probe.tt_entries, 0)
        self.assertTrue(probe.principal_variation)
        self.assertEqual(before, native_search_baseline.board_state(board))

    def test_ordered_history_is_part_of_the_contract_digest(self) -> None:
        case = native_search_baseline.load_case()
        board = native_search_baseline.build_board(case)
        candidates = tuple(
            parse_move(coordinate, board.size)
            for coordinate in case.candidates
        )
        options = dict(
            player=case.player,
            root_candidates=candidates,
            depth=8,
            node_limit=None,
            branch_candidate_limit=8,
            preselection_factor=3,
            candidate_radius=2,
            recent_move_count=4,
            threat_extension_depth=2,
            flags=MAIN_SEARCH_FLAG_PVS | MAIN_SEARCH_FLAG_TT,
        )
        original = native_core.main_search_input_digest(board, **options)
        board.move_history[-2:] = reversed(board.move_history[-2:])
        reordered = native_core.main_search_input_digest(board, **options)

        self.assertNotEqual(original, reordered)

    def test_python_oracle_locks_score_pv_nodes_and_tt_digest(self) -> None:
        case = native_search_baseline.load_case()
        expected = {
            ("F7", 1): (3700, ("F7", "J10", "F6"), 1, "f5e2174b086d72f52e1a50f6b18062e7355311dee43770c21d8f1164fccf819f"),
            ("J11", 1): (-8200, ("J11", "J10", "H6"), 1, "c7211d78b686c547f9aa18a2089aeb445e1394702261dc1b48d9de9d40c209b5"),
            ("F7", 2): (-999999996, ("F7", "F10", "H6", "G10"), 11, "38cac2d76e41127970314f715ffe7898e331fb945e52c0c38012e8bddf5d50c2"),
            ("J11", 2): (-999999996, ("J11", "J10", "H6", "G10"), 10, "de9e8313e06eaaf89822381211efde282c57025a7882cb29cba5f39cf0ce3e1e"),
            ("F7", 3): (3500, ("F7", "F10", "J10", "G10", "E10"), 37, "dfefea7e4b99cac8e58d2ff4df4ae6c97d698bdddcf746b6a9c08ca72567e579"),
            ("J11", 3): (800, ("J11", "J9", "H11", "F7", "E6"), 46, "5aeeacefd80f7ac8208818210ba73524ab5c6ae26ae6f0e8ae3c536c7ce22a3e"),
        }
        for (coordinate, depth), snapshot in expected.items():
            with self.subTest(coordinate=coordinate, depth=depth):
                run = native_search_baseline.run_full_window_candidate(
                    case,
                    coordinate,
                    depth,
                )
                self.assertEqual(snapshot, (
                    run.score,
                    run.principal_variation,
                    run.nodes,
                    run.tt_digest,
                ))

    def test_native_matches_python_score_pv_nodes_and_tt_contents(self) -> None:
        self.assertTrue(native_core.available, native_core.error)
        case = native_search_baseline.load_case()
        board = native_search_baseline.build_board(case)
        for coordinate in case.candidates:
            move = parse_move(coordinate, board.size)
            for depth in range(1, 4):
                with self.subTest(coordinate=coordinate, depth=depth):
                    ai, expected = _python_native_key_oracle(
                        case,
                        coordinate,
                        depth,
                    )
                    probe = native_core.probe_main_search_contract(
                        board,
                        case.player,
                        (move,),
                        depth=depth,
                        node_limit=None,
                        branch_candidate_limit=8,
                        preselection_factor=3,
                        candidate_radius=2,
                        recent_move_count=4,
                        threat_extension_depth=2,
                        use_pvs=True,
                        use_transposition_table=True,
                    )
                    self.assertIsNotNone(probe)
                    assert probe is not None
                    self.assertEqual(STATUS_FOUND, probe.status)
                    self.assertEqual(expected.move, probe.best_move)
                    self.assertEqual(expected.score, probe.score)
                    self.assertEqual(
                        expected.principal_variation,
                        probe.principal_variation,
                    )
                    self.assertEqual(ai._counters.nodes, probe.nodes)
                    self.assertEqual(
                        len(ai._transposition_table),
                        probe.tt_entries,
                    )
                    self.assertEqual(
                        _native_tt_digest(
                            ai._transposition_table,
                            board.size,
                        ),
                        probe.tt_digest,
                    )

    def test_native_review_matches_independent_python_oracles(self) -> None:
        self.assertTrue(native_core.available, native_core.error)
        case = native_search_baseline.load_case(MOVE_21_REVIEW_FIXTURE)
        review = native_search_baseline.run_native_full_window_review(
            case,
            (1, 2, 3),
            node_limit=None,
            threat_extension_depth=2,
            branch_candidate_limit=8,
        )

        expected_leaders: list[str] = []
        for layer in review.layers:
            expected_scores: list[tuple[str, int]] = []
            for candidate in layer.candidates:
                with self.subTest(
                    depth=layer.requested_depth,
                    candidate=candidate.coordinate,
                ):
                    standard = (
                        native_search_baseline.run_full_window_candidate(
                            case,
                            candidate.coordinate,
                            layer.requested_depth,
                            use_pvs=False,
                        )
                    )
                    ai, portable = _python_native_key_oracle(
                        case,
                        candidate.coordinate,
                        layer.requested_depth,
                        use_pvs=False,
                    )
                    self.assertEqual(standard.score, candidate.score)
                    self.assertEqual(
                        standard.principal_variation,
                        candidate.principal_variation,
                    )
                    self.assertEqual(portable.score, candidate.score)
                    self.assertEqual(
                        tuple(
                            format_move(*move)
                            for move in portable.principal_variation
                        ),
                        candidate.principal_variation,
                    )
                    self.assertEqual(ai._counters.nodes, candidate.nodes)
                    self.assertEqual(
                        len(ai._transposition_table),
                        candidate.tt_entries,
                    )
                    self.assertEqual(
                        f"{_native_tt_digest(ai._transposition_table, case.board_size):016x}",
                        candidate.tt_digest,
                    )
                    assert candidate.score is not None
                    expected_scores.append(
                        (candidate.coordinate, candidate.score)
                    )
            expected_leaders.append(
                max(
                    enumerate(expected_scores),
                    key=lambda item: (item[1][1], -item[0]),
                )[1][0]
            )
        self.assertEqual(tuple(expected_leaders), review.leader_history)

    def test_move21_reproduces_python_horizon_oscillation(self) -> None:
        case = native_search_baseline.load_case(MOVE_21_FIXTURE)
        board = native_search_baseline.build_board(case)
        candidates = tuple(
            parse_move(coordinate, board.size)
            for coordinate in case.candidates
        )
        expected = {
            6: ("K8", -11_100, (-874_900, -11_100), 1_242, 1_156),
            7: ("H7", 28_100, (28_100, 27_900), 4_636, 4_413),
            8: ("K8", -20_200, (-91_400, -20_200), 10_716, 10_334),
        }
        for depth, snapshot in expected.items():
            with self.subTest(depth=depth):
                probe = native_core.probe_main_search_contract(
                    board,
                    case.player,
                    candidates,
                    depth=depth,
                    node_limit=None,
                    branch_candidate_limit=8,
                    preselection_factor=3,
                    candidate_radius=2,
                    recent_move_count=4,
                    threat_extension_depth=2,
                    use_pvs=True,
                    use_transposition_table=True,
                )
                self.assertIsNotNone(probe)
                assert probe is not None
                selected = next(
                    coordinate
                    for coordinate, move in zip(case.candidates, candidates)
                    if move == probe.best_move
                )
                self.assertEqual(snapshot, (
                    selected,
                    probe.score,
                    tuple(score for _move, score in probe.root_scores),
                    probe.nodes,
                    probe.tt_entries,
                ))

    def test_full_window_flag_supports_future_review_callers(self) -> None:
        case = native_search_baseline.load_case()
        board = native_search_baseline.build_board(case)
        candidates = tuple(
            parse_move(coordinate, board.size)
            for coordinate in case.candidates
        )
        for use_tt in (False, True):
            with self.subTest(use_transposition_table=use_tt):
                ai = _NativeKeySearchAI(
                    case.player,
                    max_depth=3,
                    time_limit_seconds=None,
                    threat_extension_depth=2,
                    branch_candidate_limit=8,
                )
                ai.config = replace(
                    ai.config,
                    use_pvs=False,
                    use_transposition_table=use_tt,
                )
                ai._begin_move_search()
                expected = ai._search_root(
                    native_search_baseline.build_board(case),
                    case.player,
                    3,
                    list(candidates),
                    alpha=-INFINITY,
                    beta=INFINITY,
                )
                probe = native_core.probe_main_search_contract(
                    board,
                    case.player,
                    candidates,
                    depth=3,
                    node_limit=None,
                    branch_candidate_limit=8,
                    preselection_factor=3,
                    candidate_radius=2,
                    recent_move_count=4,
                    threat_extension_depth=2,
                    use_pvs=False,
                    use_transposition_table=use_tt,
                )
                self.assertIsNotNone(probe)
                assert probe is not None
                self.assertEqual(expected.move, probe.best_move)
                self.assertEqual(expected.score, probe.score)
                self.assertEqual(
                    expected.principal_variation,
                    probe.principal_variation,
                )
                self.assertEqual(ai._counters.nodes, probe.nodes)
                expected_scores = dict(expected.ranked_moves)
                self.assertEqual(
                    tuple(expected_scores[move] for move in candidates),
                    tuple(score for _move, score in probe.root_scores),
                )
                self.assertEqual(
                    len(ai._transposition_table),
                    probe.tt_entries,
                )
                self.assertEqual(
                    _native_tt_digest(ai._transposition_table, board.size),
                    probe.tt_digest,
                )

    def test_node_limit_returns_clean_cutoff_without_mutating_board(self) -> None:
        case = native_search_baseline.load_case(MOVE_21_FIXTURE)
        board = native_search_baseline.build_board(case)
        before = native_search_baseline.board_state(board)
        candidates = tuple(
            parse_move(coordinate, board.size)
            for coordinate in case.candidates
        )
        probe = native_core.probe_main_search_contract(
            board,
            case.player,
            candidates,
            depth=8,
            node_limit=10,
            branch_candidate_limit=8,
            preselection_factor=3,
            candidate_radius=2,
            recent_move_count=4,
            threat_extension_depth=2,
            use_pvs=True,
            use_transposition_table=True,
        )
        self.assertIsNotNone(probe)
        assert probe is not None
        self.assertEqual(STATUS_CUTOFF, probe.status)
        self.assertEqual(10, probe.nodes)
        self.assertEqual(0, probe.completed_depth)
        self.assertIsNone(probe.best_move)
        self.assertEqual(before, native_search_baseline.board_state(board))

    def test_deterministic_random_positions_match_python_oracle(self) -> None:
        generator = random.Random(1_700)
        for case_index in range(12):
            size = 9 if case_index < 6 else 11
            board = Board(size)
            coordinates = [
                (row, column)
                for row in range(size)
                for column in range(size)
            ]
            generator.shuffle(coordinates)
            for ply, move in enumerate(
                coordinates[: 8 + case_index % 5]
            ):
                board.place(
                    *move,
                    BLACK if ply % 2 == 0 else WHITE,
                )
            player = (
                BLACK
                if len(board.move_history) % 2 == 0
                else WHITE
            )
            legal = board.get_legal_moves()
            generator.shuffle(legal)
            candidates = tuple(legal[:3])
            depth = 1 + case_index % 2
            use_pvs = case_index % 3 != 0
            use_tt = case_index % 4 != 0
            before = native_search_baseline.board_state(board)

            ai = _NativeKeySearchAI(
                player,
                max_depth=depth,
                time_limit_seconds=None,
                branch_candidate_limit=8,
                threat_extension_depth=2,
            )
            ai.config = replace(
                ai.config,
                use_pvs=use_pvs,
                use_transposition_table=use_tt,
            )
            ai._begin_move_search()
            expected = ai._search_root(
                board,
                player,
                depth,
                list(candidates),
                alpha=-INFINITY,
                beta=INFINITY,
            )
            probe = native_core.probe_main_search_contract(
                board,
                player,
                candidates,
                depth=depth,
                node_limit=None,
                branch_candidate_limit=8,
                preselection_factor=3,
                candidate_radius=2,
                recent_move_count=4,
                threat_extension_depth=2,
                use_pvs=use_pvs,
                use_transposition_table=use_tt,
            )
            self.assertIsNotNone(probe)
            assert probe is not None
            expected_scores = dict(expected.ranked_moves)
            with self.subTest(
                case=case_index,
                depth=depth,
                use_pvs=use_pvs,
                use_tt=use_tt,
            ):
                self.assertEqual(STATUS_FOUND, probe.status)
                self.assertEqual(expected.move, probe.best_move)
                self.assertEqual(expected.score, probe.score)
                self.assertEqual(
                    expected.principal_variation,
                    probe.principal_variation,
                )
                self.assertEqual(ai._counters.nodes, probe.nodes)
                self.assertEqual(
                    tuple(expected_scores[move] for move in candidates),
                    tuple(score for _move, score in probe.root_scores),
                )
                self.assertEqual(
                    len(ai._transposition_table),
                    probe.tt_entries,
                )
                self.assertEqual(
                    _native_tt_digest(ai._transposition_table, size),
                    probe.tt_digest,
                )
                self.assertEqual(
                    before,
                    native_search_baseline.board_state(board),
                )


if __name__ == "__main__":
    unittest.main()
