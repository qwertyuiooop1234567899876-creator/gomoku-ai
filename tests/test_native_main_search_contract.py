from __future__ import annotations

import unittest

from engine.game import parse_move
from engine.native_core import (
    MAIN_SEARCH_FLAG_PVS,
    MAIN_SEARCH_FLAG_TT,
    STATUS_MAIN_SEARCH_UNSUPPORTED,
    native_core,
)
from tools import native_search_baseline


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
        if probe is not None:
            self.assertEqual(STATUS_MAIN_SEARCH_UNSUPPORTED, probe.status)
            self.assertEqual(expected_digest, probe.input_digest)
        else:
            self.assertFalse(native_core.status()["main_search_available"])
            self.assertNotEqual(0, expected_digest)
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


if __name__ == "__main__":
    unittest.main()
