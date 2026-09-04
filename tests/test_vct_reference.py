from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
import unittest

from engine.board import BLACK, WHITE, Board
from engine.evaluator import other_side
from engine.game import format_move, parse_move
from engine.proof_search import ProofBudget, ProofSearch, ProofState, ProofTable
from engine.threats import ThreatAnalyzer
from tools.vct_reference import (
    DIAGNOSTIC_SCHEMA_VERSION,
    VCTReferenceCase,
    _AuditedProofSearch,
    _AuditedThreatAnalyzer,
    analyze_query_hotspots,
    build_board,
    diagnostic_payload,
    load_case,
    run_reentry_comparison,
    run_reference,
)


POSITIONS = Path(__file__).resolve().parent / "positions"
FIXTURES = (
    POSITIONS / "v0175_reverse_move10_vct.json",
    POSITIONS / "v0175_selfplay_move24_vct.json",
    POSITIONS / "v0175_yixin_move21_vct.json",
)


class TestVCTReference(unittest.TestCase):
    def test_v0175_fixtures_preserve_ordered_history(self) -> None:
        expected = (
            (WHITE, ("G10", "K6")),
            (WHITE, ("J10", "J6")),
            (BLACK, ("I8", "H11")),
        )
        for path, (player, candidates) in zip(FIXTURES, expected):
            with self.subTest(path=path.name):
                case = load_case(path)
                board = build_board(case)
                self.assertEqual(player, case.player)
                self.assertEqual(candidates, case.candidates)
                self.assertEqual(len(case.history), len(board.move_history))
                self.assertEqual(case.expected_hash, board.zobrist_hash)

    def test_zero_node_budget_remains_unknown_for_every_fixture(self) -> None:
        for path in FIXTURES:
            with self.subTest(path=path.name):
                case = load_case(path)
                run = run_reference(
                    case,
                    seconds_per_candidate=0.1,
                    max_nodes=0,
                )
                self.assertTrue(run.candidates)
                for candidate in run.candidates:
                    self.assertEqual(
                        ProofState.UNKNOWN.value,
                        candidate.attacker_state,
                    )
                    self.assertFalse(candidate.completed)
                    self.assertEqual("node_limit", candidate.cutoff_reason)
                    self.assertEqual(0, candidate.transposition_hits)
                    self.assertEqual(0, candidate.proof_table_stats.queries)
                    self.assertEqual(0, candidate.distinct_query_keys)
                    self.assertEqual(0, candidate.repeated_queries)
                    self.assertEqual(
                        0,
                        candidate.query_hotspots.repeated_key_count,
                    )
                    self.assertEqual(
                        0,
                        candidate.query_hotspots.max_query_frequency,
                    )

    def test_reference_exposes_proof_table_diagnostics(self) -> None:
        case = load_case(FIXTURES[0])
        result = run_reference(
            case,
            coordinates=("G10",),
            seconds_per_candidate=1.0,
            max_nodes=100,
        ).candidates[0]

        stats = result.proof_table_stats
        self.assertGreater(stats.queries, 0)
        self.assertEqual(result.transposition_hits, stats.hits)
        self.assertGreater(stats.stores + stats.skipped_stores, 0)
        self.assertLessEqual(result.distinct_query_keys, stats.queries)
        self.assertEqual(
            stats.queries,
            result.distinct_query_keys + result.repeated_queries,
        )
        self.assertLessEqual(
            result.query_hotspots.repeated_key_count,
            result.distinct_query_keys,
        )
        serialized = asdict(result)
        self.assertEqual(stats.queries, serialized["proof_table_stats"]["queries"])

    def test_reference_exposes_phase1f_work_diagnostics(self) -> None:
        case = load_case(FIXTURES[0])
        result = run_reference(
            case,
            coordinates=("G10",),
            seconds_per_candidate=1.0,
            max_nodes=100,
        ).candidates[0]

        threat_stats = result.threat_analyzer_stats
        threat_audit = result.threat_audit
        proof_audit = result.proof_search_audit
        self.assertEqual(
            result.threat_exact_descriptions,
            threat_stats.exact_descriptions,
        )
        self.assertGreater(threat_audit.defense_set_generations, 0)
        self.assertLessEqual(
            threat_audit.complete_defense_sets,
            threat_audit.defense_set_generations,
        )
        self.assertLessEqual(
            proof_audit.examined_defenses,
            proof_audit.available_defenses,
        )
        replay_failures = sum(
            item.count for item in proof_audit.replay_failure_reasons
        )
        self.assertEqual(
            proof_audit.replay_attempts,
            proof_audit.replay_successes + replay_failures,
        )
        self.assertGreaterEqual(
            proof_audit.unchecked_defenses_on_budget_exhaustion,
            0,
        )

    def test_phase1f_audit_preserves_reference_search_result(self) -> None:
        case = load_case(FIXTURES[0])
        board = build_board(case)
        before = (
            tuple(tuple(row) for row in board.grid),
            tuple(board.move_history),
            board.zobrist_hash,
            board.empty_count,
        )
        plain = ProofSearch(
            budget=ProofBudget.from_now(
                1.0,
                max_nodes=100,
                max_attacker_moves=6,
                max_quiet_frontiers=16,
                max_quiet_attacker_moves=2,
                vcf_max_attacker_moves=6,
                use_vcf_oracle=True,
                clock=time.perf_counter,
            ),
            analyzer=ThreatAnalyzer(
                candidate_limit=24,
                frontier_scan_limit=48,
            ),
            table=ProofTable(),
            clock=time.perf_counter,
        ).search_after_move(
            board,
            move=parse_move("G10", board.size),
            mover=case.player,
            attacker=other_side(case.player),
            side_to_move=other_side(case.player),
        )
        audited = run_reference(
            case,
            coordinates=("G10",),
            seconds_per_candidate=1.0,
            max_nodes=100,
        ).candidates[0]

        self.assertEqual(plain.state.value, audited.attacker_state)
        self.assertEqual(plain.completed, audited.completed)
        self.assertEqual(plain.cutoff_reason, audited.cutoff_reason)
        self.assertEqual(plain.nodes, audited.nodes)
        self.assertEqual(
            tuple(format_move(*move) for move in plain.principal_variation),
            audited.principal_variation,
        )
        self.assertEqual(
            before,
            (
                tuple(tuple(row) for row in board.grid),
                tuple(board.move_history),
                board.zobrist_hash,
                board.empty_count,
            ),
        )

    def test_diagnostic_json_has_an_independent_schema(self) -> None:
        case = load_case(FIXTURES[0])
        run = run_reference(
            case,
            coordinates=("G10",),
            seconds_per_candidate=0.1,
            max_nodes=0,
        )

        payload = diagnostic_payload(run)

        self.assertEqual(
            DIAGNOSTIC_SCHEMA_VERSION,
            payload["diagnostic_schema_version"],
        )
        self.assertEqual("reference", payload["run_type"])
        self.assertIn("proof_search_audit", payload["run"]["candidates"][0])
        self.assertEqual(0, payload["run"]["config"]["nodes_per_pass"])

    def test_diagnostic_json_records_reference_and_reentry_parameters(
        self,
    ) -> None:
        case = load_case(FIXTURES[0])
        reference = run_reference(
            case,
            coordinates=("G10",),
            seconds_per_candidate=0.25,
            max_nodes=0,
            max_attacker_moves=3,
            max_quiet_frontiers=7,
            max_quiet_attacker_moves=1,
            vcf_max_attacker_moves=4,
            candidate_limit=9,
            frontier_scan_limit=None,
        )
        comparison = run_reentry_comparison(
            case,
            coordinate="G10",
            total_nodes=40,
            warm_passes=2,
            seconds_per_warm_pass=0.5,
            max_attacker_moves=4,
            max_quiet_frontiers=8,
            max_quiet_attacker_moves=0,
            vcf_max_attacker_moves=5,
            candidate_limit=10,
            frontier_scan_limit=12,
        )

        reference_config = diagnostic_payload(reference)["run"]["config"]
        comparison_config = diagnostic_payload(comparison)["run"]["config"]
        self.assertEqual(0.25, reference_config["seconds_per_pass"])
        self.assertEqual(3, reference_config["max_attacker_moves"])
        self.assertEqual(9, reference_config["candidate_limit"])
        self.assertIsNone(reference_config["frontier_scan_limit"])
        self.assertEqual(2, comparison_config["passes"])
        self.assertEqual(20, comparison_config["nodes_per_pass"])
        self.assertEqual(4, comparison_config["max_attacker_moves"])
        self.assertEqual(10, comparison_config["candidate_limit"])

    def test_deadline_audit_matches_plain_proof_search(self) -> None:
        class StepClock:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self) -> float:
                self.calls += 1
                return 0.0 if self.calls <= 3 else 2.0

        def run(search_type, analyzer):  # type: ignore[no-untyped-def]
            board = Board(size=9)
            for column in (2, 3, 4):
                board.place(4, column, BLACK)
            before = (
                tuple(tuple(row) for row in board.grid),
                tuple(board.move_history),
                board.zobrist_hash,
                board.empty_count,
            )
            result = search_type(
                budget=ProofBudget(
                    max_nodes=100,
                    max_attacker_moves=2,
                    deadline=1.0,
                ),
                analyzer=analyzer,
                table=ProofTable(),
                clock=StepClock(),
            ).search(
                board,
                attacker=BLACK,
                side_to_move=BLACK,
            )
            after = (
                tuple(tuple(row) for row in board.grid),
                tuple(board.move_history),
                board.zobrist_hash,
                board.empty_count,
            )
            self.assertEqual(before, after)
            return result

        plain = run(ProofSearch, ThreatAnalyzer())
        audited = run(_AuditedProofSearch, _AuditedThreatAnalyzer())

        self.assertIs(ProofState.UNKNOWN, plain.state)
        self.assertEqual("deadline", plain.cutoff_reason)
        self.assertEqual(plain.state, audited.state)
        self.assertEqual(plain.completed, audited.completed)
        self.assertEqual(plain.cutoff_reason, audited.cutoff_reason)
        self.assertEqual(plain.nodes, audited.nodes)
        self.assertEqual(
            plain.principal_variation,
            audited.principal_variation,
        )

    def test_hotspot_summary_measures_repeated_query_concentration(self) -> None:
        analysis = analyze_query_hotspots(
            {
                "hot": 11,
                "warm": 6,
                "cool": 3,
                "unique": 1,
            }
        )

        self.assertEqual(3, analysis.repeated_key_count)
        self.assertEqual(11, analysis.max_query_frequency)
        top_one, top_ten, top_hundred = analysis.buckets
        self.assertEqual(10, top_one.repeated_queries)
        self.assertAlmostEqual(10 / 17, top_one.repeated_query_share)
        self.assertEqual(17, top_ten.repeated_queries)
        self.assertEqual(1.0, top_ten.repeated_query_share)
        self.assertEqual(17, top_hundred.repeated_queries)
        self.assertEqual(1.0, top_hundred.repeated_query_share)

    def test_reentry_comparison_tracks_equal_node_overlap(self) -> None:
        case = load_case(FIXTURES[0])
        comparison = run_reentry_comparison(
            case,
            coordinate="G10",
            total_nodes=40,
            warm_passes=2,
            seconds_per_warm_pass=1.0,
        )

        self.assertEqual(40, comparison.cold_result.nodes)
        self.assertEqual(20, comparison.nodes_per_warm_pass)
        self.assertEqual(2, len(comparison.warm_passes))
        self.assertEqual(
            40,
            sum(item.result.nodes for item in comparison.warm_passes),
        )
        first, second = comparison.warm_passes
        self.assertEqual(0, first.previous_overlap_keys)
        self.assertGreater(second.previous_overlap_keys, 0)
        self.assertGreater(second.cumulative_overlap_keys, 0)
        self.assertEqual(
            second.result.transposition_hits,
            second.proof_table_delta.hits,
        )
        self.assertEqual(
            second.proof_table_delta,
            second.result.proof_table_stats,
        )
        self.assertGreater(
            second.result.cumulative_proof_table_stats.queries,
            second.result.proof_table_stats.queries,
        )
        payload = asdict(comparison)
        self.assertEqual(
            second.previous_overlap_keys,
            payload["warm_passes"][1]["previous_overlap_keys"],
        )

    def test_open_four_witness_is_a_completed_attacker_win(self) -> None:
        history = ("H8", "A15", "I8", "B15", "J8", "C15", "K8")
        board = Board(15)
        for index, coordinate in enumerate(history):
            board.place(
                *parse_move(coordinate, board.size),
                BLACK if index % 2 == 0 else WHITE,
            )
        case = VCTReferenceCase(
            name="open_four_reference",
            board_size=15,
            player=WHITE,
            history=history,
            candidates=("A1",),
            expected_hash=board.zobrist_hash,
        )

        run = run_reference(
            case,
            seconds_per_candidate=1.0,
            max_nodes=1_000,
            max_quiet_frontiers=0,
            max_quiet_attacker_moves=0,
        )

        result = run.candidates[0]
        self.assertEqual(ProofState.PROVEN_WIN.value, result.attacker_state)
        self.assertTrue(result.completed)
        self.assertIsNone(result.cutoff_reason)
        self.assertIn(result.best_coordinate, {"G8", "L8"})

    def test_candidate_filter_cannot_invent_a_fixture_move(self) -> None:
        case = load_case(FIXTURES[0])
        with self.assertRaisesRegex(ValueError, "必须来自夹具"):
            run_reference(
                case,
                coordinates=("A1",),
                seconds_per_candidate=0.1,
                max_nodes=0,
            )


if __name__ == "__main__":
    unittest.main()
