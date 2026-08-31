import json
import tempfile
import unittest
from pathlib import Path

from engine.board import BLACK, WHITE, Board
from engine.records import GameRecorder


class TestGameRecorder(unittest.TestCase):
    def setUp(self) -> None:
        self.board = Board()
        self.recorder = GameRecorder(
            mode="TEST",
            black_name="Black",
            white_name="White",
        )

    def _record(self, player: int, row: int, column: int) -> None:
        self.board.place(row, column, player)
        self.recorder.record_move(
            player=player,
            row=row,
            column=column,
            actor="Test",
            think_seconds=0.1,
            evaluation_before=0,
            evaluation_after=10,
        )

    def test_score_sheet_pairs_black_and_white_moves(self) -> None:
        self._record(BLACK, 7, 7)
        self._record(WHITE, 6, 7)
        self._record(BLACK, 7, 6)

        sheet = self.recorder.render_score_sheet(full=True)

        self.assertIn("1.  H8", sheet)
        self.assertIn("H7", sheet)
        self.assertIn("2.  G8", sheet)

    def test_undo_removes_moves_and_records_event(self) -> None:
        self._record(BLACK, 7, 7)
        self._record(WHITE, 6, 7)

        removed = self.recorder.undo_last_moves(2)

        self.assertEqual(2, len(removed))
        self.assertEqual([], self.recorder.moves)
        self.assertEqual("undo", self.recorder.events[-1].event_type)

    def test_save_writes_txt_and_json(self) -> None:
        self._record(BLACK, 7, 7)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.recorder.record_dir = Path(temp_dir)
            paths = self.recorder.save(
                board=self.board,
                result="测试结束",
                duration_seconds=1.25,
                prefix="test-game",
            )

            self.assertTrue(paths.txt.exists())
            self.assertTrue(paths.json.exists())

            payload = json.loads(paths.json.read_text(encoding="utf-8"))
            self.assertEqual("0.17.5", payload["engine_version"])
            self.assertEqual("H8", payload["moves"][0]["coordinate"])
            self.assertEqual("测试结束", payload["result"])

    def test_score_sheet_renders_parity_shadow_events(self) -> None:
        self.board.place(7, 7, BLACK)
        self.recorder.record_move(
            player=BLACK,
            row=7,
            column=7,
            actor="Test",
            think_seconds=0.1,
            evaluation_before=0,
            evaluation_after=10,
            analysis={
                "root_review_parity_shadow_events": [
                    {
                        "confirmed_coordinate": "H8",
                        "basis": "equal_window",
                        "completed_depth": 6,
                        "parity_state": "parity_disagreement",
                        "parity_leader_coordinate": None,
                        "would_veto": True,
                        "evidence": [
                            {"depth": 3, "coordinate": "H8"},
                            {"depth": 4, "coordinate": "I8"},
                            {"depth": 5, "coordinate": "H8"},
                            {"depth": 6, "coordinate": "H8"},
                        ],
                    }
                ]
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            self.recorder.record_dir = Path(temp_dir)
            paths = self.recorder.save(
                board=self.board,
                result="测试结束",
                duration_seconds=0.1,
                prefix="parity-shadow",
            )
            sheet = paths.txt.read_text(encoding="utf-8")

            self.assertIn(
                "root_review_parity_shadow=events:1 would_veto:1",
                sheet,
            )
            self.assertIn("state:parity_disagreement", sheet)
            self.assertIn(
                "d3:H8 -> d4:I8 -> d5:H8 -> d6:H8",
                sheet,
            )


if __name__ == "__main__":
    unittest.main()
