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
            self.assertEqual("0.16.11", payload["engine_version"])
            self.assertEqual("H8", payload["moves"][0]["coordinate"])
            self.assertEqual("测试结束", payload["result"])


if __name__ == "__main__":
    unittest.main()
