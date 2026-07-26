import unittest

from engine.board import BLACK, WHITE
from engine.game import (
    format_move,
    other_player,
    parse_move,
    player_name,
)

class TestMoveParser(unittest.TestCase):
    def test_center_position(self) -> None:
        """H8 应转换成内部坐标 (7, 7)。"""
        self.assertEqual(
            (7, 7),
            parse_move("H8"),
        )

    def test_first_position(self) -> None:
        """A1 应转换成内部坐标 (0, 0)。"""
        self.assertEqual(
            (0, 0),
            parse_move("A1"),
        )

    def test_last_position(self) -> None:
        """O15 应转换成内部坐标 (14, 14)。"""
        self.assertEqual(
            (14, 14),
            parse_move("O15"),
        )

    def test_lowercase_and_spaces(self) -> None:
        """小写字母和空格也应允许。"""
        self.assertEqual(
            (7, 7),
            parse_move(" h 8 "),
        )

    def test_invalid_column(self) -> None:
        """超出棋盘的列应被拒绝。"""
        with self.assertRaises(ValueError):
            parse_move("P8")

    def test_invalid_row(self) -> None:
        """超出棋盘的行应被拒绝。"""
        with self.assertRaises(ValueError):
            parse_move("H16")

    def test_missing_row(self) -> None:
        """只有列字母、没有行号时应被拒绝。"""
        with self.assertRaises(ValueError):
            parse_move("H")

    def test_format_center_position(self) -> None:
        """内部坐标 (7, 7) 应显示为 H8。"""
        self.assertEqual(
            "H8",
            format_move(7, 7),
        )

    def test_parse_and_format_are_reversible(self) -> None:
        """坐标解析后再格式化，应恢复原坐标。"""
        row, column = parse_move("O15")

        self.assertEqual(
            "O15",
            format_move(row, column),
        )


class TestPlayers(unittest.TestCase):
    def test_black_switches_to_white(self) -> None:
        self.assertEqual(
            WHITE,
            other_player(BLACK),
        )

    def test_white_switches_to_black(self) -> None:
        self.assertEqual(
            BLACK,
            other_player(WHITE),
        )

    def test_player_names(self) -> None:
        self.assertEqual("黑棋 X", player_name(BLACK))
        self.assertEqual("白棋 O", player_name(WHITE))


if __name__ == "__main__":
    unittest.main()