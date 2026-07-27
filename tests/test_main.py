import unittest
from unittest.mock import patch

from engine.board import BLACK, WHITE
from main import choose_human_player, create_computer, create_recorder


class TestSideSelection(unittest.TestCase):
    @patch("builtins.input", return_value="B")
    def test_b_selects_black(self, _mock_input) -> None:
        self.assertEqual(BLACK, choose_human_player())

    @patch("builtins.input", return_value="")
    def test_empty_input_selects_white(self, _mock_input) -> None:
        self.assertEqual(WHITE, choose_human_player())

    @patch("builtins.input", side_effect=["invalid", "W"])
    @patch("builtins.print")
    def test_invalid_input_reprompts(
        self,
        _mock_print,
        _mock_input,
    ) -> None:
        self.assertEqual(WHITE, choose_human_player())

    def test_computer_uses_selected_side(self) -> None:
        black_ai = create_computer(BLACK)
        white_ai = create_computer(WHITE)

        self.assertEqual(BLACK, black_ai.player)
        self.assertEqual(WHITE, white_ai.player)

    def test_recorder_names_follow_selected_side(self) -> None:
        human_black = create_recorder(BLACK)
        human_white = create_recorder(WHITE)

        self.assertEqual("Human", human_black.black_name)
        self.assertEqual("SearchAI", human_black.white_name)
        self.assertEqual("SearchAI", human_white.black_name)
        self.assertEqual("Human", human_white.white_name)


if __name__ == "__main__":
    unittest.main()
