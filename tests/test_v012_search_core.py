import random
import unittest

from engine import evaluator
from engine.board import BLACK, DIRECTIONS, WHITE, Board
from engine.evaluator import (
    ThreatProfile,
    analyze_move_threats,
    evaluate_move,
    evaluate_player,
    is_winning_move,
    other_side,
)

Move = tuple[int, int]
Direction = tuple[int, int]


def board_state(board: Board) -> tuple[object, ...]:
    return (
        tuple(tuple(row) for row in board.grid),
        tuple(board.move_history),
        board.zobrist_hash,
        board.empty_count,
    )


def direction_positions(
    board: Board,
    anchor: Move,
    direction: Direction,
) -> list[Move]:
    row, column = anchor
    row_step, column_step = direction
    return [
        candidate
        for offset in range(-4, 5)
        if offset != 0
        if board.is_inside(
            *(candidate := (
                row + offset * row_step,
                column + offset * column_step,
            ))
        )
    ]


def winning_segment(
    board: Board,
    move: Move,
    player: int,
    direction: Direction,
) -> set[Move]:
    row, column = move
    row_step, column_step = direction
    segment = {move}
    for sign in (-1, 1):
        current_row = row + sign * row_step
        current_column = column + sign * column_step
        while (
            board.is_inside(current_row, current_column)
            and board.grid[current_row][current_column] == player
        ):
            segment.add((current_row, current_column))
            current_row += sign * row_step
            current_column += sign * column_step
    return segment


def winning_moves_in_direction(
    board: Board,
    anchor: Move,
    player: int,
    direction: Direction,
) -> set[Move]:
    wins: set[Move] = set()
    for candidate in direction_positions(board, anchor, direction):
        if not board.is_empty(*candidate):
            continue
        board.place(*candidate, player)
        try:
            segment = winning_segment(
                board,
                candidate,
                player,
                direction,
            )
            if len(segment) >= 5 and anchor in segment:
                wins.add(candidate)
        finally:
            board.undo()
    return wins


def creates_open_three(
    board: Board,
    anchor: Move,
    player: int,
    direction: Direction,
) -> bool:
    for extension in direction_positions(board, anchor, direction):
        if not board.is_empty(*extension):
            continue
        board.place(*extension, player)
        try:
            if len(
                winning_moves_in_direction(
                    board,
                    anchor,
                    player,
                    direction,
                )
            ) >= 2:
                return True
        finally:
            board.undo()
    return False


def reference_threat_profile(
    board: Board,
    move: Move,
    player: int,
) -> ThreatProfile:
    board.place(*move, player)
    try:
        all_wins: set[Move] = set()
        open_fours = 0
        fours = 0
        open_threes = 0
        for direction in DIRECTIONS:
            wins = winning_moves_in_direction(
                board,
                move,
                player,
                direction,
            )
            if wins:
                fours += 1
                all_wins.update(wins)
                if len(wins) >= 2:
                    open_fours += 1
            elif creates_open_three(
                board,
                move,
                player,
                direction,
            ):
                open_threes += 1
        return ThreatProfile(
            immediate_win=board.check_win(*move),
            open_four_directions=open_fours,
            four_directions=fours,
            open_three_directions=open_threes,
            winning_moves=tuple(sorted(all_wins)),
        )
    finally:
        board.undo()


def reference_evaluate_move(
    board: Board,
    move: Move,
    player: int,
) -> int:
    opponent = other_side(player)
    own_before = evaluate_player(board, player)
    opponent_before = evaluate_player(board, opponent)
    own_profile = reference_threat_profile(board, move, player)
    opponent_profile = reference_threat_profile(board, move, opponent)

    board.place(*move, player)
    try:
        own_after = evaluate_player(board, player)
    finally:
        board.undo()
    board.place(*move, opponent)
    try:
        opponent_after = evaluate_player(board, opponent)
    finally:
        board.undo()

    return int(
        max(0, own_after - own_before)
        + evaluator._profile_bonus(own_profile)
        + 1.15
        * (
            max(0, opponent_after - opponent_before)
            + evaluator._profile_bonus(opponent_profile)
        )
        + evaluator._center_bonus(board, *move)
    )


def seeded_boards() -> list[Board]:
    rng = random.Random(0x120)
    boards: list[Board] = []
    pool = [
        (row, column)
        for row in range(2, 13)
        for column in range(2, 13)
    ]
    for stone_count in (8, 11, 14, 17):
        board = Board()
        for index, move in enumerate(rng.sample(pool, stone_count)):
            board.place(
                *move,
                BLACK if index % 2 == 0 else WHITE,
            )
        boards.append(board)
    return boards


class TestV012EquivalentTacticalKernels(unittest.TestCase):
    def test_local_threat_profiles_match_simulation_reference(self) -> None:
        rng = random.Random(0xA11CE)
        for board in seeded_boards():
            before = board_state(board)
            candidates = rng.sample(board.get_legal_moves(), 8)
            for move in candidates:
                for player in (BLACK, WHITE):
                    self.assertEqual(
                        reference_threat_profile(board, move, player),
                        analyze_move_threats(board, *move, player),
                        (move, player),
                    )
            self.assertEqual(before, board_state(board))

    def test_line_delta_move_scores_match_full_board_reference(self) -> None:
        rng = random.Random(0xE7A1)
        for board in seeded_boards():
            before = board_state(board)
            candidates = rng.sample(board.get_legal_moves(), 4)
            for move in candidates:
                for player in (BLACK, WHITE):
                    self.assertEqual(
                        reference_evaluate_move(board, move, player),
                        evaluate_move(board, *move, player),
                        (move, player),
                    )
            self.assertEqual(before, board_state(board))

    def test_non_mutating_win_check_matches_place_and_check(self) -> None:
        for board in seeded_boards():
            before = board_state(board)
            for move in board.get_legal_moves():
                for player in (BLACK, WHITE):
                    board.place(*move, player)
                    try:
                        reference = board.check_win(*move)
                    finally:
                        board.undo()
                    self.assertEqual(
                        reference,
                        is_winning_move(board, *move, player),
                        (move, player),
                    )
            self.assertEqual(before, board_state(board))


if __name__ == "__main__":
    unittest.main()
