"""Continuous-four (VCF) line search.

The algorithm is isolated from the main PVS coordinator.  All mutable search
state (deadline, node counter, candidate ordering, and position key) is
supplied explicitly through callbacks, so this module neither shares nor
silently borrows PVS/Proof transposition state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from engine.ai import Move
from engine.board import Board
from engine.evaluator import (
    _find_winning_moves_python,
    find_winning_moves,
    other_side,
)

PositionKey = Callable[[Board, int], int]
ForcingCandidates = Callable[[Board, int], list[Move]]
CheckTimeout = Callable[[], None]
CountNode = Callable[[], None]


def validate_vcf_certificate(
    board: Board,
    attacker: int,
    line: tuple[Move, ...],
) -> bool:
    """Strictly replay a VCF witness and always restore the board.

    Native code is only an accelerator.  Its result becomes proof evidence
    after this independent Python replay confirms every forced block and the
    terminal win/double-win condition.
    """
    if not line:
        return False
    defender = other_side(attacker)
    placed = 0
    index = 0
    try:
        while index < len(line):
            attack_move = line[index]
            if not board.is_empty(*attack_move):
                return False
            board.place(*attack_move, attacker)
            placed += 1
            if board.check_win(*attack_move):
                return index == len(line) - 1

            attack_wins = tuple(
                _find_winning_moves_python(board, attacker)
            )
            if len(attack_wins) >= 2:
                return index == len(line) - 1
            if len(attack_wins) != 1:
                return False
            if _find_winning_moves_python(board, defender):
                return False
            if index + 1 >= len(line):
                return False

            forced_block = line[index + 1]
            if forced_block != attack_wins[0] or not board.is_empty(*forced_block):
                return False
            board.place(*forced_block, defender)
            placed += 1
            if board.check_win(*forced_block):
                return False
            index += 2
        return False
    finally:
        for _ in range(placed):
            board.undo()


@dataclass(frozen=True, slots=True)
class VCFSearch:
    position_key: PositionKey
    forcing_candidates: ForcingCandidates
    check_timeout: CheckTimeout
    count_node: CountNode

    def find(
        self,
        board: Board,
        attacker: int,
        remaining_attacker_moves: int,
    ) -> tuple[Move, ...] | None:
        visited: set[tuple[int, int]] = set()
        return self._search(
            board,
            attacker,
            remaining_attacker_moves,
            visited,
        )

    def _search(
        self,
        board: Board,
        attacker: int,
        remaining_attacker_moves: int,
        visited: set[tuple[int, int]],
    ) -> tuple[Move, ...] | None:
        self.check_timeout()
        self.count_node()

        key = (
            self.position_key(board, attacker),
            remaining_attacker_moves,
        )
        if key in visited:
            return None
        visited.add(key)

        legal_moves = board.get_legal_moves()
        immediate = find_winning_moves(board, attacker, legal_moves)
        if immediate:
            return (immediate[0],)
        if remaining_attacker_moves <= 0:
            return None

        defender = other_side(attacker)
        for move in self.forcing_candidates(board, attacker):
            self.check_timeout()
            board.place(*move, attacker)
            try:
                if board.check_win(*move):
                    return (move,)

                attack_wins = find_winning_moves(board, attacker)
                if len(attack_wins) >= 2:
                    return (move,)
                if len(attack_wins) != 1:
                    continue

                # A direct counter-win means the defender need not obey the
                # attacker's continuous-four line.
                defender_wins = find_winning_moves(board, defender)
                if defender_wins:
                    continue

                forced_block = attack_wins[0]
                board.place(*forced_block, defender)
                try:
                    if board.check_win(*forced_block):
                        continue
                    child = self._search(
                        board,
                        attacker,
                        remaining_attacker_moves - 1,
                        visited,
                    )
                finally:
                    board.undo()

                if child:
                    return (move, forced_block, *child)
            finally:
                board.undo()

        return None
