from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from engine.board import Board

Move = tuple[int, int]
ABI_VERSION = 1
STATUS_NOT_FOUND = 0
STATUS_FOUND = 1
STATUS_CUTOFF = 2


@dataclass(frozen=True, slots=True)
class NativeThreatProfile:
    immediate_win: bool
    open_four_directions: int
    four_directions: int
    open_three_directions: int
    winning_moves: tuple[Move, ...]


@dataclass(frozen=True, slots=True)
class NativeVCFResult:
    status: int
    line: tuple[Move, ...]
    nodes: int

    @property
    def found(self) -> bool:
        return self.status == STATUS_FOUND

    @property
    def cutoff(self) -> bool:
        return self.status == STATUS_CUTOFF


class NativeCore:
    """Version-independent C ABI wrapper with a safe Python fallback path."""

    def __init__(self) -> None:
        self._library: ctypes.CDLL | None = None
        self._path: Path | None = None
        self._error: str | None = None
        self._load()

    @property
    def available(self) -> bool:
        return self._library is not None

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def error(self) -> str | None:
        return self._error

    def status(self) -> dict[str, object]:
        return {
            "available": self.available,
            "abi_version": ABI_VERSION if self.available else None,
            "path": None if self._path is None else str(self._path),
            "error": self._error,
        }

    def _load(self) -> None:
        if os.environ.get("GOMOKU_NATIVE_DISABLE", "").strip() in {
            "1", "true", "yes", "on",
        }:
            self._error = "已通过GOMOKU_NATIVE_DISABLE禁用"
            return
        suffix = (
            ".dll" if sys.platform == "win32"
            else ".dylib" if sys.platform == "darwin"
            else ".so"
        )
        path = Path(__file__).resolve().parent.parent / "native" / "bin" / f"gomoku_native{suffix}"
        if not path.is_file():
            self._error = f"原生库不存在：{path.name}"
            return
        try:
            library = ctypes.CDLL(str(path))
            library.gn_abi_version.argtypes = []
            library.gn_abi_version.restype = ctypes.c_int
            if library.gn_abi_version() != ABI_VERSION:
                self._error = "NativeCore ABI版本不匹配"
                return

            byte_pointer = ctypes.POINTER(ctypes.c_uint8)
            int_pointer = ctypes.POINTER(ctypes.c_int)
            library.gn_find_winning_moves.argtypes = [
                byte_pointer, ctypes.c_int, ctypes.c_int,
                int_pointer, ctypes.c_int, int_pointer, ctypes.c_int,
            ]
            library.gn_find_winning_moves.restype = ctypes.c_int
            library.gn_analyze_move.argtypes = [
                byte_pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, int_pointer, ctypes.c_int,
            ]
            library.gn_analyze_move.restype = ctypes.c_int
            library.gn_analyze_moves.argtypes = [
                byte_pointer, ctypes.c_int, ctypes.c_int,
                int_pointer, ctypes.c_int, int_pointer, ctypes.c_int,
            ]
            library.gn_analyze_moves.restype = ctypes.c_int
            library.gn_counter_support_mask.argtypes = [
                byte_pointer, ctypes.c_int, ctypes.c_int,
                int_pointer, ctypes.c_int, ctypes.c_int,
                byte_pointer,
            ]
            library.gn_counter_support_mask.restype = ctypes.c_int
            library.gn_find_vcf.argtypes = [
                byte_pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                int_pointer, ctypes.c_int, int_pointer, int_pointer,
            ]
            library.gn_find_vcf.restype = ctypes.c_int
        except (AttributeError, OSError) as exc:
            self._error = f"原生库加载失败：{exc}"
            return
        self._library = library
        self._path = path
        self._error = None

    @staticmethod
    def _grid(board: Board) -> ctypes.Array[ctypes.c_uint8]:
        # ``ctypes`` positional construction converts every cell through the
        # Python C-API.  Proof replay calls this bridge thousands of times, so
        # that conversion cost can dominate the native kernels on Windows.
        # Rows already contain byte-sized cell values; joining them and doing
        # one contiguous copy preserves the exact ABI layout with far less
        # interpreter overhead.
        flattened = b"".join(map(bytes, board.grid))
        return (ctypes.c_uint8 * len(flattened)).from_buffer_copy(flattened)

    def find_winning_moves(
        self,
        board: Board,
        player: int,
        candidates: Sequence[Move] | None = None,
    ) -> list[Move] | None:
        library = self._library
        if library is None:
            return None
        grid = self._grid(board)
        if candidates is None:
            candidate_array = None
            candidate_count = 0
            capacity = board.empty_count
        else:
            encoded = [row * board.size + column for row, column in candidates]
            candidate_array = (ctypes.c_int * len(encoded))(*encoded)
            candidate_count = len(encoded)
            capacity = len(encoded)
        output = (ctypes.c_int * max(1, capacity))()
        count = library.gn_find_winning_moves(
            grid,
            board.size,
            player,
            candidate_array,
            candidate_count,
            output,
            capacity,
        )
        if count < 0:
            raise RuntimeError("NativeCore一步胜点计算失败")
        return [
            (output[index] // board.size, output[index] % board.size)
            for index in range(count)
        ]

    def analyze_move(
        self,
        board: Board,
        row: int,
        column: int,
        player: int,
    ) -> NativeThreatProfile | None:
        library = self._library
        if library is None:
            return None
        grid = self._grid(board)
        output = (ctypes.c_int * (5 + board.size * board.size))()
        count = library.gn_analyze_move(
            grid,
            board.size,
            row,
            column,
            player,
            output,
            len(output),
        )
        if count < 5:
            raise RuntimeError("NativeCore威胁画像计算失败")
        winning_count = output[4]
        winning_moves = tuple(
            (
                output[5 + index] // board.size,
                output[5 + index] % board.size,
            )
            for index in range(winning_count)
        )
        return NativeThreatProfile(
            immediate_win=bool(output[0]),
            open_four_directions=output[1],
            four_directions=output[2],
            open_three_directions=output[3],
            winning_moves=winning_moves,
        )

    def analyze_moves(
        self,
        board: Board,
        moves: Sequence[Move],
        player: int,
    ) -> list[NativeThreatProfile] | None:
        library = self._library
        if library is None:
            return None
        if not moves:
            return []
        grid = self._grid(board)
        encoded = [row * board.size + column for row, column in moves]
        candidates = (ctypes.c_int * len(encoded))(*encoded)
        stride = 13
        output = (ctypes.c_int * (stride * len(encoded)))()
        count = library.gn_analyze_moves(
            grid,
            board.size,
            player,
            candidates,
            len(encoded),
            output,
            stride,
        )
        if count != len(encoded):
            raise RuntimeError("NativeCore批量威胁画像计算失败")
        results: list[NativeThreatProfile] = []
        for offset in range(count):
            base = offset * stride
            winning_count = output[base + 4]
            results.append(
                NativeThreatProfile(
                    immediate_win=bool(output[base]),
                    open_four_directions=output[base + 1],
                    four_directions=output[base + 2],
                    open_three_directions=output[base + 3],
                    winning_moves=tuple(
                        (
                            output[base + 5 + index] // board.size,
                            output[base + 5 + index] % board.size,
                        )
                        for index in range(winning_count)
                    ),
                )
            )
        return results

    def counter_support_mask(
        self,
        board: Board,
        moves: Sequence[Move],
        player: int,
        *,
        minimum: int,
    ) -> tuple[bool, ...] | None:
        library = self._library
        if library is None:
            return None
        if not moves:
            return ()
        grid = self._grid(board)
        encoded = [row * board.size + column for row, column in moves]
        candidates = (ctypes.c_int * len(encoded))(*encoded)
        output = (ctypes.c_uint8 * len(encoded))()
        count = library.gn_counter_support_mask(
            grid,
            board.size,
            player,
            candidates,
            len(encoded),
            minimum,
            output,
        )
        if count != len(encoded):
            raise RuntimeError("NativeCore批量反击支撑计算失败")
        return tuple(bool(output[index]) for index in range(count))

    def find_vcf(
        self,
        board: Board,
        attacker: int,
        remaining_attacker_moves: int,
        *,
        max_nodes: int,
        timeout_seconds: float | None,
        candidate_limit: int,
    ) -> NativeVCFResult | None:
        library = self._library
        if library is None:
            return None
        grid = self._grid(board)
        output = (ctypes.c_int * (board.size * board.size))()
        nodes = ctypes.c_int(0)
        line_length = ctypes.c_int(0)
        timeout_ms = (
            0 if timeout_seconds is None
            else max(1, int(timeout_seconds * 1000))
        )
        status = library.gn_find_vcf(
            grid,
            board.size,
            attacker,
            remaining_attacker_moves,
            max_nodes,
            timeout_ms,
            candidate_limit,
            output,
            len(output),
            ctypes.byref(nodes),
            ctypes.byref(line_length),
        )
        if status < 0:
            raise RuntimeError("NativeCore VCF搜索失败")
        line = tuple(
            (output[index] // board.size, output[index] % board.size)
            for index in range(line_length.value)
        )
        return NativeVCFResult(status=status, line=line, nodes=nodes.value)


native_core = NativeCore()


def native_core_status() -> dict[str, object]:
    return native_core.status()
