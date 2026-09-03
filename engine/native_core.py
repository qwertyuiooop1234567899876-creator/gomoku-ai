from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from engine.board import Board
    from engine.threats import ThreatContinuation

Move = tuple[int, int]
ABI_VERSION = 1
STATUS_NOT_FOUND = 0
STATUS_FOUND = 1
STATUS_CUTOFF = 2
STATUS_MAIN_SEARCH_UNSUPPORTED = -2
MAIN_SEARCH_SCHEMA_VERSION = 1
MAIN_SEARCH_FLAG_PVS = 1 << 0
MAIN_SEARCH_FLAG_TT = 1 << 1
DEFENSE_CLASSIFICATION_SCHEMA_VERSION = 1
DEFENSE_CLASSIFICATION_COMPLETE = 1
DEFENSE_CLASSIFICATION_CUTOFF = 2
DEFENSE_CUTOFF_NONE = 0
DEFENSE_CUTOFF_TIMEOUT = 1
DEFENSE_CUTOFF_REPLY_LIMIT = 2
MIN_BOARD_SIZE = 5
MAX_BOARD_SIZE = 25


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


@dataclass(frozen=True, slots=True)
class NativeDefenseRefutation:
    defense_move: Move
    continuation_move: Move
    continuation_is_immediate: bool
    winning_points: tuple[Move, ...] = ()

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            self.defense_move,
            self.continuation_move,
            self.continuation_is_immediate,
            self.winning_points,
        )


@dataclass(frozen=True, slots=True)
class NativeDefenseClassification:
    required_defenses: tuple[Move, ...]
    counter_wins: tuple[Move, ...]
    refutations: tuple[NativeDefenseRefutation, ...]
    unclassified_replies: tuple[Move, ...]
    legal_reply_count: int
    refuted_reply_count: int
    coverage_complete: bool
    analysis_completed: bool
    status: int
    cutoff_reason: int
    processed_reply_count: int

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            self.required_defenses,
            self.counter_wins,
            tuple(item.signature for item in self.refutations),
            self.unclassified_replies,
            self.legal_reply_count,
            self.refuted_reply_count,
            self.coverage_complete,
            self.analysis_completed,
        )


@dataclass(frozen=True, slots=True)
class NativeMainSearchProbe:
    status: int
    completed_depth: int
    stop_reason: int
    best_move: Move | None
    score: int
    nodes: int
    tt_entries: int
    input_digest: int
    tt_digest: int
    root_scores: tuple[tuple[Move, int], ...]
    principal_variation: tuple[Move, ...]

    @property
    def completed(self) -> bool:
        return self.status == STATUS_FOUND

    @property
    def cutoff(self) -> bool:
        return self.status == STATUS_CUTOFF


class _MainSearchRequestV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("schema_version", ctypes.c_uint32),
        ("cells", ctypes.POINTER(ctypes.c_uint8)),
        ("board_size", ctypes.c_int32),
        ("history_indices", ctypes.POINTER(ctypes.c_int32)),
        ("history_players", ctypes.POINTER(ctypes.c_uint8)),
        ("history_count", ctypes.c_int32),
        ("player", ctypes.c_int32),
        ("root_candidates", ctypes.POINTER(ctypes.c_int32)),
        ("root_candidate_count", ctypes.c_int32),
        ("depth", ctypes.c_int32),
        ("node_limit", ctypes.c_int64),
        ("branch_candidate_limit", ctypes.c_int32),
        ("preselection_factor", ctypes.c_int32),
        ("candidate_radius", ctypes.c_int32),
        ("recent_move_count", ctypes.c_int32),
        ("threat_extension_depth", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
    ]


class _MainSearchResultV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("schema_version", ctypes.c_uint32),
        ("status", ctypes.c_int32),
        ("completed_depth", ctypes.c_int32),
        ("stop_reason", ctypes.c_int32),
        ("best_move", ctypes.c_int32),
        ("score", ctypes.c_int32),
        ("nodes", ctypes.c_int64),
        ("tt_entries", ctypes.c_int64),
        ("input_digest", ctypes.c_uint64),
        ("tt_digest", ctypes.c_uint64),
        ("root_scores", ctypes.POINTER(ctypes.c_int32)),
        ("root_score_capacity", ctypes.c_int32),
        ("root_score_count", ctypes.c_int32),
        ("principal_variation", ctypes.POINTER(ctypes.c_int32)),
        ("pv_capacity", ctypes.c_int32),
        ("pv_length", ctypes.c_int32),
    ]


class _DefenseContinuationV1(ctypes.Structure):
    _fields_ = [
        ("move", ctypes.c_int32),
        ("immediate_win", ctypes.c_int32),
        ("winning_points", ctypes.POINTER(ctypes.c_int32)),
        ("winning_point_count", ctypes.c_int32),
    ]


class _DefenseRefutationV1(ctypes.Structure):
    _fields_ = [
        ("defense_move", ctypes.c_int32),
        ("continuation_move", ctypes.c_int32),
        ("continuation_is_immediate", ctypes.c_int32),
        ("winning_point_offset", ctypes.c_int32),
        ("winning_point_count", ctypes.c_int32),
    ]


class _DefenseClassificationRequestV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("schema_version", ctypes.c_uint32),
        ("cells", ctypes.POINTER(ctypes.c_uint8)),
        ("board_size", ctypes.c_int32),
        ("attacker", ctypes.c_int32),
        ("continuations", ctypes.POINTER(_DefenseContinuationV1)),
        ("continuation_count", ctypes.c_int32),
        ("counter_wins", ctypes.POINTER(ctypes.c_int32)),
        ("counter_win_count", ctypes.c_int32),
        ("reply_limit", ctypes.c_int32),
        ("timeout_ms", ctypes.c_int32),
    ]


class _DefenseClassificationResultV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("schema_version", ctypes.c_uint32),
        ("status", ctypes.c_int32),
        ("cutoff_reason", ctypes.c_int32),
        ("legal_reply_count", ctypes.c_int32),
        ("processed_reply_count", ctypes.c_int32),
        ("coverage_complete", ctypes.c_int32),
        ("analysis_completed", ctypes.c_int32),
        ("required_defenses", ctypes.POINTER(ctypes.c_int32)),
        ("required_capacity", ctypes.c_int32),
        ("required_count", ctypes.c_int32),
        ("refutations", ctypes.POINTER(_DefenseRefutationV1)),
        ("refutation_capacity", ctypes.c_int32),
        ("refutation_count", ctypes.c_int32),
        ("unclassified_replies", ctypes.POINTER(ctypes.c_int32)),
        ("unclassified_capacity", ctypes.c_int32),
        ("unclassified_count", ctypes.c_int32),
        ("refutation_winning_points", ctypes.POINTER(ctypes.c_int32)),
        ("winning_point_capacity", ctypes.c_int32),
        ("winning_point_count", ctypes.c_int32),
    ]


class NativeCore:
    """Version-independent C ABI wrapper with a safe Python fallback path."""

    def __init__(self) -> None:
        self._library: ctypes.CDLL | None = None
        self._main_search_available = False
        self._defense_classification_available = False
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
            "main_search_available": self._main_search_available,
            "defense_classification_available": (
                self._defense_classification_available
            ),
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
            try:
                library.gn_main_search_v1.argtypes = [
                    ctypes.POINTER(_MainSearchRequestV1),
                    ctypes.POINTER(_MainSearchResultV1),
                ]
                library.gn_main_search_v1.restype = ctypes.c_int
                self._main_search_available = True
            except AttributeError:
                # ABI 1 remains compatible with pre-main-search runtimes. The
                # production kernels stay available while the new coarse
                # search capability is compiled and verified independently.
                self._main_search_available = False
            try:
                library.gn_classify_defenses_v1.argtypes = [
                    ctypes.POINTER(_DefenseClassificationRequestV1),
                    ctypes.POINTER(_DefenseClassificationResultV1),
                ]
                library.gn_classify_defenses_v1.restype = ctypes.c_int
                self._defense_classification_available = True
            except AttributeError:
                self._defense_classification_available = False
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

    @staticmethod
    def _supports_board(board: Board) -> bool:
        # The C ABI validates this same range.  Larger Python boards remain
        # supported by returning ``None`` and using the reference kernels.
        return MIN_BOARD_SIZE <= board.size <= MAX_BOARD_SIZE

    def find_winning_moves(
        self,
        board: Board,
        player: int,
        candidates: Sequence[Move] | None = None,
    ) -> list[Move] | None:
        library = self._library
        if library is None or not self._supports_board(board):
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
        if library is None or not self._supports_board(board):
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
        if library is None or not self._supports_board(board):
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
        if library is None or not self._supports_board(board):
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

    def classify_defenses(
        self,
        board: Board,
        attacker: int,
        continuations: Sequence[ThreatContinuation],
        counter_wins: Sequence[Move],
        *,
        reply_limit: int | None = None,
        timeout_seconds: float | None = None,
    ) -> NativeDefenseClassification | None:
        """Classify replies without changing the board or production policy."""
        library = self._library
        if (
            library is None
            or not self._defense_classification_available
            or not self._supports_board(board)
        ):
            return None
        if reply_limit is not None and reply_limit < 0:
            raise ValueError("reply_limit 不能小于 0")
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds 不能小于 0")

        grid = self._grid(board)
        winning_point_arrays: list[ctypes.Array[ctypes.c_int32]] = []
        native_continuations: list[_DefenseContinuationV1] = []
        for continuation in continuations:
            encoded_points = tuple(
                row * board.size + column
                for row, column in continuation.winning_points
            )
            if encoded_points:
                point_array = (ctypes.c_int32 * len(encoded_points))(
                    *encoded_points
                )
                winning_point_arrays.append(point_array)
                point_pointer = ctypes.cast(
                    point_array,
                    ctypes.POINTER(ctypes.c_int32),
                )
            else:
                point_pointer = ctypes.POINTER(ctypes.c_int32)()
            native_continuations.append(
                _DefenseContinuationV1(
                    move=(
                        continuation.move[0] * board.size
                        + continuation.move[1]
                    ),
                    immediate_win=int(continuation.immediate_win),
                    winning_points=point_pointer,
                    winning_point_count=len(encoded_points),
                )
            )
        if native_continuations:
            continuation_array = (
                _DefenseContinuationV1 * len(native_continuations)
            )(*native_continuations)
            continuation_pointer = ctypes.cast(
                continuation_array,
                ctypes.POINTER(_DefenseContinuationV1),
            )
        else:
            continuation_array = None
            continuation_pointer = ctypes.POINTER(_DefenseContinuationV1)()

        encoded_counters = tuple(
            row * board.size + column for row, column in counter_wins
        )
        if encoded_counters:
            counter_array = (ctypes.c_int32 * len(encoded_counters))(
                *encoded_counters
            )
            counter_pointer = ctypes.cast(
                counter_array,
                ctypes.POINTER(ctypes.c_int32),
            )
        else:
            counter_array = None
            counter_pointer = ctypes.POINTER(ctypes.c_int32)()

        capacity = max(1, board.empty_count)
        required_output = (ctypes.c_int32 * capacity)()
        refutation_output = (_DefenseRefutationV1 * capacity)()
        unclassified_output = (ctypes.c_int32 * capacity)()
        point_capacity = max(1, board.empty_count * board.empty_count)
        refutation_points = (ctypes.c_int32 * point_capacity)()
        request = _DefenseClassificationRequestV1(
            struct_size=ctypes.sizeof(_DefenseClassificationRequestV1),
            schema_version=DEFENSE_CLASSIFICATION_SCHEMA_VERSION,
            cells=grid,
            board_size=board.size,
            attacker=attacker,
            continuations=continuation_pointer,
            continuation_count=len(native_continuations),
            counter_wins=counter_pointer,
            counter_win_count=len(encoded_counters),
            reply_limit=-1 if reply_limit is None else reply_limit,
            timeout_ms=(
                -1
                if timeout_seconds is None
                else int(timeout_seconds * 1000)
            ),
        )
        result = _DefenseClassificationResultV1(
            struct_size=ctypes.sizeof(_DefenseClassificationResultV1),
            schema_version=DEFENSE_CLASSIFICATION_SCHEMA_VERSION,
            required_defenses=required_output,
            required_capacity=capacity,
            refutations=refutation_output,
            refutation_capacity=capacity,
            unclassified_replies=unclassified_output,
            unclassified_capacity=capacity,
            refutation_winning_points=refutation_points,
            winning_point_capacity=point_capacity,
        )
        status = library.gn_classify_defenses_v1(
            ctypes.byref(request),
            ctypes.byref(result),
        )
        if status not in {
            DEFENSE_CLASSIFICATION_COMPLETE,
            DEFENSE_CLASSIFICATION_CUTOFF,
        }:
            raise RuntimeError(
                f"NativeCore防守分类失败：status={status}"
            )

        def decode(encoded: int) -> Move:
            return encoded // board.size, encoded % board.size

        refutations = tuple(
            NativeDefenseRefutation(
                defense_move=decode(refutation_output[index].defense_move),
                continuation_move=decode(
                    refutation_output[index].continuation_move
                ),
                continuation_is_immediate=bool(
                    refutation_output[index].continuation_is_immediate
                ),
                winning_points=tuple(
                    decode(
                        refutation_points[
                            refutation_output[index].winning_point_offset
                            + point
                        ]
                    )
                    for point in range(
                        refutation_output[index].winning_point_count
                    )
                ),
            )
            for index in range(result.refutation_count)
        )
        return NativeDefenseClassification(
            required_defenses=tuple(
                decode(required_output[index])
                for index in range(result.required_count)
            ),
            counter_wins=tuple(counter_wins),
            refutations=refutations,
            unclassified_replies=tuple(
                decode(unclassified_output[index])
                for index in range(result.unclassified_count)
            ),
            legal_reply_count=result.legal_reply_count,
            refuted_reply_count=result.refutation_count,
            coverage_complete=bool(result.coverage_complete),
            analysis_completed=bool(result.analysis_completed),
            status=result.status,
            cutoff_reason=result.cutoff_reason,
            processed_reply_count=result.processed_reply_count,
        )

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
        if library is None or not self._supports_board(board):
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

    @staticmethod
    def main_search_input_digest(
        board: Board,
        player: int,
        root_candidates: Sequence[Move],
        *,
        depth: int,
        node_limit: int | None,
        branch_candidate_limit: int,
        preselection_factor: int,
        candidate_radius: int,
        recent_move_count: int,
        threat_extension_depth: int,
        flags: int,
    ) -> int:
        values: list[int] = [MAIN_SEARCH_SCHEMA_VERSION, board.size]
        values.extend(cell for row in board.grid for cell in row)
        values.append(len(board.move_history))
        for row, column, stone in board.move_history:
            values.extend((row * board.size + column, stone))
        values.extend((player, len(root_candidates)))
        values.extend(row * board.size + column for row, column in root_candidates)
        values.extend((
            depth,
            0 if node_limit is None else node_limit,
            branch_candidate_limit,
            preselection_factor,
            candidate_radius,
            recent_move_count,
            threat_extension_depth,
            flags,
        ))
        digest = 1_469_598_103_934_665_603
        for value in values:
            for shift in range(0, 64, 8):
                digest ^= (value >> shift) & 0xFF
                digest = (digest * 1_099_511_628_211) & ((1 << 64) - 1)
        return digest

    def probe_main_search_contract(
        self,
        board: Board,
        player: int,
        root_candidates: Sequence[Move],
        *,
        depth: int,
        node_limit: int | None,
        branch_candidate_limit: int,
        preselection_factor: int,
        candidate_radius: int,
        recent_move_count: int,
        threat_extension_depth: int,
        use_pvs: bool,
        use_transposition_table: bool,
    ) -> NativeMainSearchProbe | None:
        library = self._library
        if (
            library is None
            or not self._main_search_available
            or not self._supports_board(board)
        ):
            return None
        if not root_candidates:
            raise ValueError("Native主搜索至少需要一个根候选。")
        grid = self._grid(board)
        history_indices = (ctypes.c_int32 * len(board.move_history))(*(
            row * board.size + column
            for row, column, _stone in board.move_history
        ))
        history_players = (ctypes.c_uint8 * len(board.move_history))(*(
            stone for _row, _column, stone in board.move_history
        ))
        encoded_candidates = (ctypes.c_int32 * len(root_candidates))(*(
            row * board.size + column for row, column in root_candidates
        ))
        flags = (
            (MAIN_SEARCH_FLAG_PVS if use_pvs else 0)
            | (MAIN_SEARCH_FLAG_TT if use_transposition_table else 0)
        )
        request = _MainSearchRequestV1(
            struct_size=ctypes.sizeof(_MainSearchRequestV1),
            schema_version=MAIN_SEARCH_SCHEMA_VERSION,
            cells=grid,
            board_size=board.size,
            history_indices=history_indices,
            history_players=history_players,
            history_count=len(board.move_history),
            player=player,
            root_candidates=encoded_candidates,
            root_candidate_count=len(root_candidates),
            depth=depth,
            node_limit=0 if node_limit is None else node_limit,
            branch_candidate_limit=branch_candidate_limit,
            preselection_factor=preselection_factor,
            candidate_radius=candidate_radius,
            recent_move_count=recent_move_count,
            threat_extension_depth=threat_extension_depth,
            flags=flags,
        )
        root_scores = (ctypes.c_int32 * len(root_candidates))()
        principal_variation = (ctypes.c_int32 * (board.size * board.size))()
        result = _MainSearchResultV1(
            struct_size=ctypes.sizeof(_MainSearchResultV1),
            schema_version=MAIN_SEARCH_SCHEMA_VERSION,
            root_scores=root_scores,
            root_score_capacity=len(root_candidates),
            principal_variation=principal_variation,
            pv_capacity=board.size * board.size,
        )
        status = library.gn_main_search_v1(
            ctypes.byref(request),
            ctypes.byref(result),
        )
        if status == -1:
            raise RuntimeError("Native主搜索ABI请求无效。")
        if not 0 <= result.root_score_count <= len(root_candidates):
            raise RuntimeError("Native主搜索返回了越界的根分数量。")
        if not 0 <= result.pv_length <= board.size * board.size:
            raise RuntimeError("Native主搜索返回了越界的主变化长度。")
        return NativeMainSearchProbe(
            status=status,
            completed_depth=result.completed_depth,
            stop_reason=result.stop_reason,
            best_move=(
                None
                if result.best_move < 0
                else (
                    result.best_move // board.size,
                    result.best_move % board.size,
                )
            ),
            score=result.score,
            nodes=result.nodes,
            tt_entries=result.tt_entries,
            input_digest=result.input_digest,
            tt_digest=result.tt_digest,
            root_scores=tuple(
                (root_candidates[index], root_scores[index])
                for index in range(result.root_score_count)
            ),
            principal_variation=tuple(
                (
                    principal_variation[index] // board.size,
                    principal_variation[index] % board.size,
                )
                for index in range(result.pv_length)
            ),
        )


native_core = NativeCore()


def native_core_status() -> dict[str, object]:
    return native_core.status()
