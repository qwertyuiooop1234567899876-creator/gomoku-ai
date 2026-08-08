from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TextIO

from engine.board import BLACK, WHITE, Board
from engine.evaluator import PositionEvaluation, render_evaluation_bar
from engine.game import format_move


DEFAULT_YIXIN_SETTINGS_PATH = Path("yixin_settings.json")
DEFAULT_YIXIN_EXECUTABLE = Path("yixin") / "engine.exe"

_MOVE_LINE_PATTERN = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*$")
_DETAIL_PATTERN = re.compile(
    r"DEPTH:\s*(\d+)\s*-\s*(\d+).*?"
    r"VAL:\s*([+-]?\d+).*?"
    r"TIME:\s*(\d+)\s*MS.*?"
    r"NODE:\s*(\d+)\s*M",
    re.IGNORECASE,
)
_SUMMARY_DEPTH_PATTERN = re.compile(
    r"DEPTH:\s*(\d+)\s*-\s*(\d+).*?"
    r"TIME:\s*(\d+)\s*MS.*?"
    r"NODE:\s*(\d+)\s*M",
    re.IGNORECASE,
)
_EVALUATION_PATTERN = re.compile(
    r"EVALUATION:\s*([+-]?\d+)",
    re.IGNORECASE,
)
_SPEED_PATTERN = re.compile(r"SPEED:\s*(\d+)", re.IGNORECASE)
_REALTIME_BEST_PATTERN = re.compile(
    r"REALTIME\s+BEST\s*:?\s*(\d+)\s*,\s*(\d+)",
    re.IGNORECASE,
)
_BRACKET_COORDINATE_PATTERN = re.compile(
    r"\[\s*([A-Z])\s*,?\s*(\d+)\s*\]",
    re.IGNORECASE,
)
_PROCESS_EOF = object()


class YixinError(RuntimeError):
    """YiXin 外部引擎基础异常。"""


class YixinConfigurationError(YixinError):
    """YiXin 设置或可执行文件无效。"""


class YixinProtocolError(YixinError):
    """YiXin 返回了无法接受的协议内容。"""


class YixinTimeoutError(YixinError):
    """YiXin 未在约定时间内返回落子。"""


class YixinProcessError(YixinError):
    """YiXin 进程异常退出。"""


@dataclass(frozen=True, slots=True)
class YixinConfig:
    """YiXin 2017 核心的协议和算力设置。"""

    executable_path: str = str(DEFAULT_YIXIN_EXECUTABLE)
    launch_arguments: tuple[str, ...] = ()
    board_size: int = 15
    timeout_turn_seconds: float = 10.0
    evaluation_time_seconds: float = 2.0
    startup_timeout_seconds: float = 5.0
    response_grace_seconds: float = 5.0
    thread_num: int = 2
    thread_split_depth: int = 6
    hash_size: int = 21
    caution_factor: int = 2
    checkmate: int = 0
    rule: int = 0
    pondering: bool = False
    show_detail: bool = True
    use_database: bool = False
    max_depth: int | None = None
    max_node: int | None = None

    def __post_init__(self) -> None:
        if not self.executable_path.strip():
            raise YixinConfigurationError("YiXin executable_path 不能为空。")
        if self.board_size != 15:
            raise YixinConfigurationError(
                "当前程序只支持 15×15 YiXin 对局。"
            )
        if not 0.1 <= self.timeout_turn_seconds <= 3600.0:
            raise YixinConfigurationError(
                "YiXin timeout_turn_seconds 必须在 0.1～3600 秒之间。"
            )
        if not 0.1 <= self.evaluation_time_seconds <= 60.0:
            raise YixinConfigurationError(
                "YiXin evaluation_time_seconds 必须在 0.1～60 秒之间。"
            )
        if not 0.1 <= self.startup_timeout_seconds <= 60.0:
            raise YixinConfigurationError(
                "YiXin startup_timeout_seconds 必须在 0.1～60 秒之间。"
            )
        if not 0.0 <= self.response_grace_seconds <= 60.0:
            raise YixinConfigurationError(
                "YiXin response_grace_seconds 必须在 0～60 秒之间。"
            )
        if not 1 <= self.thread_num <= 256:
            raise YixinConfigurationError(
                "YiXin thread_num 必须在 1～256 之间。"
            )
        if not 0 <= self.thread_split_depth <= 64:
            raise YixinConfigurationError(
                "YiXin thread_split_depth 必须在 0～64 之间。"
            )
        if not 1 <= self.hash_size <= 31:
            raise YixinConfigurationError(
                "YiXin hash_size 必须在 1～31 之间。"
            )
        if not 0 <= self.caution_factor <= 10:
            raise YixinConfigurationError(
                "YiXin caution_factor 必须在 0～10 之间。"
            )
        if self.checkmate not in (0, 1, 2):
            raise YixinConfigurationError(
                "YiXin checkmate 必须是 0、1 或 2。"
            )
        if self.rule not in (0, 1, 2):
            raise YixinConfigurationError(
                "YiXin rule 必须是 0、1 或 2。"
            )
        if self.max_depth is not None and self.max_depth < 1:
            raise YixinConfigurationError(
                "YiXin max_depth 必须大于 0。"
            )
        if self.max_node is not None and self.max_node < 1:
            raise YixinConfigurationError(
                "YiXin max_node 必须大于 0。"
            )

    def with_time_limit(self, seconds: float) -> "YixinConfig":
        return replace(self, timeout_turn_seconds=seconds)

    def resolve_executable(
        self,
        *,
        base_directory: str | Path | None = None,
    ) -> Path:
        path = Path(self.executable_path).expanduser()
        if path.is_absolute():
            return path
        base = (
            Path(base_directory)
            if base_directory is not None
            else Path.cwd()
        )
        return (base / path).resolve()

    def info_commands(self) -> list[str]:
        timeout_ms = max(100, round(self.timeout_turn_seconds * 1000))
        commands = [
            f"INFO timeout_turn {timeout_ms}",
            f"INFO rule {self.rule}",
            f"INFO thread_num {self.thread_num}",
            f"INFO thread_split_depth {self.thread_split_depth}",
            f"INFO hash_size {self.hash_size}",
            f"INFO caution_factor {self.caution_factor}",
            f"INFO checkmate {self.checkmate}",
            f"INFO pondering {int(self.pondering)}",
            f"INFO show_detail {int(self.show_detail)}",
            f"INFO usedatabase {int(self.use_database)}",
        ]
        if self.max_depth is not None:
            commands.append(f"INFO max_depth {self.max_depth}")
        if self.max_node is not None:
            commands.append(f"INFO max_node {self.max_node}")
        return commands

    def to_dict(self) -> dict[str, Any]:
        return {
            "executable_path": self.executable_path,
            "launch_arguments": list(self.launch_arguments),
            "board_size": self.board_size,
            "timeout_turn_seconds": self.timeout_turn_seconds,
            "evaluation_time_seconds": self.evaluation_time_seconds,
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "response_grace_seconds": self.response_grace_seconds,
            "thread_num": self.thread_num,
            "thread_split_depth": self.thread_split_depth,
            "hash_size": self.hash_size,
            "caution_factor": self.caution_factor,
            "checkmate": self.checkmate,
            "rule": self.rule,
            "pondering": self.pondering,
            "show_detail": self.show_detail,
            "use_database": self.use_database,
            "max_depth": self.max_depth,
            "max_node": self.max_node,
        }


def _optional_positive_integer(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def load_yixin_config(
    path: str | Path = DEFAULT_YIXIN_SETTINGS_PATH,
) -> YixinConfig:
    """读取 YiXin 设置；文件不存在时使用可直接运行的默认设置。"""
    settings_path = Path(path)
    defaults = YixinConfig()
    if not settings_path.exists():
        return defaults

    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError
        arguments = payload.get(
            "launch_arguments",
            defaults.launch_arguments,
        )
        if not isinstance(arguments, (list, tuple)):
            raise TypeError
        return YixinConfig(
            executable_path=str(
                payload.get(
                    "executable_path",
                    defaults.executable_path,
                )
            ),
            launch_arguments=tuple(str(item) for item in arguments),
            board_size=int(
                payload.get("board_size", defaults.board_size)
            ),
            timeout_turn_seconds=float(
                payload.get(
                    "timeout_turn_seconds",
                    defaults.timeout_turn_seconds,
                )
            ),
            evaluation_time_seconds=float(
                payload.get(
                    "evaluation_time_seconds",
                    defaults.evaluation_time_seconds,
                )
            ),
            startup_timeout_seconds=float(
                payload.get(
                    "startup_timeout_seconds",
                    defaults.startup_timeout_seconds,
                )
            ),
            response_grace_seconds=float(
                payload.get(
                    "response_grace_seconds",
                    defaults.response_grace_seconds,
                )
            ),
            thread_num=int(
                payload.get("thread_num", defaults.thread_num)
            ),
            thread_split_depth=int(
                payload.get(
                    "thread_split_depth",
                    defaults.thread_split_depth,
                )
            ),
            hash_size=int(
                payload.get("hash_size", defaults.hash_size)
            ),
            caution_factor=int(
                payload.get(
                    "caution_factor",
                    defaults.caution_factor,
                )
            ),
            checkmate=int(
                payload.get("checkmate", defaults.checkmate)
            ),
            rule=int(payload.get("rule", defaults.rule)),
            pondering=bool(
                payload.get("pondering", defaults.pondering)
            ),
            show_detail=bool(
                payload.get("show_detail", defaults.show_detail)
            ),
            use_database=bool(
                payload.get("use_database", defaults.use_database)
            ),
            max_depth=_optional_positive_integer(
                payload.get("max_depth", defaults.max_depth)
            ),
            max_node=_optional_positive_integer(
                payload.get("max_node", defaults.max_node)
            ),
        )
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise YixinConfigurationError(
            f"无法读取 YiXin 设置 {settings_path}: {error}"
        ) from error


def save_yixin_config(
    config: YixinConfig,
    path: str | Path = DEFAULT_YIXIN_SETTINGS_PATH,
) -> Path:
    settings_path = Path(path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = settings_path.with_suffix(
        settings_path.suffix + ".tmp"
    )
    temporary_path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(settings_path)
    return settings_path


def yixin_executable_sha256(
    config: YixinConfig,
    *,
    base_directory: str | Path | None = None,
) -> str | None:
    path = config.resolve_executable(base_directory=base_directory)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yixin_display_coordinate(
    display_column: str,
    display_row: str,
) -> str | None:
    """Convert YiXin DETAIL/BESTLINE coordinates to record coordinates.

    Numeric Gomocup responses use ``x,y``.  YiXin's human-readable bracket
    coordinates use a board display rotated 90 degrees counter-clockwise from
    our records, so those two protocol forms must not share a conversion.
    """
    display_column_index = ord(display_column.upper()) - ord("A")
    display_row_index = int(display_row) - 1
    if not (
        0 <= display_column_index < 15
        and 0 <= display_row_index < 15
    ):
        return None
    record_row = display_column_index
    record_column = 14 - display_row_index
    return format_move(record_row, record_column)


def _coordinate_tokens(line: str) -> list[str]:
    coordinates: list[str] = []
    for display_column, display_row in (
        _BRACKET_COORDINATE_PATTERN.findall(line)
    ):
        coordinate = _yixin_display_coordinate(
            display_column,
            display_row,
        )
        if coordinate is not None:
            coordinates.append(coordinate)
    return coordinates


@dataclass(slots=True)
class YixinSearchReport:
    """一次 YiXin 搜索的结构化输出。"""

    move: tuple[int, int] | None = None
    realtime_move: tuple[int, int] | None = None
    depth: int = 0
    selective_depth: int = 0
    evaluation: int | None = None
    elapsed_ms: int = 0
    nodes: int = 0
    speed: int = 0
    bestline: list[str] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)
    protocol_errors: list[str] = field(default_factory=list)

    @property
    def coordinate(self) -> str | None:
        if self.move is None:
            return None
        return format_move(*self.move)

    @property
    def realtime_coordinate(self) -> str | None:
        if self.realtime_move is None:
            return None
        return format_move(*self.realtime_move)

    @property
    def completed_best_coordinate(self) -> str | None:
        return self.bestline[0] if self.bestline else None

    @property
    def evaluation_aligned_with_move(self) -> bool | None:
        completed = self.completed_best_coordinate
        returned = self.coordinate
        if completed is None or returned is None:
            return None
        return returned == completed

    def consume(self, line: str) -> None:
        normalized = line.strip()
        if not normalized:
            return
        self.raw_lines.append(normalized)

        realtime_match = _REALTIME_BEST_PATTERN.search(normalized)
        if realtime_match:
            column = int(realtime_match.group(1))
            row = int(realtime_match.group(2))
            self.realtime_move = (row, column)

        detail_match = _DETAIL_PATTERN.search(normalized)
        if detail_match:
            self.depth = int(detail_match.group(1))
            self.selective_depth = int(detail_match.group(2))
            self.evaluation = int(detail_match.group(3))
            self.elapsed_ms = int(detail_match.group(4))
            self.nodes = int(detail_match.group(5)) * 1_000_000
            coordinates = _coordinate_tokens(normalized)
            if coordinates:
                self.bestline = coordinates
            return

        depth_match = _SUMMARY_DEPTH_PATTERN.search(normalized)
        if depth_match:
            self.depth = int(depth_match.group(1))
            self.selective_depth = int(depth_match.group(2))
            self.elapsed_ms = int(depth_match.group(3))
            self.nodes = int(depth_match.group(4)) * 1_000_000

        evaluation_match = _EVALUATION_PATTERN.search(normalized)
        if evaluation_match:
            self.evaluation = int(evaluation_match.group(1))

        speed_match = _SPEED_PATTERN.search(normalized)
        if speed_match:
            self.speed = int(speed_match.group(1))

        if "BESTLINE" in normalized.upper():
            coordinates = _coordinate_tokens(normalized)
            if coordinates:
                self.bestline = coordinates

        if normalized.upper().startswith("ERROR"):
            self.protocol_errors.append(normalized)

    def to_analysis_dict(
        self,
        *,
        player: int,
        requested_seconds: float,
    ) -> dict[str, Any]:
        evaluation_white = None
        if self.evaluation is not None:
            evaluation_white = (
                self.evaluation
                if player == WHITE
                else -self.evaluation
            )
        elapsed_seconds = self.elapsed_ms / 1000.0
        completed_best = self.completed_best_coordinate
        evaluation_coordinate = completed_best
        top_coordinate = completed_best or self.coordinate
        return {
            "engine_name": "yixin",
            "engine_version": "2017-kernel-0.6.69",
            "reason": "YiXin 外部核心协议搜索",
            "candidate_count": 1 if self.move is not None else 0,
            "search_depth": self.depth,
            "selective_depth": self.selective_depth,
            "requested_depth": 0,
            "nodes": self.nodes,
            "nps": self.speed * 1000,
            "cutoffs": 0,
            "transposition_hits": 0,
            "transposition_cutoffs": 0,
            "transposition_size": 0,
            "elapsed_seconds": elapsed_seconds,
            "search_completed": self.move is not None,
            "stop_reason": "external_engine_move",
            "time_used_ratio": (
                elapsed_seconds / requested_seconds
                if requested_seconds > 0
                else None
            ),
            "evaluation": self.evaluation,
            "evaluation_perspective": "side_to_move",
            "evaluation_white": evaluation_white,
            "best_coordinate": self.coordinate,
            "returned_coordinate": self.coordinate,
            "realtime_coordinate": self.realtime_coordinate,
            "completed_best_coordinate": completed_best,
            "evaluation_coordinate": evaluation_coordinate,
            "evaluation_aligned_with_returned_move": (
                self.evaluation_aligned_with_move
            ),
            "bestline": list(self.bestline),
            "raw_protocol_lines": list(self.raw_lines),
            "protocol_errors": list(self.protocol_errors),
            "principal_variation": [
                {"coordinate": coordinate}
                for coordinate in self.bestline
            ],
            "top_candidates": (
                [
                    {
                        "coordinate": top_coordinate,
                        "score": self.evaluation or 0,
                        "own_threat": "YiXin 完成层",
                        "opponent_threat": "外部评价",
                    }
                ]
                if top_coordinate is not None
                else []
            ),
        }


class YixinEngine:
    """通过 Gomocup/YiXin 文本协议驱动独立 YiXin 进程。"""

    def __init__(
        self,
        *,
        player: int,
        config: YixinConfig | None = None,
        base_directory: str | Path | None = None,
    ) -> None:
        if player not in (BLACK, WHITE):
            raise ValueError("player 必须是 BLACK 或 WHITE。")
        self.player = player
        self.config = config or YixinConfig()
        self.base_directory = (
            Path(base_directory)
            if base_directory is not None
            else Path.cwd()
        )
        self.last_report: YixinSearchReport | None = None
        self.last_analysis: dict[str, Any] | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._reader_thread: threading.Thread | None = None
        self._lines: queue.Queue[str | object] = queue.Queue()
        self._write_lock = threading.Lock()
        self._closed = False
        self._synced_history: tuple[tuple[int, int, int], ...] | None = None

    @property
    def is_running(self) -> bool:
        return (
            self._process is not None
            and self._process.poll() is None
        )

    def _launch_command(self) -> list[str]:
        executable = self.config.resolve_executable(
            base_directory=self.base_directory
        )
        if not executable.is_file():
            raise YixinConfigurationError(
                "找不到 YiXin 核心："
                f"{executable}。请检查 yixin_settings.json。"
            )
        return [
            str(executable),
            *self.config.launch_arguments,
        ]

    def start(self) -> None:
        if self._closed:
            raise YixinProcessError("YiXin 对象已经关闭，不能重新启动。")
        if self.is_running:
            return

        command = self._launch_command()
        executable_directory = str(Path(command[0]).parent)
        creation_flags = (
            subprocess.CREATE_NO_WINDOW
            if os.name == "nt"
            and hasattr(subprocess, "CREATE_NO_WINDOW")
            else 0
        )
        try:
            self._process = subprocess.Popen(
                command,
                cwd=executable_directory,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                creationflags=creation_flags,
            )
        except OSError as error:
            raise YixinProcessError(
                f"无法启动 YiXin 核心：{error}"
            ) from error

        self._reader_thread = threading.Thread(
            target=self._read_output,
            name="yixin-output-reader",
            daemon=True,
        )
        self._reader_thread.start()
        self._send(f"START {self.config.board_size}")
        self._wait_for_startup()
        for command_text in self.config.info_commands():
            self._send(command_text)

    @staticmethod
    def _decode_line(raw_line: bytes) -> str:
        try:
            return raw_line.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError:
            return raw_line.decode(
                "gb18030",
                errors="replace",
            ).rstrip("\r\n")

    def _read_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            self._lines.put(_PROCESS_EOF)
            return
        try:
            while True:
                raw_line = process.stdout.readline()
                if not raw_line:
                    break
                self._lines.put(self._decode_line(raw_line))
        finally:
            self._lines.put(_PROCESS_EOF)

    def _send(self, command: str) -> None:
        process = self._process
        if (
            process is None
            or process.stdin is None
            or process.poll() is not None
        ):
            raise YixinProcessError("YiXin 进程没有运行。")
        payload = (command + "\n").encode("ascii", errors="strict")
        try:
            with self._write_lock:
                process.stdin.write(payload)
                process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise YixinProcessError(
                f"向 YiXin 发送命令失败：{error}"
            ) from error

    def _next_line(self, timeout_seconds: float) -> str:
        try:
            item = self._lines.get(timeout=timeout_seconds)
        except queue.Empty as error:
            raise YixinTimeoutError("等待 YiXin 输出超时。") from error
        if item is _PROCESS_EOF:
            return_code = (
                self._process.poll()
                if self._process is not None
                else None
            )
            raise YixinProcessError(
                f"YiXin 进程提前退出，退出码：{return_code}。"
            )
        return str(item)

    def _wait_for_startup(self) -> None:
        deadline = (
            time.monotonic() + self.config.startup_timeout_seconds
        )
        received: list[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise YixinTimeoutError(
                    "YiXin 启动后没有返回 OK。"
                    f"已收到：{received[-5:]}"
                )
            line = self._next_line(remaining).strip()
            if line:
                received.append(line)
            if line.upper() == "OK":
                return
            if line.upper().startswith("ERROR"):
                raise YixinProtocolError(
                    f"YiXin 拒绝 START：{line}"
                )

    def _restart_position(self) -> None:
        self._send("RESTART")
        deadline = (
            time.monotonic() + self.config.startup_timeout_seconds
        )
        received: list[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise YixinTimeoutError(
                    "YiXin 没有确认 RESTART。"
                    f"已收到：{received[-5:]}"
                )
            line = self._next_line(remaining).strip()
            if line:
                received.append(line)
            if line.upper() == "OK":
                return
            if line.upper().startswith(("ERROR", "UNKNOWN")):
                raise YixinProtocolError(
                    f"YiXin 拒绝 RESTART：{line}"
                )

    def _drain_completed_output(self) -> None:
        while True:
            try:
                item = self._lines.get_nowait()
            except queue.Empty:
                return
            if item is _PROCESS_EOF:
                self._lines.put(item)
                return

    def _send_board(self, board: Board) -> None:
        self._send("BOARD")
        for row, column, player in board.move_history:
            protocol_player = 1 if player == self.player else 2
            self._send(f"{column},{row},{protocol_player}")
        self._send("DONE")

    def _send_position(self, board: Board) -> None:
        history = tuple(board.move_history)
        synced = self._synced_history
        if (
            synced is not None
            and len(history) == len(synced) + 1
            and history[: len(synced)] == synced
            and history[-1][2] != self.player
        ):
            row, column, _player = history[-1]
            self._send(f"TURN {column},{row}")
            return

        if synced is not None:
            self._restart_position()
            self._synced_history = None
        self._send_board(board)

    def choose_move(self, board: Board) -> tuple[int, int]:
        if board.size != self.config.board_size:
            raise YixinConfigurationError(
                "棋盘尺寸与 YiXin 设置不一致。"
            )
        if board.is_full():
            raise ValueError("棋盘已满，YiXin 无法落子。")

        self.start()
        self._drain_completed_output()
        self._send_position(board)

        report = YixinSearchReport()
        timeout_seconds = (
            self.config.timeout_turn_seconds
            + self.config.response_grace_seconds
        )
        deadline = time.monotonic() + timeout_seconds

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._stop_after_timeout()
                raise YixinTimeoutError(
                    "YiXin 在 "
                    f"{timeout_seconds:g} 秒内没有返回落子。"
                )

            try:
                line = self._next_line(remaining)
            except YixinTimeoutError:
                self._stop_after_timeout()
                raise YixinTimeoutError(
                    "YiXin 在 "
                    f"{timeout_seconds:g} 秒内没有返回落子。"
                ) from None

            stripped = line.strip()
            move_match = _MOVE_LINE_PATTERN.fullmatch(stripped)
            if move_match:
                column = int(move_match.group(1))
                row = int(move_match.group(2))
                if not board.is_inside(row, column):
                    raise YixinProtocolError(
                        "YiXin 返回越界坐标："
                        f"{column},{row}。"
                    )
                if not board.is_empty(row, column):
                    raise YixinProtocolError(
                        "YiXin 返回已占用坐标："
                        f"{format_move(row, column)}。"
                    )
                report.move = (row, column)
                self.last_report = report
                self.last_analysis = report.to_analysis_dict(
                    player=self.player,
                    requested_seconds=self.config.timeout_turn_seconds,
                )
                self._synced_history = (
                    *tuple(board.move_history),
                    (row, column, self.player),
                )
                return row, column

            report.consume(stripped)
            if report.protocol_errors:
                raise YixinProtocolError(
                    f"YiXin 协议错误：{report.protocol_errors[-1]}"
                )

    def _stop_after_timeout(self) -> None:
        if not self.is_running:
            return
        try:
            self._send("YXSTOP")
        except YixinError:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is None:
            return

        if process.poll() is None:
            try:
                self._send("END")
            except YixinError:
                pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)

        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
        if process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass

    def __enter__(self) -> "YixinEngine":
        self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class YixinPositionEvaluator:
    """用独立短时 YiXin 搜索为界面评价当前局面。"""

    def __init__(
        self,
        *,
        config: YixinConfig | None = None,
        base_directory: str | Path | None = None,
    ) -> None:
        selected = config or YixinConfig()
        self.config = selected.with_time_limit(
            selected.evaluation_time_seconds
        )
        self.base_directory = (
            Path(base_directory)
            if base_directory is not None
            else Path.cwd()
        )
        self.last_evaluation: PositionEvaluation | None = None
        self._last_key: (
            tuple[int, tuple[tuple[int, int, int], ...]] | None
        ) = None
        self._configuration_error: str | None = None

    @classmethod
    def from_settings(
        cls,
        path: str | Path = DEFAULT_YIXIN_SETTINGS_PATH,
        *,
        base_directory: str | Path | None = None,
    ) -> "YixinPositionEvaluator":
        try:
            config = load_yixin_config(path)
        except YixinConfigurationError as error:
            evaluator = cls(base_directory=base_directory)
            evaluator._configuration_error = str(error)
            return evaluator
        return cls(
            config=config,
            base_directory=base_directory,
        )

    def evaluate(
        self,
        board: Board,
        current_player: int,
    ) -> PositionEvaluation:
        if current_player not in (BLACK, WHITE):
            raise ValueError("current_player 必须是 BLACK 或 WHITE。")
        if board.is_full():
            raise ValueError("棋盘已满，不需要 YiXin 局面评价。")
        if self._configuration_error is not None:
            raise YixinConfigurationError(self._configuration_error)

        key = (current_player, tuple(board.move_history))
        if key == self._last_key and self.last_evaluation is not None:
            return self.last_evaluation

        engine = YixinEngine(
            player=current_player,
            config=self.config,
            base_directory=self.base_directory,
        )
        try:
            engine.choose_move(board)
            report = engine.last_report
            if report is None or report.evaluation is None:
                raise YixinProtocolError(
                    "YiXin 已返回落子，但没有返回局面评价。"
                )

            score_white = (
                report.evaluation
                if current_player == WHITE
                else -report.evaluation
            )
            evaluation = PositionEvaluation(
                source="YiXin",
                score_white=score_white,
                raw_score=report.evaluation,
                depth=report.depth,
                selective_depth=report.selective_depth,
                elapsed_seconds=report.elapsed_ms / 1000.0,
                best_move=report.move,
                bestline=tuple(report.bestline),
            )
        finally:
            engine.close()

        self._last_key = key
        self.last_evaluation = evaluation
        return evaluation

    def evaluate_for_display(
        self,
        board: Board,
        current_player: int,
    ) -> PositionEvaluation:
        """把 YiXin 故障转换为可显示状态，不中断正在进行的对局。"""
        try:
            return self.evaluate(board, current_player)
        except YixinError as error:
            evaluation = PositionEvaluation(
                source="YiXin",
                score_white=None,
                error=str(error),
            )
            self._last_key = (
                current_player,
                tuple(board.move_history),
            )
            self.last_evaluation = evaluation
            return evaluation


def render_yixin_evaluation_bar(
    evaluator: YixinPositionEvaluator,
    board: Board,
    current_player: int,
    *,
    width: int = 24,
) -> str:
    """以规则终局为最高优先级，其余局面交给 YiXin 评价。"""
    evaluation: PositionEvaluation | None = None
    if not board.is_full():
        is_finished = False
        if board.move_history:
            last_row, last_column, _last_player = board.move_history[-1]
            is_finished = board.check_win(last_row, last_column)
        if not is_finished:
            evaluation = evaluator.evaluate_for_display(
                board,
                current_player,
            )

    return render_evaluation_bar(
        board,
        current_player=current_player,
        width=width,
        position_evaluation=evaluation,
    )
