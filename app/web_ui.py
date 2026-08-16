from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.arena import create_ai, engine_display_name
from engine.arena_settings import AISelection, load_arena_settings
from engine.board import BLACK, WHITE, Board
from engine.evaluator import evaluate_board
from engine.game import format_move, other_player
from engine.records import GameRecorder, RecordPaths
from engine.version import ENGINE_VERSION
from app.ui_common import (
    ClickConfirmation,
    clone_board,
    normalized_ai_selection,
    stone_name,
)


STATIC_FILE = Path(__file__).resolve().parents[1] / "ui" / "gomoku.html"


def _analysis_dict(ai: object) -> dict[str, Any] | None:
    analysis = getattr(ai, "last_analysis", None)
    if analysis is None:
        return None
    if isinstance(analysis, Mapping):
        return dict(analysis)
    to_dict = getattr(analysis, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return dict(value) if isinstance(value, Mapping) else None
    return None


class WebGameController:
    """Thread-safe game controller used by the local browser UI."""

    def __init__(self) -> None:
        defaults = load_arena_settings().white
        self.lock = threading.RLock()
        self.board = Board()
        self.current_player = BLACK
        self.human_player = BLACK
        self.ai_player = WHITE
        self.selection = normalized_ai_selection(
            defaults.engine_name,
            defaults.max_depth,
            defaults.time_limit_seconds,
        )
        self.ai: object | None = None
        self.recorder = self._new_recorder()
        self.confirmation = ClickConfirmation()
        self.status = "请点击棋盘交叉点。再次点击同一点确认落子。"
        self.analysis: dict[str, Any] | None = None
        self.game_over = False
        self.ai_thinking = False
        self.auto_save = True
        self.game_started = time.perf_counter()
        self.turn_started = self.game_started
        self.saved_move_count = 0
        self.last_record_paths: RecordPaths | None = None
        self._token = 0
        self._closed = False
        self._replace_ai()

    def _new_recorder(self) -> GameRecorder:
        ai_name = engine_display_name(self.selection)
        return GameRecorder(
            mode="PVC-WEB-UI",
            black_name="Human" if self.human_player == BLACK else ai_name,
            white_name="Human" if self.human_player == WHITE else ai_name,
        )

    def _replace_ai(self) -> None:
        old_ai = self.ai
        self.ai = create_ai(self.selection, self.ai_player)
        close = getattr(old_ai, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def new_game(
        self,
        *,
        human_player: int,
        engine: str,
        depth: float,
        time_limit: float,
        auto_save: bool,
    ) -> dict[str, Any]:
        if human_player not in (BLACK, WHITE):
            raise ValueError("执子方必须是黑棋或白棋。")
        selection = normalized_ai_selection(engine, depth, time_limit)
        with self.lock:
            if self.ai_thinking:
                raise RuntimeError("AI 正在思考，暂时不能重开。")
            self._token += 1
            self.human_player = human_player
            self.ai_player = other_player(human_player)
            self.selection = selection
            self.auto_save = bool(auto_save)
            self._replace_ai()
            self.board = Board()
            self.current_player = BLACK
            self.recorder = self._new_recorder()
            self.confirmation.cancel()
            self.analysis = None
            self.game_over = False
            self.game_started = time.perf_counter()
            self.turn_started = self.game_started
            self.saved_move_count = 0
            self.last_record_paths = None
            self.status = "新对局已开始。"
            if self.current_player == self.ai_player:
                self._start_ai_locked()
            return self.state()

    def select(self, row: int, column: int) -> dict[str, Any]:
        with self.lock:
            self._require_human_turn()
            if not self.board.is_inside(row, column):
                raise ValueError("落点超出棋盘范围。")
            if not self.board.is_empty(row, column):
                raise ValueError("该交叉点已经有棋子。")
            move = (row, column)
            if self.confirmation.register(move):
                self._commit_human_locked(move)
            else:
                self.status = (
                    f"已预选 {format_move(row, column)}。"
                    "再次点击该点或按确认按钮后落子。"
                )
            return self.state()

    def confirm(self) -> dict[str, Any]:
        with self.lock:
            self._require_human_turn()
            move = self.confirmation.pending
            if move is None:
                raise ValueError("请先在棋盘上选择一个空点。")
            self.confirmation.cancel()
            self._commit_human_locked(move)
            return self.state()

    def cancel(self) -> dict[str, Any]:
        with self.lock:
            self.confirmation.cancel()
            self.status = "已取消预选落点。"
            return self.state()

    def _require_human_turn(self) -> None:
        if self.game_over:
            raise RuntimeError("对局已经结束，请开始新对局。")
        if self.ai_thinking or self.current_player != self.human_player:
            raise RuntimeError("当前不是玩家回合。")

    def _record_move(
        self,
        *,
        move: tuple[int, int],
        player: int,
        actor: str,
        think_seconds: float,
        before: int,
        analysis: dict[str, Any] | None,
    ) -> None:
        row, column = move
        self.recorder.record_move(
            player=player,
            row=row,
            column=column,
            actor=actor,
            think_seconds=max(0.0, think_seconds),
            evaluation_before=before,
            evaluation_after=evaluate_board(self.board, WHITE),
            analysis=analysis,
        )

    def _commit_human_locked(self, move: tuple[int, int]) -> None:
        if not self.board.is_empty(*move):
            raise ValueError("该交叉点已经有棋子。")
        before = evaluate_board(self.board, WHITE)
        elapsed = time.perf_counter() - self.turn_started
        self.board.place(*move, self.human_player)
        self._record_move(
            move=move,
            player=self.human_player,
            actor="Human",
            think_seconds=elapsed,
            before=before,
            analysis=None,
        )
        self.confirmation.cancel()
        if self._finish_if_needed_locked(move, self.human_player):
            return
        self.current_player = self.ai_player
        self.turn_started = time.perf_counter()
        self._start_ai_locked()

    def _start_ai_locked(self) -> None:
        if self.ai is None or self._closed:
            return
        self.ai_thinking = True
        self.status = f"{engine_display_name(self.selection)} 正在思考…"
        token = self._token
        ai = self.ai
        board = clone_board(self.board)
        threading.Thread(
            target=self._ai_worker,
            args=(token, ai, board),
            name="gomoku-ui-ai",
            daemon=True,
        ).start()

    def _ai_worker(self, token: int, ai: object, board: Board) -> None:
        started = time.perf_counter()
        try:
            move = getattr(ai, "choose_move")(board)
            analysis = _analysis_dict(ai)
            error: Exception | None = None
        except Exception as caught:
            move = None
            analysis = None
            error = caught
        elapsed = time.perf_counter() - started
        with self.lock:
            if token != self._token or self._closed:
                if self._closed:
                    close = getattr(ai, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            pass
                return
            self.ai_thinking = False
            if error is not None:
                self.status = f"AI 运行失败：{error}"
                return
            if move is None or not self.board.is_empty(*move):
                self.status = "AI 返回了无效落点，请重开对局。"
                return
            before = evaluate_board(self.board, WHITE)
            self.board.place(*move, self.ai_player)
            self.analysis = analysis
            self._record_move(
                move=move,
                player=self.ai_player,
                actor=engine_display_name(self.selection),
                think_seconds=elapsed,
                before=before,
                analysis=analysis,
            )
            if self._finish_if_needed_locked(move, self.ai_player):
                return
            self.current_player = self.human_player
            self.turn_started = time.perf_counter()
            self.status = (
                f"AI 落子 {format_move(*move)}，轮到你了。"
                "点击两次同一交叉点确认。"
            )

    def _finish_if_needed_locked(
        self,
        move: tuple[int, int],
        player: int,
    ) -> bool:
        result: str | None = None
        if self.board.check_win(*move):
            result = f"{stone_name(player)} 获胜"
        elif self.board.is_full():
            result = "和棋"
        if result is None:
            return False
        self.game_over = True
        self.status = f"对局结束：{result}"
        if self.auto_save:
            self._save_locked(result)
        return True

    def undo(self) -> dict[str, Any]:
        with self.lock:
            if self.ai_thinking:
                raise RuntimeError("AI 正在思考，暂时不能悔棋。")
            if len(self.board.move_history) < 2 or len(self.recorder.moves) < 2:
                raise ValueError("至少完成一个人机回合后才能悔棋。")
            self.confirmation.cancel()
            self.recorder.undo_last_moves(2)
            self.board.undo()
            self.board.undo()
            self.current_player = self.human_player
            self.game_over = False
            self.analysis = None
            self.turn_started = time.perf_counter()
            self.status = "已撤销最近一个人机回合。"
            return self.state()

    def save(self, result: str = "对局进行中") -> dict[str, Any]:
        with self.lock:
            if not self.recorder.moves:
                raise ValueError("当前没有着法可保存。")
            paths = self._save_locked(result)
            return {
                "state": self.state(),
                "paths": {"txt": str(paths.txt), "json": str(paths.json)},
            }

    def _save_locked(self, result: str) -> RecordPaths:
        paths = self.recorder.save(
            board=self.board,
            result=result,
            duration_seconds=time.perf_counter() - self.game_started,
            prefix="pvc-ui",
        )
        self.saved_move_count = len(self.recorder.moves)
        self.last_record_paths = paths
        self.status = f"棋谱已保存：{paths.json.name}"
        return paths

    def state(self) -> dict[str, Any]:
        with self.lock:
            moves = [
                {
                    "number": index + 1,
                    "row": row,
                    "column": column,
                    "player": player,
                    "coordinate": format_move(row, column),
                }
                for index, (row, column, player) in enumerate(
                    self.board.move_history
                )
            ]
            pending = self.confirmation.pending
            evaluation = evaluate_board(self.board, WHITE)
            return {
                "version": ENGINE_VERSION,
                "size": self.board.size,
                "grid": self.board.grid,
                "moves": moves,
                "pending": (
                    {"row": pending[0], "column": pending[1]}
                    if pending is not None
                    else None
                ),
                "current_player": self.current_player,
                "human_player": self.human_player,
                "ai_player": self.ai_player,
                "ai_thinking": self.ai_thinking,
                "game_over": self.game_over,
                "status": self.status,
                "evaluation": evaluation,
                "engine": self.selection.engine_name,
                "depth": self.selection.max_depth,
                "time_limit": self.selection.time_limit_seconds,
                "auto_save": self.auto_save,
                "analysis": self.analysis,
                "saved": self.saved_move_count == len(self.recorder.moves),
            }

    def close(self) -> None:
        with self.lock:
            self._closed = True
            self._token += 1
            ai = self.ai
            self.ai = None
            if not self.ai_thinking:
                close = getattr(ai, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass


class GomokuHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], game: WebGameController) -> None:
        super().__init__(address, GomokuRequestHandler)
        self.game = game
        self.html = STATIC_FILE.read_bytes()


class GomokuRequestHandler(BaseHTTPRequestHandler):
    server: GomokuHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(self.server.html)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(self.server.html)
            return
        if path == "/api/state":
            self._json(self.server.game.state())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 65_536:
                raise ValueError("请求数据过大。")
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("请求必须是 JSON 对象。")
            path = urlparse(self.path).path
            result = self._dispatch(path, payload)
            self._json({"ok": True, "result": result})
        except (ValueError, RuntimeError, OSError) as error:
            self._json(
                {"ok": False, "error": str(error)},
                HTTPStatus.BAD_REQUEST,
            )
        except Exception as error:
            self._json(
                {"ok": False, "error": f"服务器错误：{error}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _dispatch(self, path: str, payload: dict[str, Any]) -> object:
        game = self.server.game
        if path == "/api/select":
            return game.select(int(payload["row"]), int(payload["column"]))
        if path == "/api/confirm":
            return game.confirm()
        if path == "/api/cancel":
            return game.cancel()
        if path == "/api/new":
            side = BLACK if payload.get("side") == "black" else WHITE
            return game.new_game(
                human_player=side,
                engine=str(payload.get("engine", "search")),
                depth=float(payload.get("depth", 3)),
                time_limit=float(payload.get("time_limit", 2)),
                auto_save=bool(payload.get("auto_save", True)),
            )
        if path == "/api/undo":
            return game.undo()
        if path == "/api/save":
            return game.save()
        if path == "/api/shutdown":
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return {"closing": True}
        raise ValueError("未知操作。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gomoku local browser UI")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    game = WebGameController()
    server = GomokuHTTPServer(("127.0.0.1", args.port), game)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Gomoku UI: {url}")
    if not args.no_browser:
        threading.Timer(0.25, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        game.close()
        server.server_close()


if __name__ == "__main__":
    main()
