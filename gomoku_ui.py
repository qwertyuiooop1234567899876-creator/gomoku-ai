from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from collections.abc import Mapping
from dataclasses import dataclass
from tkinter import messagebox, ttk

from arena import create_ai, engine_display_name
from engine.arena_settings import AISelection, load_arena_settings
from engine.board import BLACK, EMPTY, WHITE, Board
from engine.evaluator import evaluate_board
from engine.game import format_move, other_player
from engine.records import GameRecorder, RecordPaths
from engine.version import ENGINE_VERSION
from gomoku_ui_common import (
    ClickConfirmation,
    clone_board,
    normalized_ai_selection,
    stone_name,
)


Move = tuple[int, int]

AI_LABELS = {
    "SearchAI（PVS / VCF / VCT / Proof）": "search",
    "ScoringAI（棋型评分）": "scoring",
    "TacticalAI（胜负与封堵）": "tactical",
    "RandomAI（随机基准）": "random",
    "YiXin（外部引擎）": "yixin",
}
AI_LABEL_BY_NAME = {value: key for key, value in AI_LABELS.items()}

COLORS = {
    "window": "#101722",
    "panel": "#182231",
    "panel_alt": "#202d3d",
    "text": "#ecf2f8",
    "muted": "#9fb0c3",
    "accent": "#46b3a5",
    "accent_hover": "#56c8b8",
    "board": "#d6a85f",
    "board_edge": "#916733",
    "grid": "#4d351d",
    "black": "#18202a",
    "black_edge": "#05080b",
    "white": "#f2f2e9",
    "white_edge": "#8d9297",
    "preview": "#46b3a5",
    "last": "#e64b4b",
}


@dataclass(frozen=True, slots=True)
class BoardGeometry:
    size: int
    width: float
    height: float
    padding: float = 46.0

    @property
    def cell(self) -> float:
        usable = max(1.0, min(self.width, self.height) - self.padding * 2)
        return usable / (self.size - 1)

    @property
    def origin(self) -> tuple[float, float]:
        extent = self.cell * (self.size - 1)
        return (
            (self.width - extent) / 2,
            (self.height - extent) / 2,
        )

    def point(self, row: int, column: int) -> tuple[float, float]:
        origin_x, origin_y = self.origin
        return (
            origin_x + column * self.cell,
            origin_y + row * self.cell,
        )

    def nearest_move(self, x: float, y: float) -> Move | None:
        origin_x, origin_y = self.origin
        column = round((x - origin_x) / self.cell)
        row = round((y - origin_y) / self.cell)
        if not (0 <= row < self.size and 0 <= column < self.size):
            return None
        point_x, point_y = self.point(row, column)
        if max(abs(x - point_x), abs(y - point_y)) > self.cell * 0.44:
            return None
        return row, column


def side_name(player: int) -> str:
    return "黑棋" if player == BLACK else "白棋"


class GomokuApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"Gomoku AI · 点击落子 · V{ENGINE_VERSION}")
        self.root.geometry("1180x800")
        self.root.minsize(980, 700)
        self.root.configure(bg=COLORS["window"])

        self.board = Board()
        self.current_player = BLACK
        self.human_player = BLACK
        self.ai_player = WHITE
        self.ai = None
        self.ai_selection = AISelection()
        self.recorder = GameRecorder(
            mode="PVC-UI",
            black_name="Human",
            white_name="SearchAI",
        )
        self.confirmation = ClickConfirmation()
        self.game_started = time.perf_counter()
        self.turn_started = self.game_started
        self.game_over = False
        self.ai_thinking = False
        self.saved_move_count = 0
        self.last_record_paths: RecordPaths | None = None
        self._game_token = 0
        self._closed = False
        self._ai_results: queue.Queue[
            tuple[int, Move | None, float, Exception | None]
        ] = queue.Queue()

        arena_defaults = load_arena_settings().white
        self.engine_var = tk.StringVar(
            value=AI_LABEL_BY_NAME.get(
                arena_defaults.engine_name,
                AI_LABEL_BY_NAME["search"],
            )
        )
        self.depth_var = tk.DoubleVar(value=arena_defaults.max_depth)
        self.time_var = tk.DoubleVar(value=arena_defaults.time_limit_seconds)
        self.human_side_var = tk.IntVar(value=BLACK)
        self.auto_save_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar()
        self.selection_var = tk.StringVar(value="尚未选择落点")
        self.depth_text_var = tk.StringVar()
        self.time_text_var = tk.StringVar()
        self.score_var = tk.StringVar(value="白方静态评估：0")

        self._configure_style()
        self._build_layout()
        self._bind_shortcuts()
        self._update_slider_labels()
        self._update_engine_controls()
        self._start_new_game(confirm=False)
        self.root.after(80, self._poll_ai_results)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background=COLORS["window"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure(
            "Panel.TLabelframe",
            background=COLORS["panel"],
            foreground=COLORS["text"],
            bordercolor=COLORS["panel_alt"],
            relief="solid",
        )
        style.configure(
            "Panel.TLabelframe.Label",
            background=COLORS["panel"],
            foreground=COLORS["text"],
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure(
            "Panel.TLabel",
            background=COLORS["panel"],
            foreground=COLORS["text"],
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Muted.TLabel",
            background=COLORS["panel"],
            foreground=COLORS["muted"],
            font=("Microsoft YaHei UI", 8),
        )
        style.configure(
            "Status.TLabel",
            background=COLORS["panel_alt"],
            foreground=COLORS["text"],
            padding=(10, 8),
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure(
            "Accent.TButton",
            background=COLORS["accent"],
            foreground="#08110f",
            padding=(10, 7),
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[
                ("disabled", COLORS["panel_alt"]),
                ("active", COLORS["accent_hover"]),
            ],
            foreground=[("disabled", COLORS["muted"])],
        )
        style.configure(
            "Panel.TButton",
            background=COLORS["panel_alt"],
            foreground=COLORS["text"],
            padding=(9, 7),
            font=("Microsoft YaHei UI", 9),
        )
        style.map(
            "Panel.TButton",
            background=[
                ("disabled", COLORS["panel_alt"]),
                ("active", "#2b3d50"),
            ],
            foreground=[("disabled", "#66788a")],
        )
        style.configure(
            "Panel.TCombobox",
            fieldbackground=COLORS["panel_alt"],
            background=COLORS["panel_alt"],
            foreground=COLORS["text"],
            arrowcolor=COLORS["muted"],
            bordercolor="#33465b",
            lightcolor="#33465b",
            darkcolor="#33465b",
        )
        style.map(
            "Panel.TCombobox",
            fieldbackground=[("readonly", COLORS["panel_alt"])],
            foreground=[("readonly", COLORS["text"])],
            selectbackground=[("readonly", COLORS["panel_alt"])],
            selectforeground=[("readonly", COLORS["text"])],
        )
        style.configure(
            "Panel.TRadiobutton",
            background=COLORS["panel"],
            foreground=COLORS["text"],
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Panel.TCheckbutton",
            background=COLORS["panel"],
            foreground=COLORS["text"],
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Treeview",
            background=COLORS["panel_alt"],
            fieldbackground=COLORS["panel_alt"],
            foreground=COLORS["text"],
            rowheight=26,
            borderwidth=0,
            font=("Consolas", 9),
        )
        style.configure(
            "Treeview.Heading",
            background=COLORS["panel"],
            foreground=COLORS["muted"],
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        style.map("Treeview", background=[("selected", "#31586a")])
        style.configure(
            "AI.Horizontal.TProgressbar",
            troughcolor=COLORS["panel_alt"],
            background=COLORS["accent"],
            lightcolor=COLORS["accent"],
            darkcolor=COLORS["accent"],
            bordercolor=COLORS["panel_alt"],
            thickness=8,
        )
        style.configure(
            "Panel.TNotebook",
            background=COLORS["panel"],
            bordercolor=COLORS["panel_alt"],
        )
        style.configure(
            "Panel.TNotebook.Tab",
            background=COLORS["panel_alt"],
            foreground=COLORS["muted"],
            padding=(10, 5),
        )
        style.map(
            "Panel.TNotebook.Tab",
            background=[("selected", "#2b3d50")],
            foreground=[("selected", COLORS["text"])],
        )

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, style="App.TFrame", padding=14)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)

        header = ttk.Frame(outer, style="App.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Label(
            header,
            text="GOMOKU  ·  HUMAN vs AI",
            background=COLORS["window"],
            foreground=COLORS["text"],
            font=("Segoe UI", 18, "bold"),
        ).pack(side="left")
        ttk.Label(
            header,
            text="第一次点击预览，第二次点击同一点确认",
            background=COLORS["window"],
            foreground=COLORS["muted"],
            font=("Microsoft YaHei UI", 9),
        ).pack(side="right", pady=(8, 0))

        board_panel = ttk.Frame(outer, style="Panel.TFrame", padding=10)
        board_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        board_panel.rowconfigure(0, weight=1)
        board_panel.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            board_panel,
            background=COLORS["board_edge"],
            highlightthickness=0,
            cursor="hand2",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_board_click)
        self.canvas.bind("<Configure>", lambda _event: self._draw_board())

        action_bar = ttk.Frame(board_panel, style="Panel.TFrame")
        action_bar.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        action_bar.columnconfigure(0, weight=1)
        ttk.Label(
            action_bar,
            textvariable=self.selection_var,
            style="Panel.TLabel",
        ).grid(row=0, column=0, sticky="w", padx=(2, 8))
        self.cancel_button = ttk.Button(
            action_bar,
            text="取消选择  Esc",
            style="Panel.TButton",
            command=self._cancel_pending,
        )
        self.cancel_button.grid(row=0, column=1, padx=4)
        self.confirm_button = ttk.Button(
            action_bar,
            text="确认落子  Enter",
            style="Accent.TButton",
            command=self._confirm_pending,
        )
        self.confirm_button.grid(row=0, column=2, padx=(4, 0))

        sidebar = ttk.Frame(outer, style="App.TFrame", width=350)
        sidebar.grid(row=1, column=1, sticky="ns")
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(3, weight=1)

        self._build_settings_panel(sidebar)

        status = ttk.Label(
            sidebar,
            textvariable=self.status_var,
            style="Status.TLabel",
            wraplength=320,
            justify="left",
        )
        status.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        progress_frame = ttk.Frame(sidebar, style="Panel.TFrame", padding=(8, 7))
        progress_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        progress_frame.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(
            progress_frame,
            mode="indeterminate",
            style="AI.Horizontal.TProgressbar",
        )
        self.progress.grid(row=0, column=0, sticky="ew")
        self.progress.grid_remove()
        ttk.Label(
            progress_frame,
            textvariable=self.score_var,
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(5, 0))

        notebook = ttk.Notebook(sidebar, style="Panel.TNotebook")
        notebook.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        moves_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=6)
        analysis_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=6)
        notebook.add(moves_tab, text="着法")
        notebook.add(analysis_tab, text="AI 分析")
        moves_tab.rowconfigure(0, weight=1)
        moves_tab.columnconfigure(0, weight=1)
        analysis_tab.rowconfigure(0, weight=1)
        analysis_tab.columnconfigure(0, weight=1)

        self.move_tree = ttk.Treeview(
            moves_tab,
            columns=("round", "black", "white"),
            show="headings",
            selectmode="browse",
        )
        self.move_tree.heading("round", text="回合")
        self.move_tree.heading("black", text="黑棋")
        self.move_tree.heading("white", text="白棋")
        self.move_tree.column("round", width=55, anchor="center")
        self.move_tree.column("black", width=90, anchor="center")
        self.move_tree.column("white", width=90, anchor="center")
        self.move_tree.grid(row=0, column=0, sticky="nsew")
        move_scroll = ttk.Scrollbar(
            moves_tab,
            orient="vertical",
            command=self.move_tree.yview,
        )
        move_scroll.grid(row=0, column=1, sticky="ns")
        self.move_tree.configure(yscrollcommand=move_scroll.set)

        self.analysis_text = tk.Text(
            analysis_tab,
            background=COLORS["panel_alt"],
            foreground=COLORS["text"],
            insertbackground=COLORS["text"],
            relief="flat",
            wrap="word",
            padx=9,
            pady=8,
            font=("Microsoft YaHei UI", 9),
            state="disabled",
        )
        self.analysis_text.grid(row=0, column=0, sticky="nsew")
        analysis_scroll = ttk.Scrollbar(
            analysis_tab,
            orient="vertical",
            command=self.analysis_text.yview,
        )
        analysis_scroll.grid(row=0, column=1, sticky="ns")
        self.analysis_text.configure(yscrollcommand=analysis_scroll.set)

    def _build_settings_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(
            parent,
            text="本局设置",
            style="Panel.TLabelframe",
            padding=10,
        )
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text="玩家执棋", style="Panel.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        side_row = ttk.Frame(panel, style="Panel.TFrame")
        side_row.grid(row=1, column=0, sticky="ew", pady=(3, 8))
        ttk.Radiobutton(
            side_row,
            text="黑棋先手",
            variable=self.human_side_var,
            value=BLACK,
            style="Panel.TRadiobutton",
        ).pack(side="left")
        ttk.Radiobutton(
            side_row,
            text="白棋后手",
            variable=self.human_side_var,
            value=WHITE,
            style="Panel.TRadiobutton",
        ).pack(side="left", padx=(12, 0))

        ttk.Label(panel, text="AI 类别", style="Panel.TLabel").grid(
            row=2,
            column=0,
            sticky="w",
        )
        self.engine_combo = ttk.Combobox(
            panel,
            textvariable=self.engine_var,
            values=tuple(AI_LABELS),
            state="readonly",
            style="Panel.TCombobox",
            font=("Microsoft YaHei UI", 9),
        )
        self.engine_combo.grid(row=3, column=0, sticky="ew", pady=(3, 8))
        self.engine_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._update_engine_controls(),
        )

        slider_header = ttk.Frame(panel, style="Panel.TFrame")
        slider_header.grid(row=4, column=0, sticky="ew")
        slider_header.columnconfigure(0, weight=1)
        ttk.Label(slider_header, text="搜索深度", style="Panel.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(
            slider_header,
            textvariable=self.depth_text_var,
            style="Muted.TLabel",
        ).grid(row=0, column=1, sticky="e")
        self.depth_scale = tk.Scale(
            panel,
            from_=1,
            to=8,
            resolution=1,
            orient="horizontal",
            variable=self.depth_var,
            command=lambda _value: self._update_slider_labels(),
            showvalue=False,
            background=COLORS["panel"],
            foreground=COLORS["text"],
            troughcolor=COLORS["panel_alt"],
            activebackground=COLORS["accent"],
            highlightthickness=0,
            bd=0,
        )
        self.depth_scale.grid(row=5, column=0, sticky="ew", pady=(0, 5))

        time_header = ttk.Frame(panel, style="Panel.TFrame")
        time_header.grid(row=6, column=0, sticky="ew")
        time_header.columnconfigure(0, weight=1)
        ttk.Label(time_header, text="单步时间", style="Panel.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(
            time_header,
            textvariable=self.time_text_var,
            style="Muted.TLabel",
        ).grid(row=0, column=1, sticky="e")
        self.time_scale = tk.Scale(
            panel,
            from_=0.5,
            to=60.0,
            resolution=0.5,
            orient="horizontal",
            variable=self.time_var,
            command=lambda _value: self._update_slider_labels(),
            showvalue=False,
            background=COLORS["panel"],
            foreground=COLORS["text"],
            troughcolor=COLORS["panel_alt"],
            activebackground=COLORS["accent"],
            highlightthickness=0,
            bd=0,
        )
        self.time_scale.grid(row=7, column=0, sticky="ew", pady=(0, 6))

        ttk.Checkbutton(
            panel,
            text="对局结束时自动保存新棋谱",
            variable=self.auto_save_var,
            style="Panel.TCheckbutton",
        ).grid(row=8, column=0, sticky="w", pady=(2, 8))
        ttk.Label(
            panel,
            text="参数修改将在“新对局”时生效",
            style="Muted.TLabel",
        ).grid(row=9, column=0, sticky="w", pady=(0, 8))

        buttons = ttk.Frame(panel, style="Panel.TFrame")
        buttons.grid(row=10, column=0, sticky="ew")
        for column in range(2):
            buttons.columnconfigure(column, weight=1)
        self.new_button = ttk.Button(
            buttons,
            text="新对局",
            style="Accent.TButton",
            command=self._start_new_game,
        )
        self.new_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.undo_button = ttk.Button(
            buttons,
            text="悔棋一回合",
            style="Panel.TButton",
            command=self._undo_round,
        )
        self.undo_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.save_button = ttk.Button(
            buttons,
            text="保存棋谱",
            style="Panel.TButton",
            command=self._save_record_interactive,
        )
        self.save_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Return>", lambda _event: self._confirm_pending())
        self.root.bind("<Escape>", lambda _event: self._cancel_pending())
        self.root.bind("<Control-n>", lambda _event: self._start_new_game())
        self.root.bind("<Control-s>", lambda _event: self._save_record_interactive())
        self.root.bind("<Control-z>", lambda _event: self._undo_round())

    def _geometry(self) -> BoardGeometry:
        return BoardGeometry(
            size=self.board.size,
            width=max(1, self.canvas.winfo_width()),
            height=max(1, self.canvas.winfo_height()),
        )

    def _draw_board(self) -> None:
        if not hasattr(self, "canvas"):
            return
        canvas = self.canvas
        canvas.delete("all")
        geometry = self._geometry()
        origin_x, origin_y = geometry.origin
        extent = geometry.cell * (self.board.size - 1)
        inset = geometry.cell * 0.62

        canvas.create_rectangle(
            origin_x - inset,
            origin_y - inset,
            origin_x + extent + inset,
            origin_y + extent + inset,
            fill=COLORS["board"],
            outline=COLORS["board_edge"],
            width=3,
        )
        for index in range(self.board.size):
            x, _ = geometry.point(0, index)
            _, y = geometry.point(index, 0)
            canvas.create_line(
                x,
                origin_y,
                x,
                origin_y + extent,
                fill=COLORS["grid"],
                width=1,
            )
            canvas.create_line(
                origin_x,
                y,
                origin_x + extent,
                y,
                fill=COLORS["grid"],
                width=1,
            )
            canvas.create_text(
                x,
                origin_y - inset * 0.62,
                text=chr(ord("A") + index),
                fill=COLORS["grid"],
                font=("Segoe UI", max(8, int(geometry.cell * 0.25)), "bold"),
            )
            canvas.create_text(
                origin_x - inset * 0.64,
                y,
                text=str(index + 1),
                fill=COLORS["grid"],
                font=("Segoe UI", max(8, int(geometry.cell * 0.23)), "bold"),
            )

        star_radius = max(2.5, geometry.cell * 0.09)
        for row, column in ((3, 3), (3, 11), (7, 7), (11, 3), (11, 11)):
            x, y = geometry.point(row, column)
            canvas.create_oval(
                x - star_radius,
                y - star_radius,
                x + star_radius,
                y + star_radius,
                fill=COLORS["grid"],
                outline="",
            )

        last_move = (
            self.board.move_history[-1][:2]
            if self.board.move_history
            else None
        )
        stone_radius = geometry.cell * 0.42
        for row in range(self.board.size):
            for column in range(self.board.size):
                player = self.board.grid[row][column]
                if player == EMPTY:
                    continue
                self._draw_stone(
                    row,
                    column,
                    player,
                    stone_radius,
                    geometry,
                )
                if last_move == (row, column):
                    x, y = geometry.point(row, column)
                    marker = stone_radius * 0.26
                    canvas.create_oval(
                        x - marker,
                        y - marker,
                        x + marker,
                        y + marker,
                        outline=COLORS["last"],
                        width=max(2, int(geometry.cell * 0.06)),
                    )

        if self.confirmation.pending is not None:
            row, column = self.confirmation.pending
            x, y = geometry.point(row, column)
            canvas.create_oval(
                x - stone_radius,
                y - stone_radius,
                x + stone_radius,
                y + stone_radius,
                fill=(COLORS["black"] if self.human_player == BLACK else COLORS["white"]),
                outline=COLORS["preview"],
                width=3,
                dash=(5, 3),
                stipple="gray50",
            )
            canvas.create_text(
                x,
                y,
                text="✓",
                fill=COLORS["preview"],
                font=("Segoe UI Symbol", max(10, int(geometry.cell * 0.38)), "bold"),
            )

    def _draw_stone(
        self,
        row: int,
        column: int,
        player: int,
        radius: float,
        geometry: BoardGeometry,
    ) -> None:
        x, y = geometry.point(row, column)
        fill = COLORS["black"] if player == BLACK else COLORS["white"]
        outline = (
            COLORS["black_edge"] if player == BLACK else COLORS["white_edge"]
        )
        shadow = max(1.5, radius * 0.10)
        self.canvas.create_oval(
            x - radius + shadow,
            y - radius + shadow,
            x + radius + shadow,
            y + radius + shadow,
            fill="#6f512e",
            outline="",
        )
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=fill,
            outline=outline,
            width=max(1, int(radius * 0.08)),
        )
        highlight = radius * 0.28
        self.canvas.create_oval(
            x - radius * 0.48,
            y - radius * 0.52,
            x - radius * 0.48 + highlight,
            y - radius * 0.52 + highlight,
            fill=("#3f4a55" if player == BLACK else "#ffffff"),
            outline="",
        )

    def _on_board_click(self, event: tk.Event) -> None:
        if self.game_over:
            self._set_status("本局已经结束，请开始新对局。")
            return
        if self.ai_thinking or self.current_player != self.human_player:
            self._set_status("现在是 AI 回合，请等待 AI 完成思考。")
            return
        move = self._geometry().nearest_move(float(event.x), float(event.y))
        if move is None:
            return
        if not self.board.is_empty(*move):
            self._set_status(f"{format_move(*move)} 已有棋子，请选择空位。")
            return

        confirmed = self.confirmation.register(move)
        if confirmed:
            self._commit_human_move(move)
            return

        coordinate = format_move(*move)
        self.selection_var.set(
            f"预落子：{coordinate} · 再点同一点或按 Enter 确认"
        )
        self._set_status(f"已选择 {coordinate}，尚未正式落子。")
        self._update_action_buttons()
        self._draw_board()

    def _confirm_pending(self) -> None:
        move = self.confirmation.pending
        if move is None:
            self._set_status("请先在棋盘上选择一个空位。")
            return
        self.confirmation.cancel()
        self._commit_human_move(move)

    def _cancel_pending(self) -> None:
        if self.confirmation.pending is None:
            return
        self.confirmation.cancel()
        self.selection_var.set("尚未选择落点")
        self._set_status("已取消预落子。")
        self._update_action_buttons()
        self._draw_board()

    def _commit_human_move(self, move: Move) -> None:
        if (
            self.game_over
            or self.ai_thinking
            or self.current_player != self.human_player
            or not self.board.is_empty(*move)
        ):
            return
        think_seconds = max(0.0, time.perf_counter() - self.turn_started)
        self._place_and_record(
            move,
            self.human_player,
            actor="Human",
            think_seconds=think_seconds,
            analysis=None,
        )
        self.confirmation.cancel()
        self.selection_var.set("尚未选择落点")
        if self._finish_turn(move, self.human_player):
            return
        self.current_player = self.ai_player
        self._refresh_all()
        self.root.after(180, self._start_ai_turn)

    def _start_ai_turn(self) -> None:
        if (
            self._closed
            or self.game_over
            or self.ai_thinking
            or self.current_player != self.ai_player
        ):
            return
        self.ai_thinking = True
        self._set_status(
            f"{stone_name(self.ai_player)} · {engine_display_name(self.ai_selection)} 正在思考…"
        )
        self.progress.grid()
        self.progress.start(12)
        self._update_action_buttons()
        token = self._game_token
        ai = self.ai
        board = clone_board(self.board)

        def worker() -> None:
            started = time.perf_counter()
            try:
                if ai is None:
                    raise RuntimeError("AI 尚未创建。")
                move = ai.choose_move(board)
                error = None
            except Exception as caught:  # UI boundary: surface engine errors.
                move = None
                error = caught
            self._ai_results.put(
                (token, move, time.perf_counter() - started, error)
            )

        threading.Thread(
            target=worker,
            name="gomoku-ai-turn",
            daemon=True,
        ).start()

    def _poll_ai_results(self) -> None:
        if self._closed:
            return
        while True:
            try:
                result = self._ai_results.get_nowait()
            except queue.Empty:
                break
            self._handle_ai_result(*result)
        self.root.after(80, self._poll_ai_results)

    def _handle_ai_result(
        self,
        token: int,
        move: Move | None,
        think_seconds: float,
        error: Exception | None,
    ) -> None:
        if token != self._game_token or self._closed:
            return
        self.ai_thinking = False
        self.progress.stop()
        self.progress.grid_remove()
        if error is not None:
            self._set_status(f"AI 运行失败：{error}")
            self._update_action_buttons()
            messagebox.showerror("AI 运行失败", str(error), parent=self.root)
            return
        if move is None or not self.board.is_empty(*move):
            self._set_status("AI 返回了无效落点，本局已暂停。")
            self._update_action_buttons()
            return

        analysis = self._analysis_payload(self.ai)
        self._place_and_record(
            move,
            self.ai_player,
            actor=engine_display_name(self.ai_selection),
            think_seconds=think_seconds,
            analysis=analysis,
        )
        self._render_analysis(analysis, move, think_seconds)
        if self._finish_turn(move, self.ai_player):
            return
        self.current_player = self.human_player
        self.turn_started = time.perf_counter()
        self._set_status(
            f"AI 落子 {format_move(*move)}。轮到你执{side_name(self.human_player)}。"
        )
        self._refresh_all()

    @staticmethod
    def _analysis_payload(ai: object | None) -> dict[str, object] | None:
        analysis = getattr(ai, "last_analysis", None)
        if analysis is None:
            return None
        if isinstance(analysis, Mapping):
            return dict(analysis)
        to_dict = getattr(analysis, "to_dict", None)
        if not callable(to_dict):
            return None
        payload = to_dict()
        return dict(payload) if isinstance(payload, Mapping) else None

    def _place_and_record(
        self,
        move: Move,
        player: int,
        *,
        actor: str,
        think_seconds: float,
        analysis: dict[str, object] | None,
    ) -> None:
        before = evaluate_board(self.board, WHITE)
        self.board.place(*move, player)
        after = evaluate_board(self.board, WHITE)
        self.recorder.record_move(
            player=player,
            row=move[0],
            column=move[1],
            actor=actor,
            think_seconds=think_seconds,
            evaluation_before=before,
            evaluation_after=after,
            analysis=analysis,
        )

    def _finish_turn(self, move: Move, player: int) -> bool:
        if self.board.check_win(*move):
            self.game_over = True
            result = f"{stone_name(player)}获胜"
            self._set_status(f"{result} · 共 {len(self.board.move_history)} 手")
            self._refresh_all()
            if self.auto_save_var.get():
                self._save_record(result, notify=False)
            messagebox.showinfo("对局结束", result, parent=self.root)
            return True
        if self.board.is_full():
            self.game_over = True
            self._set_status("棋盘已满，本局和棋。")
            self._refresh_all()
            if self.auto_save_var.get():
                self._save_record("和棋", notify=False)
            messagebox.showinfo("对局结束", "本局和棋。", parent=self.root)
            return True
        return False

    def _start_new_game(self, confirm: bool = True) -> None:
        if self.ai_thinking:
            messagebox.showinfo(
                "AI 正在思考",
                "请等待当前 AI 回合结束后再开始新对局。",
                parent=self.root,
            )
            return
        if confirm and self.recorder.moves and self.saved_move_count < len(self.recorder.moves):
            answer = messagebox.askyesnocancel(
                "开始新对局",
                "当前棋谱尚未保存。是否保存后再开始新对局？",
                parent=self.root,
            )
            if answer is None:
                return
            if answer:
                self._save_record("重新开局", notify=False)

        self._close_ai()
        self._game_token += 1
        self.board = Board()
        self.current_player = BLACK
        self.human_player = int(self.human_side_var.get())
        self.ai_player = other_player(self.human_player)
        engine_name = AI_LABELS.get(self.engine_var.get(), "search")
        self.ai_selection = normalized_ai_selection(
            engine_name,
            self.depth_var.get(),
            self.time_var.get(),
        )
        try:
            self.ai = create_ai(self.ai_selection, self.ai_player)
            ai_name = engine_display_name(self.ai_selection)
        except Exception as error:
            self.ai = None
            self._set_status(f"无法创建 AI：{error}")
            messagebox.showerror("无法创建 AI", str(error), parent=self.root)
            return

        self.recorder = GameRecorder(
            mode="PVC-UI",
            black_name=("Human" if self.human_player == BLACK else ai_name),
            white_name=("Human" if self.human_player == WHITE else ai_name),
        )
        self.confirmation.cancel()
        self.game_started = time.perf_counter()
        self.turn_started = self.game_started
        self.game_over = False
        self.saved_move_count = 0
        self.last_record_paths = None
        self.selection_var.set("尚未选择落点")
        self._render_analysis(None, None, 0.0)
        self._set_status(
            f"新对局：你执{side_name(self.human_player)}，"
            f"AI 使用 {ai_name}。"
        )
        self._refresh_all()
        if self.ai_player == BLACK:
            self.root.after(250, self._start_ai_turn)

    def _undo_round(self) -> None:
        if self.ai_thinking:
            self._set_status("AI 思考期间不能悔棋。")
            return
        if len(self.board.move_history) < 2:
            self._set_status("至少完成一个人机回合后才能悔棋。")
            return
        if len(self.recorder.moves) < 2:
            self._set_status("棋盘与棋谱记录不一致，悔棋已停止。")
            return
        self.confirmation.cancel()
        self.recorder.undo_last_moves(2)
        for _ in range(2):
            self.board.undo()
        self.game_over = False
        self.current_player = self.human_player
        self.turn_started = time.perf_counter()
        self.selection_var.set("尚未选择落点")
        self._set_status("已撤销最近一个人机回合，轮到你重新落子。")
        self._render_analysis(None, None, 0.0)
        self._refresh_all()

    def _save_record_interactive(self) -> None:
        if not self.recorder.moves:
            self._set_status("当前没有着法可保存。")
            return
        result = "对局进行中"
        if self.game_over and self.board.move_history:
            last = self.board.move_history[-1]
            result = (
                f"{stone_name(last[2])}获胜"
                if self.board.check_win(last[0], last[1])
                else "和棋"
            )
        self._save_record(result, notify=True)

    def _save_record(self, result: str, *, notify: bool) -> RecordPaths | None:
        if not self.recorder.moves:
            return None
        try:
            paths = self.recorder.save(
                board=self.board,
                result=result,
                duration_seconds=time.perf_counter() - self.game_started,
                prefix="pvc-ui",
            )
        except OSError as error:
            self._set_status(f"保存棋谱失败：{error}")
            if notify:
                messagebox.showerror("保存失败", str(error), parent=self.root)
            return None
        self.saved_move_count = len(self.recorder.moves)
        self.last_record_paths = paths
        self._set_status(f"棋谱已保存：{paths.json.name}")
        if notify:
            messagebox.showinfo(
                "棋谱已保存",
                f"TXT：{paths.txt}\nJSON：{paths.json}",
                parent=self.root,
            )
        return paths

    def _render_analysis(
        self,
        analysis: dict[str, object] | None,
        move: Move | None,
        think_seconds: float,
    ) -> None:
        lines: list[str] = []
        if move is not None:
            lines.append(f"AI 落子：{format_move(*move)}")
            lines.append(f"实际耗时：{think_seconds:.3f}s")
        if analysis:
            reason = analysis.get("reason")
            if reason:
                lines.extend(("", "决策说明", str(reason)))
            depth = analysis.get("search_depth", 0)
            requested = analysis.get("requested_depth", depth)
            nodes = analysis.get("nodes", 0)
            nps = analysis.get("nps", 0)
            lines.extend(
                (
                    "",
                    f"深度：{depth}/{requested}",
                    f"节点：{int(nodes):,}" if isinstance(nodes, int) else f"节点：{nodes}",
                    f"NPS：{int(nps):,}" if isinstance(nps, int) else f"NPS：{nps}",
                    f"停止：{analysis.get('stop_reason', '—')}",
                )
            )
            if analysis.get("proof_checked"):
                lines.append(f"Proof：{analysis.get('proof_state', 'unknown')}")
            if analysis.get("final_proof_checked"):
                lines.append(
                    "Final Proof："
                    f"{analysis.get('final_proof_state', 'unknown')}"
                )
            pv = analysis.get("principal_variation", [])
            if isinstance(pv, list) and pv:
                coordinates = [
                    str(item.get("coordinate", "?"))
                    for item in pv
                    if isinstance(item, Mapping)
                ]
                if coordinates:
                    lines.extend(("", "主变化", " → ".join(coordinates)))
            candidates = analysis.get("top_candidates", [])
            if isinstance(candidates, list) and candidates:
                lines.extend(("", "候选着"))
                for index, item in enumerate(candidates[:5], start=1):
                    if not isinstance(item, Mapping):
                        continue
                    lines.append(
                        f"{index}. {item.get('coordinate', '?')}  "
                        f"score={item.get('score', 0):+}"
                    )
        if not lines:
            lines = [
                "AI 分析将在电脑落子后显示。",
                "",
                "SearchAI 会展示搜索深度、节点数、Proof 状态、主变化和候选着。",
            ]
        self.analysis_text.configure(state="normal")
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", "\n".join(lines))
        self.analysis_text.configure(state="disabled")

    def _refresh_move_tree(self) -> None:
        for item in self.move_tree.get_children():
            self.move_tree.delete(item)
        moves = self.recorder.moves
        for index in range(0, len(moves), 2):
            black = moves[index].coordinate
            white = moves[index + 1].coordinate if index + 1 < len(moves) else ""
            item = self.move_tree.insert(
                "",
                "end",
                values=(index // 2 + 1, black, white),
            )
            self.move_tree.see(item)

    def _refresh_all(self) -> None:
        self._draw_board()
        self._refresh_move_tree()
        self.score_var.set(
            f"白方静态评估：{evaluate_board(self.board, WHITE):+,}"
        )
        self._update_action_buttons()

    def _update_action_buttons(self) -> None:
        human_turn = (
            not self.game_over
            and not self.ai_thinking
            and self.current_player == self.human_player
        )
        pending = self.confirmation.pending is not None
        self.confirm_button.configure(
            state=("normal" if human_turn and pending else "disabled")
        )
        self.cancel_button.configure(
            state=("normal" if pending and not self.ai_thinking else "disabled")
        )
        self.undo_button.configure(
            state=(
                "normal"
                if not self.ai_thinking and len(self.board.move_history) >= 2
                else "disabled"
            )
        )
        self.new_button.configure(
            state=("disabled" if self.ai_thinking else "normal")
        )
        self.save_button.configure(
            state=("normal" if self.recorder.moves else "disabled")
        )

    def _update_slider_labels(self) -> None:
        self.depth_text_var.set(f"{int(round(self.depth_var.get()))} 层")
        self.time_text_var.set(f"{self.time_var.get():g} 秒")

    def _update_engine_controls(self) -> None:
        engine = AI_LABELS.get(self.engine_var.get(), "search")
        self.depth_scale.configure(
            state=("normal" if engine == "search" else "disabled")
        )
        self.time_scale.configure(
            state=("normal" if engine in {"search", "yixin"} else "disabled")
        )
        self._update_slider_labels()

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _close_ai(self) -> None:
        ai = self.ai
        self.ai = None
        close = getattr(ai, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _on_close(self) -> None:
        if self.ai_thinking:
            if not messagebox.askyesno(
                "AI 正在思考",
                "AI 仍在思考。确定立即关闭窗口吗？",
                parent=self.root,
            ):
                return
        if self.recorder.moves and self.saved_move_count < len(self.recorder.moves):
            answer = messagebox.askyesnocancel(
                "退出",
                "当前棋谱尚未保存。是否保存后退出？",
                parent=self.root,
            )
            if answer is None:
                return
            if answer:
                self._save_record("玩家退出", notify=False)
        self._closed = True
        self._game_token += 1
        if not self.ai_thinking:
            self._close_ai()
        self.root.destroy()


def main() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as error:
        print(f"Tk desktop UI is unavailable ({error}).")
        print("Starting the dependency-free local browser UI instead.")
        from gomoku_web_ui import main as web_main

        web_main()
        return
    GomokuApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
