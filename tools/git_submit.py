"""Small local GUI for selecting, committing, and pushing Git changes."""

from __future__ import annotations

import subprocess
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_ROOTS = ("records/", "native/bin/", "tools/_local/")
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache"}
EXCLUDED_FILES = {"search-benchmark-results.json"}


@dataclass(frozen=True, slots=True)
class Change:
    index_status: str
    worktree_status: str
    path: str

    @property
    def staged(self) -> bool:
        return self.index_status != " " and self.index_status != "?"

    @property
    def status(self) -> str:
        return f"{self.index_status}{self.worktree_status}".strip() or "?"


def run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def parse_changes(output: str) -> list[Change]:
    changes: list[Change] = []
    for line in output.splitlines():
        if len(line) < 4:
            continue
        changes.append(Change(line[0], line[1], line[3:]))
    return changes


def exclusion_reason(path: str) -> str | None:
    normalized = path.replace("\\", "/")
    if normalized in EXCLUDED_FILES:
        return "运行生成的结果文件"
    if normalized.startswith(EXCLUDED_ROOTS):
        return "记录、编译产物或本机临时目录"
    if any(part in EXCLUDED_PARTS for part in normalized.split("/")):
        return "Git 或 Python 缓存目录"
    if normalized.endswith((".pyc", ".tmp")):
        return "缓存或临时文件"
    return None


class GitSubmitApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Gomoku AI Git 提交")
        self.geometry("900x650")
        self.minsize(700, 500)
        self._changes: list[Change] = []
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="正在读取 Git 状态…")
        ttk.Label(frame, textvariable=self.status_var).pack(anchor=tk.W)
        ttk.Label(
            frame,
            text="选中要提交的核心文件；records、缓存和临时结果会自动排除。",
        ).pack(anchor=tk.W, pady=(2, 8))

        columns = ("status", "path")
        self.tree = ttk.Treeview(
            frame,
            columns=columns,
            show="headings",
            selectmode="extended",
            height=14,
        )
        self.tree.heading("status", text="状态")
        self.tree.heading("path", text="文件")
        self.tree.column("status", width=70, anchor=tk.CENTER, stretch=False)
        self.tree.column("path", width=700, anchor=tk.W)
        self.tree.pack(fill=tk.BOTH, expand=True)

        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X, pady=8)
        self.refresh_button = ttk.Button(buttons, text="刷新状态", command=self.refresh)
        self.refresh_button.pack(side=tk.LEFT)
        ttk.Button(buttons, text="全部选择", command=self.select_all).pack(
            side=tk.LEFT, padx=6
        )
        self.submit_button = ttk.Button(
            buttons,
            text="提交并推送",
            command=self.submit,
        )
        self.submit_button.pack(side=tk.RIGHT)
        self.push_button = ttk.Button(
            buttons,
            text="仅重试推送",
            command=self.retry_push,
        )
        self.push_button.pack(side=tk.RIGHT, padx=6)

        ttk.Label(frame, text="提交说明：").pack(anchor=tk.W)
        self.message = ttk.Entry(frame)
        self.message.pack(fill=tk.X, pady=(2, 8))
        self.message.focus_set()

        ttk.Label(frame, text="执行反馈：").pack(anchor=tk.W)
        self.output = tk.Text(frame, height=11, wrap=tk.WORD, state=tk.DISABLED)
        self.output.pack(fill=tk.BOTH, expand=True)

    def _append(self, text: str) -> None:
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, f"{text}\n")
        self.output.see(tk.END)
        self.output.configure(state=tk.DISABLED)

    def _set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        self.refresh_button.configure(state=state)
        self.submit_button.configure(state=state)
        self.push_button.configure(state=state)

    def refresh(self) -> None:
        result = run_git("-c", "core.quotepath=false", "status", "--short")
        if result.returncode:
            self.status_var.set("无法读取 Git 状态")
            self._append(result.stderr.strip() or result.stdout.strip())
            return
        self._changes = parse_changes(result.stdout)
        for item in self.tree.get_children():
            self.tree.delete(item)
        visible = 0
        excluded = 0
        for index, change in enumerate(self._changes):
            reason = exclusion_reason(change.path)
            if reason:
                excluded += 1
                continue
            item = str(index)
            self.tree.insert("", tk.END, iid=item, values=(change.status, change.path))
            self.tree.selection_add(item)
            visible += 1
        branch = run_git("branch", "--show-current").stdout.strip() or "detached HEAD"
        self.status_var.set(
            f"分支：{branch}｜可选文件：{visible}｜自动排除：{excluded}"
        )
        if excluded:
            self._append(f"已自动排除 {excluded} 个记录、缓存或临时文件。")

    def select_all(self) -> None:
        self.tree.selection_set(self.tree.get_children())

    def submit(self) -> None:
        selected_ids = self.tree.selection()
        message = self.message.get().strip()
        if not selected_ids:
            messagebox.showwarning("没有选择文件", "请选择至少一个要提交的文件。")
            return
        if not message:
            messagebox.showwarning("缺少提交说明", "请填写提交说明。")
            return
        selected = {self._changes[int(item)].path for item in selected_ids}
        unrelated_staged = [
            change.path
            for change in self._changes
            if change.staged and change.path not in selected
        ]
        if unrelated_staged:
            messagebox.showerror(
                "检测到其他已暂存文件",
                "为避免误提交，工具不会提交未选中的暂存文件：\n"
                + "\n".join(unrelated_staged),
            )
            return
        if not messagebox.askyesno(
            "确认提交",
            f"将提交 {len(selected)} 个文件并推送到当前分支。\n\n说明：{message}",
        ):
            return
        self._set_busy(True)
        threading.Thread(
            target=self._submit_worker,
            args=(sorted(selected), message),
            daemon=True,
        ).start()

    def _submit_worker(self, paths: list[str], message: str) -> None:
        steps = [
            ("暂存所选文件", ("add", "--", *paths)),
            ("创建提交", ("commit", "-m", message)),
        ]
        branch = run_git("branch", "--show-current").stdout.strip()
        if branch:
            steps.append(("推送到 GitHub", ("push", "origin", branch)))
        for title, args in steps:
            result = run_git(*args)
            text = result.stdout.strip() or result.stderr.strip() or "完成。"
            self.after(0, self._append, f"{title}：\n{text}")
            if result.returncode:
                self.after(0, self._submit_finished, False)
                return
        self.after(0, self._submit_finished, True)

    def retry_push(self) -> None:
        branch = run_git("branch", "--show-current").stdout.strip()
        if not branch:
            messagebox.showerror("无法推送", "当前不在可推送的本地分支上。")
            return
        if not messagebox.askyesno(
            "确认推送",
            f"重试把当前分支 {branch} 的已有提交推送到 GitHub？",
        ):
            return
        self._set_busy(True)
        threading.Thread(
            target=self._push_worker,
            args=(branch,),
            daemon=True,
        ).start()

    def _push_worker(self, branch: str) -> None:
        result = run_git("push", "origin", branch)
        text = result.stdout.strip() or result.stderr.strip() or "完成。"
        self.after(0, self._append, f"推送到 GitHub：\n{text}")
        self.after(0, self._push_finished, result.returncode == 0)

    def _push_finished(self, success: bool) -> None:
        self._set_busy(False)
        if success:
            self._append("已有提交推送成功。")
            messagebox.showinfo("完成", "已有提交推送成功。")
        else:
            self._append("推送未完成；请查看上方 Git 返回信息。")
            messagebox.showerror("未完成", "推送失败，详情见执行反馈。")
        self.refresh()

    def _submit_finished(self, success: bool) -> None:
        self._set_busy(False)
        if success:
            self._append("提交并推送成功。")
            messagebox.showinfo("完成", "提交并推送成功。")
        else:
            self._append("操作未完成；请查看上方 Git 返回信息。")
            messagebox.showerror("未完成", "Git 操作失败，详情见执行反馈。")
        self.refresh()


def main() -> None:
    GitSubmitApp().mainloop()


if __name__ == "__main__":
    main()
