"""Small, dependency-free Telegram Bot API bridge for local notifications.

The bot token is read only from ``TELEGRAM_BOT_TOKEN``.  Chat IDs can be
provided through ``TELEGRAM_CHAT_ID`` or discovered from the most recent
``/start`` update.  Nothing is written to the repository.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_ROOT = "https://api.telegram.org/bot"


def _token() -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "未设置 TELEGRAM_BOT_TOKEN；请在本机环境变量中配置 BotFather Token。"
        )
    return token


def _call(method: str, **params: object) -> dict[str, Any]:
    token = _token()
    query = urlencode({key: value for key, value in params.items() if value is not None})
    request = Request(
        f"{API_ROOT}{token}/{method}",
        data=query.encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urlopen(request, timeout=15) as response:  # noqa: S310 - fixed API host
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram API {method} 失败：{payload}")
    return payload


def latest_chat_id() -> str:
    configured = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if configured:
        return configured
    updates = _call("getUpdates", limit=20, allowed_updates='["message"]')
    messages = [
        update.get("message", {})
        for update in updates.get("result", [])
        if update.get("message", {}).get("chat", {}).get("id") is not None
    ]
    if not messages:
        raise RuntimeError(
            "没有找到聊天 ID；请先向机器人发送 /start，或设置 TELEGRAM_CHAT_ID。"
        )
    return str(messages[-1]["chat"]["id"])


def send_message(text: str, *, chat_id: str | None = None) -> None:
    target = chat_id or latest_chat_id()
    _call("sendMessage", chat_id=target, text=text[:4096])


def main() -> int:
    parser = argparse.ArgumentParser(description="发送 Gomoku AI Telegram 通知")
    parser.add_argument("--get-chat-id", action="store_true")
    parser.add_argument("--test", action="store_true", help="发送一条连接测试消息")
    parser.add_argument("text", nargs="?", help="要发送的消息")
    args = parser.parse_args()
    if args.get_chat_id:
        print(latest_chat_id())
        return 0
    if args.test:
        send_message("Gomoku AI Telegram 通知已连接。")
        return 0
    if not args.text:
        parser.error("请提供消息文本，或使用 --get-chat-id / --test。")
    send_message(args.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
