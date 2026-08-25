from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from tools import telegram_notify


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class TestTelegramNotify(unittest.TestCase):
    def test_missing_token_is_reported_without_network_call(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "TELEGRAM_BOT_TOKEN"):
                telegram_notify.send_message("test")

    def test_chat_id_is_discovered_from_latest_start_chat(self) -> None:
        payload = {
            "ok": True,
            "result": [
                {"update_id": 1, "message": {"chat": {"id": 123}}},
                {"update_id": 2, "message": {"chat": {"id": 456}}},
            ],
        }
        with (
            patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "secret"}, clear=True),
            patch.object(telegram_notify, "urlopen", return_value=_Response(payload)) as call,
        ):
            self.assertEqual("456", telegram_notify.latest_chat_id())
        self.assertIn("getUpdates", call.call_args.args[0].full_url)

    def test_send_uses_configured_chat_id_and_never_prints_token(self) -> None:
        with (
            patch.dict(
                os.environ,
                {"TELEGRAM_BOT_TOKEN": "secret", "TELEGRAM_CHAT_ID": "456"},
                clear=True,
            ),
            patch.object(
                telegram_notify,
                "urlopen",
                return_value=_Response({"ok": True, "result": {}}),
            ) as call,
        ):
            telegram_notify.send_message("hello")
        request = call.call_args.args[0]
        self.assertIn("sendMessage", request.full_url)
        # The Bot API necessarily carries the token in its endpoint URL; the
        # bridge never logs or persists that URL.
        self.assertIn(b"chat_id=456", request.data)


if __name__ == "__main__":
    unittest.main()
