"""Telegram notifier.

Credentials come from the environment only, never from a config file, so a shared
repository cannot leak a bot token. Telegram caps messages at 4096 characters, so long
books are split rather than silently truncated — losing the last three positions of a
fifteen-name book without saying so would be worse than sending two messages.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"
_LIMIT = 4000


class TelegramNotifier:
    name = "telegram"

    def __init__(
        self,
        *,
        token_env: str = "TELEGRAM_BOT_TOKEN",
        chat_id_env: str = "TELEGRAM_CHAT_ID",
        timeout: float = 20.0,
    ) -> None:
        self.token_env = token_env
        self.chat_id_env = chat_id_env
        self.timeout = timeout

    def available(self) -> bool:
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get(self.token_env) and os.environ.get(self.chat_id_env))

    def send(self, subject: str, body: str, *, html: str | None = None) -> bool:  # noqa: ARG002
        if not self.available():
            log.debug("telegram not configured, skipping")
            return False
        import httpx

        token = os.environ[self.token_env]
        chat_id = os.environ[self.chat_id_env]
        text = f"*{subject}*\n```\n{body}\n```"

        ok = True
        with httpx.Client(timeout=self.timeout) as client:
            for chunk in _split(text, _LIMIT):
                try:
                    response = client.post(
                        _API.format(token=token),
                        json={
                            "chat_id": chat_id,
                            "text": chunk,
                            "parse_mode": "Markdown",
                            "disable_web_page_preview": True,
                        },
                    )
                    response.raise_for_status()
                except Exception as exc:
                    log.warning("telegram send failed: %s", exc)
                    ok = False
        return ok


def _split(text: str, limit: int) -> list[str]:
    """Split on line boundaries so a table never breaks mid-row."""
    if len(text) <= limit:
        return [text]
    parts, current = [], []
    length = 0
    for line in text.splitlines(keepends=True):
        if length + len(line) > limit and current:
            parts.append("".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line)
    if current:
        parts.append("".join(current))
    return parts
