"""Email notifier over SMTP.

Sends both a plain-text and an HTML part, so the message is readable in a terminal client
and in a browser. Credentials come from the environment only.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

log = logging.getLogger(__name__)


class EmailNotifier:
    name = "email"

    def __init__(
        self,
        *,
        smtp_host: str | None = None,
        smtp_port: int = 587,
        use_tls: bool = True,
        sender: str | None = None,
        recipients: list[str] | None = None,
        username_env: str = "SMTP_USERNAME",
        password_env: str = "SMTP_PASSWORD",
        timeout: float = 30.0,
    ) -> None:
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.use_tls = use_tls
        self.sender = sender
        self.recipients = recipients or []
        self.username_env = username_env
        self.password_env = password_env
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.smtp_host and self.sender and self.recipients)

    def send(self, subject: str, body: str, *, html: str | None = None) -> bool:
        if not self.available():
            log.debug("email not configured, skipping")
            return False

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.sender
        message["To"] = ", ".join(self.recipients)
        message.set_content(body)
        if html:
            message.add_alternative(html, subtype="html")

        username = os.environ.get(self.username_env)
        password = os.environ.get(self.password_env)

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=self.timeout) as server:
                if self.use_tls:
                    server.starttls()
                if username and password:
                    server.login(username, password)
                server.send_message(message)
            return True
        except Exception as exc:
            log.warning("email send failed: %s", exc)
            return False
