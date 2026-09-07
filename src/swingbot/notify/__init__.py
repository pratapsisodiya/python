"""Notification channels for the weekly signal."""

from .base import Notifier, NullNotifier, format_signal_message
from .email_smtp import EmailNotifier
from .telegram import TelegramNotifier

__all__ = [
    "EmailNotifier",
    "Notifier",
    "NullNotifier",
    "TelegramNotifier",
    "build_notifiers",
    "format_signal_message",
]


def build_notifiers(cfg) -> list:
    """Every enabled and usable notifier.

    A configured-but-unusable channel logs a warning rather than raising: a missing bot
    token should not abort a signal run that has already computed the book.
    """
    import logging

    log = logging.getLogger(__name__)
    out = []

    if cfg.notify.telegram.enabled:
        notifier = TelegramNotifier(
            token_env=cfg.notify.telegram.bot_token_env,
            chat_id_env=cfg.notify.telegram.chat_id_env,
        )
        if notifier.available():
            out.append(notifier)
        else:
            log.warning(
                "telegram enabled but %s / %s are not set in the environment",
                cfg.notify.telegram.bot_token_env, cfg.notify.telegram.chat_id_env,
            )

    if cfg.notify.email.enabled:
        notifier = EmailNotifier(
            smtp_host=cfg.notify.email.smtp_host,
            smtp_port=cfg.notify.email.smtp_port,
            use_tls=cfg.notify.email.use_tls,
            sender=cfg.notify.email.sender,
            recipients=cfg.notify.email.recipients,
            username_env=cfg.notify.email.username_env,
            password_env=cfg.notify.email.password_env,
        )
        if notifier.available():
            out.append(notifier)
        else:
            log.warning("email enabled but smtp_host, sender or recipients are unset")

    return out
