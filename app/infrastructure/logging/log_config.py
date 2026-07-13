"""Structured logging configuration.

Sets up three rotating file handlers:
  - application.log  — all INFO and above
  - error.log        — WARNING and above only
  - audit.log        — document processing events (custom AUDIT level)

A stream handler (console) is always active during development.
"""

import logging
import logging.handlers
from pathlib import Path

from app.infrastructure.configuration.settings import Settings

# Custom level between INFO (20) and WARNING (30) for audit trail events
AUDIT_LEVEL = 25
logging.addLevelName(AUDIT_LEVEL, "AUDIT")


def audit(self: logging.Logger, message: str, *args, **kwargs) -> None:
    if self.isEnabledFor(AUDIT_LEVEL):
        self._log(AUDIT_LEVEL, message, args, **kwargs)


logging.Logger.audit = audit  # type: ignore[attr-defined]


def configure_logging(settings: Settings) -> None:
    """Initialise logging for the entire application.

    Called once at application startup from app/main.py lifespan.
    """
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, settings.log_level, logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    handlers: list[logging.Handler] = []

    # Console
    console = logging.StreamHandler()
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    handlers.append(console)

    # application.log — all INFO+
    app_handler = logging.handlers.RotatingFileHandler(
        log_dir / "application.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(formatter)
    handlers.append(app_handler)

    # error.log — WARNING+
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "error.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)
    handlers.append(error_handler)

    # audit.log — AUDIT level and above
    audit_handler = logging.handlers.RotatingFileHandler(
        log_dir / "audit.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    audit_handler.setLevel(AUDIT_LEVEL)
    audit_handler.setFormatter(formatter)
    handlers.append(audit_handler)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()
    for handler in handlers:
        root.addHandler(handler)
