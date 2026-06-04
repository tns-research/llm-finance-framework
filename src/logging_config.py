"""Centralized logging setup for the framework.

Diagnostic output (progress, warnings, errors, debug dumps) goes through the
logging module so it can be leveled, silenced, or redirected. Formatted report
blocks (banners, tables, per-decision summaries) deliberately stay on print():
they are the framework's human-readable console report on stdout, not log lines,
and prefixing them with timestamps would mangle the layout.

Diagnostics are emitted on stderr (logging default), so the report on stdout
stays clean and pipeable.
"""

import logging
import os
from typing import Optional

_CONFIGURED = False


def setup_logging(level: Optional[str] = None) -> None:
    """Configure root logging once.

    Level resolution: explicit ``level`` arg, else the ``LOG_LEVEL`` env var,
    else ``INFO``. Idempotent: safe to call from multiple entry points.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    resolved = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Thin wrapper kept for call-site clarity."""
    return logging.getLogger(name)
