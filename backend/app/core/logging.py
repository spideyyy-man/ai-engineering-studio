"""Centralized logging setup using loguru."""
from __future__ import annotations

import sys

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> | {message}",
        backtrace=False,
        diagnose=False,
    )


__all__ = ["logger", "configure_logging"]
