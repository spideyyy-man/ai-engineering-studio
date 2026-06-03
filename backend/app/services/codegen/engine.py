"""Jinja2 engine over the templates/ folder."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(disabled_extensions=("py", "j2")),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_template(name: str, context: dict[str, Any]) -> str:
    tpl = _env.get_template(name)
    return tpl.render(**context)


def list_templates() -> list[str]:
    if not TEMPLATES_DIR.exists():
        return []
    return sorted(
        str(p.relative_to(TEMPLATES_DIR).as_posix())
        for p in TEMPLATES_DIR.rglob("*.j2")
    )
