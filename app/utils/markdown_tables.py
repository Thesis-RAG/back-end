"""Small, conservative Markdown-table normalisation helpers.

Some extracted document chunks and model fallbacks contain a valid-looking
Markdown table whose rows have been concatenated into one line, for example:
``| A | B | |---|---| | x | y |``.  Markdown parsers cannot recognise that
form as a table because table rows must be separated by newlines.
"""
from __future__ import annotations

import re


_ROW_BOUNDARY_RE = re.compile(r"\|\s*\|")
_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")


def _parse_pipe_row(segment: str) -> list[str] | None:
    """Parse one pipe-delimited row fragment into cells."""
    value = segment.strip()
    if "|" not in value:
        return None
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|"):
        value = value[:-1]
    cells = [cell.strip() for cell in value.split("|")]
    return cells if len(cells) >= 2 else None


def _normalise_collapsed_line(line: str) -> str:
    """Expand a single-line table only when its second row is a separator."""
    if "|" not in line or "---" not in line:
        return line

    fragments = _ROW_BOUNDARY_RE.split(line)
    rows = [_parse_pipe_row(fragment) for fragment in fragments]
    if len(rows) < 2 or any(row is None for row in rows):
        return line

    parsed_rows = [row for row in rows if row is not None]
    if not all(_SEPARATOR_CELL_RE.fullmatch(cell) for cell in parsed_rows[1]):
        return line
    if len(parsed_rows[0]) != len(parsed_rows[1]):
        return line
    if any(len(row) != len(parsed_rows[0]) for row in parsed_rows[2:]):
        return line

    return "\n".join(
        "| " + " | ".join(row) + " |"
        for row in parsed_rows
    )


def normalize_markdown_tables(text: str | None) -> str:
    """Repair collapsed Markdown table rows while preserving normal text."""
    if not text:
        return text or ""

    value = text.replace("\r\n", "\n").replace("\r", "\n")
    # A JSON/string transport can leave literal ``\\n`` in stored content.
    if "\n" not in value and "\\n" in value:
        value = value.replace("\\n", "\n")

    return "\n".join(_normalise_collapsed_line(line) for line in value.split("\n"))
