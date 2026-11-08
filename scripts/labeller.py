#!/usr/bin/env python3
"""Keypress labelling engine, shared by every coding sheet in this project."""

from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Scheme:
    """One coding scheme."""

    name: str
    sheet: Path
    label_field: str
    display: tuple[str, ...]
    keys: dict[str, str]
    hints: dict[str, str]
    doc: str
    regenerate: str
    next_step: str
    forbidden: tuple[tuple[str, ...], ...] = field(default_factory=tuple)


def assert_blind(scheme: Scheme, *sources: Path) -> None:
    """Refuse to run if any source file mentions what the scheme says must stay hidden."""
    for path in sources:
        body = path.read_text(encoding="utf-8").split('"""', 2)[-1]
        for fragments in scheme.forbidden:
            needle = "".join(fragments)
            if needle in body:
                raise SystemExit(
                    f"[label] {path.name} refers to '{needle}'. The coding is blind by design: "
                    "the coder must never see model output, or the finding measures the coder."
                )


def read_sheet(path: Path, scheme: Scheme) -> tuple[list[dict], list[str]]:
    """Rows plus the column order, so a rewrite preserves the file's own shape."""
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    required = [scheme.label_field, *scheme.display]
    missing = [f for f in required if f not in fields]
    if missing:
        raise SystemExit(f"[label] {path} is missing column(s): {missing}")
    return rows, fields


def write_sheet(path: Path, rows: list[dict], fields: list[str]) -> None:
    """Atomic."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".sheet_", suffix=".csv")
    os.close(fd)
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fields})
    shutil.move(tmp, path)


def read_key() -> str:
    """One character, no Enter."""
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        return line.strip()[:1].lower() if line else "q"
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
    if ch in ("\x03", "\x04"):
        return "q"
    return ch.lower()


def _width() -> int:
    """Wrap to the terminal, capped for readability."""
    return max(60, min(shutil.get_terminal_size((80, 24)).columns - 4, 100))


def wrap(text: str, width: int | None = None) -> list[str]:
    width = _width() if width is None else width
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


def render(scheme: Scheme, rows: list[dict], index: int) -> None:
    done = sum(1 for r in rows if r.get(scheme.label_field, "").strip())
    if sys.stdout.isatty():
        print("\033[H\033[2J\033[3J", end="")
    print()
    print(f"  [ {done}/{len(rows)} labelled ]        row {index + 1} of {len(rows)}")
    for column in scheme.display:
        value = (rows[index].get(column) or "").strip()
        if not value:
            continue
        print()
        for line in wrap(value):
            print(f"  {line}")
    print()
    rule = "  " + "-" * _width()
    print(rule)
    for key, label in scheme.keys.items():
        hint = scheme.hints.get(label, "")
        print(f"  [{key}] {hint}" if key == label else f"  [{key}] {label:<13} {hint}")
    print()
    print("  [u] undo       [s] skip       [q] save & quit")
    print(rule)
    existing = rows[index].get(scheme.label_field, "").strip()
    if existing:
        print(f"  currently: {existing}")
    print()


def run(scheme: Scheme, argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else scheme.sheet
    if not path.exists():
        raise SystemExit(f"[label] {path} not found. Generate it first:\n  {scheme.regenerate}")
    if not os.access(path, os.W_OK):
        raise SystemExit(f"[label] {path} is not writable by you. Fix its ownership first.")

    rows, fields = read_sheet(path, scheme)
    if not rows:
        raise SystemExit(f"[label] {path} has no rows.")

    index = next(
        (i for i, r in enumerate(rows) if not r.get(scheme.label_field, "").strip()), None
    )
    if index is None:
        print(f"\n  All {len(rows)} rows are already labelled.\n")
        print(f"  Next:  {scheme.next_step}\n")
        return 0

    history: list[int] = []
    while True:
        if index >= len(rows):
            index = next(
                (i for i, r in enumerate(rows) if not r.get(scheme.label_field, "").strip()), -1
            )
            if index == -1:
                break
        render(scheme, rows, index)
        key = read_key()

        if key == "q":
            break
        if key == "u":
            if history:
                index = history.pop()
                rows[index][scheme.label_field] = ""
                write_sheet(path, rows, fields)
            continue
        if key == "s":
            history.append(index)
            index += 1
            continue
        if key in scheme.keys:
            rows[index][scheme.label_field] = scheme.keys[key]
            write_sheet(path, rows, fields)
            history.append(index)
            index += 1
            continue

    done = sum(1 for r in rows if r.get(scheme.label_field, "").strip())
    remaining = len(rows) - done
    print(f"\n  Saved. {done}/{len(rows)} labelled.")
    if remaining:
        print(f"  {remaining} left - run this again to continue where you stopped.\n")
    else:
        print(f"\n  All done. Next:\n\n    {scheme.next_step}\n")
    return 0
