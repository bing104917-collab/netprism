"""Shared error type for the NetPrism parser core."""

from __future__ import annotations

from pathlib import Path


class AnalyzeError(Exception):
    """Raised when a file / model cannot be analyzed.

    Carries the offending file (and line when known) so the CLI and the folder
    scanner can report actionable errors instead of raw tracebacks.
    """

    def __init__(self, message: str, *, file: str | Path | None = None, line: int | None = None) -> None:
        self.file = str(file) if file is not None else None
        self.line = int(line) if line is not None else None
        self.message = message
        super().__init__(self.describe())

    def describe(self) -> str:
        location = ""
        if self.file:
            location = self.file
            if self.line:
                location = f"{location}:{self.line}"
            location = f"{location}: "
        return f"{location}{self.message}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.describe()


__all__ = ["AnalyzeError"]
