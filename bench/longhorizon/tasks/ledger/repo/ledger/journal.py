"""The book: balanced transactions of entries, appended to a JSONL file."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


class Unbalanced(ValueError):
    """A transaction whose entries do not sum to zero."""


@dataclass(frozen=True)
class Entry:
    date: str
    account: str
    cents: int
    memo: str = ""


def check_balanced(entries: list[Entry]) -> None:
    total = sum(e.cents for e in entries)
    if total != 0 or not entries:
        raise Unbalanced(f"entries sum to {total} cents, not 0")


def book_path() -> Path:
    return Path(os.environ.get("LEDGER_BOOK", "book.jsonl"))


class Journal:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or book_path()
        self.entries: list[Entry] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.entries.append(Entry(**json.loads(line)))

    def post(self, entries: list[Entry]) -> None:
        """Append one balanced transaction; an unbalanced one changes nothing."""
        check_balanced(entries)
        with self.path.open("a", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(asdict(e)) + "\n")
        self.entries.extend(entries)

    def balance(self, account: str) -> int:
        return sum(e.cents for e in self.entries if e.account == account)

    def accounts(self) -> list[str]:
        return sorted({e.account for e in self.entries})
