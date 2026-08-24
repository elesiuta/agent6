"""Command handlers: argv in, lines out."""

from __future__ import annotations

from ledger.journal import Entry, Journal
from ledger.money import fmt, parse


def add(argv: list[str]) -> list[str]:
    if len(argv) < 4:
        raise SystemExit("usage: add DATE FROM TO AMOUNT [MEMO]")
    date, src, dst, amount = argv[:4]
    memo = " ".join(argv[4:])
    cents = parse(amount)
    j = Journal()
    j.post([Entry(date, src, -cents, memo), Entry(date, dst, cents, memo)])
    return [f"posted {fmt(cents)} {src} -> {dst}"]


def add_legacy(argv: list[str]) -> list[str]:
    # The pre-memo posting path kept for scripts that pass no memo.
    if len(argv) < 4:
        raise SystemExit("usage: add DATE FROM TO AMOUNT")
    date, src, dst, amount = argv[:4]
    cents = parse(amount)
    j = Journal()
    j.post([Entry(date, src, -cents), Entry(date, dst, cents)])
    return [f"posted {fmt(cents)} {src} -> {dst}"]


def balance(argv: list[str]) -> list[str]:
    if len(argv) != 1:
        raise SystemExit("usage: balance ACCOUNT")
    return [fmt(Journal().balance(argv[0]))]


def show(argv: list[str]) -> list[str]:
    j = Journal()
    return [f"{e.date} {e.account} {fmt(e.cents)} {e.memo}".rstrip() for e in j.entries]
