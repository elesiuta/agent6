# ledger

A small double-entry bookkeeping CLI. The book is `book.jsonl` in the working
directory (override with `LEDGER_BOOK`).

    python3 -m ledger.cli add 2026-01-05 assets:cash expenses:food 12.50 lunch
    python3 -m ledger.cli balance assets:cash
    python3 -m ledger.cli show

`./verify.sh` is the acceptance gate. Project conventions live in `docs/`.
