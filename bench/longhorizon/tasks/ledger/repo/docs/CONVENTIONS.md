# Conventions

- Money is integer cents everywhere inside the package. `ledger.money.parse`
  turns a decimal string into cents and rejects floats and more than two
  decimals; `ledger.money.fmt` renders cents.
- Rounding is half-up on the cent (`ledger.money.round_half_up`), never
  Python's `round()` (banker's rounding) and never truncation.
- `ledger/_commands.py` is GENERATED from `commands.toml` by
  `tools/gen_cli.py`; `./verify.sh` regenerates it first, so a hand edit is
  overwritten. Register or change a command in `commands.toml`.
- Every mutation of the book goes through `Journal.post`, which refuses an
  unbalanced transaction (`ledger.journal.Unbalanced`). Nothing writes
  `book.jsonl` directly.
