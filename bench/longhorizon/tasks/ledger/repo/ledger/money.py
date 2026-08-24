"""Money as integer cents; see docs/CONVENTIONS.md."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation


def parse(text: str) -> int:
    """A decimal string with at most two decimals -> cents. Floats are refused:
    a float already lost the cents."""
    if not isinstance(text, str):
        raise TypeError(f"amount must be a decimal string, not {type(text).__name__}")
    try:
        value = Decimal(text.strip())
    except InvalidOperation as exc:
        raise ValueError(f"not an amount: {text!r}") from exc
    if value.as_tuple().exponent < -2:
        raise ValueError(f"more than two decimals: {text!r}")
    return int(value.scaleb(2))


def round_half_up(value: Decimal) -> int:
    """Cents from an exact value, half-up on the cent (0.125 -> 13, 0.5 cents up)."""
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def fmt(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}{cents // 100}.{cents % 100:02d}"
