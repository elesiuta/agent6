import unittest
from decimal import Decimal

from ledger.money import fmt, parse, round_half_up


class MoneyTest(unittest.TestCase):
    def test_parse_cents(self) -> None:
        self.assertEqual(parse("12.50"), 1250)
        self.assertEqual(parse("7"), 700)
        self.assertEqual(parse("-0.05"), -5)

    def test_parse_refuses_floats_and_extra_decimals(self) -> None:
        with self.assertRaises(TypeError):
            parse(12.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            parse("1.005")

    def test_round_half_up_on_the_cent(self) -> None:
        self.assertEqual(round_half_up(Decimal("12.5")), 13)
        self.assertEqual(round_half_up(Decimal("13.5")), 14)
        self.assertEqual(round_half_up(Decimal("12.4999")), 12)

    def test_fmt(self) -> None:
        self.assertEqual(fmt(1250), "12.50")
        self.assertEqual(fmt(-5), "-0.05")
