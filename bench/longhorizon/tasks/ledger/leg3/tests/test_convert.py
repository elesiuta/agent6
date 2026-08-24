import tempfile
import unittest
from pathlib import Path

from tests.test_cli import cli


class ConvertTest(unittest.TestCase):
    def test_convert(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            book = Path(d) / "book.jsonl"
            self.assertEqual(cli(book, "convert", "10.00", "2"), "20.00\n")
            self.assertEqual(cli(book, "convert", "10.00", "1.0875"), "10.88\n")
