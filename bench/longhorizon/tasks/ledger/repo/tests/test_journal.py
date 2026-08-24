import tempfile
import unittest
from pathlib import Path

from ledger.journal import Entry, Journal, Unbalanced


class JournalTest(unittest.TestCase):
    def test_post_refuses_unbalanced_and_changes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "book.jsonl"
            j = Journal(path)
            with self.assertRaises(Unbalanced):
                j.post([Entry("2026-01-01", "a", 100), Entry("2026-01-01", "b", -99)])
            self.assertFalse(path.exists())
            j.post([Entry("2026-01-01", "a", 100), Entry("2026-01-01", "b", -100)])
            self.assertEqual(Journal(path).balance("a"), 100)
