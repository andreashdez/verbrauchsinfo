from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import polars as pl

from verbrauchsinfo.io import read_files


class ReadFilesTests(unittest.TestCase):
    def test_read_files_reads_all_csvs(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "part_1.csv").write_text("Monat,A\nJan,10\n", encoding="utf-8")
            (tmp_path / "part_2.csv").write_text("Monat,A\nFeb,20\n", encoding="utf-8")

            df = read_files(str(tmp_path))

            self.assertIsInstance(df, pl.DataFrame)
            self.assertEqual(df.height, 2)
            self.assertEqual(sorted(df["Monat"].to_list()), ["Feb", "Jan"])


if __name__ == "__main__":
    unittest.main()
