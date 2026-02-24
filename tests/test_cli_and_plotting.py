from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib
import polars as pl

from verbrauchsinfo.cli import build_month_labels, sort_months, validate_dataframe
from verbrauchsinfo.plotting import plot_graph

matplotlib.use("Agg")


class ValidateDataFrameTests(unittest.TestCase):
    def test_rejects_missing_month_column(self) -> None:
        df = pl.DataFrame({"month": ["Jan"], "TenantA": [10]})

        with self.assertRaisesRegex(ValueError, "Missing month column"):
            validate_dataframe(df, month_col_key="Monat")

    def test_rejects_non_numeric_tenant_column(self) -> None:
        df = pl.DataFrame({"Monat": ["2025-01"], "TenantA": ["x"]})

        with self.assertRaisesRegex(ValueError, "must be numeric"):
            validate_dataframe(df, month_col_key="Monat")

    def test_rejects_duplicate_months(self) -> None:
        df = pl.DataFrame({"Monat": ["2025-01", "2025-01"], "TenantA": [1, 2]})

        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_dataframe(df, month_col_key="Monat")


class SortMonthsTests(unittest.TestCase):
    def test_sorts_date_formats_chronologically(self) -> None:
        df = pl.DataFrame(
            {
                "Monat": ["2025-03", "2025-01", "2025-02"],
                "TenantA": [30, 10, 20],
            }
        )

        sorted_df = sort_months(df, month_col_key="Monat", mode="chronological")

        self.assertEqual(
            sorted_df["Monat"].to_list(), ["2025-01", "2025-02", "2025-03"]
        )


class MonthLabelTests(unittest.TestCase):
    def test_builds_german_labels_without_year_for_single_year(self) -> None:
        df = pl.DataFrame({"Monat": ["2025-01", "2025-03"]})

        labels = build_month_labels(df, month_col_key="Monat")

        self.assertEqual(labels, ["Januar", "Maerz"])

    def test_builds_german_labels_with_year_for_multiple_years(self) -> None:
        df = pl.DataFrame({"Monat": ["2025-12", "2026-01"]})

        labels = build_month_labels(df, month_col_key="Monat")

        self.assertEqual(labels, ["Dezember 2025", "Januar 2026"])


class PlotGraphTests(unittest.TestCase):
    def test_grouped_output_is_written(self) -> None:
        df = pl.DataFrame(
            {"Monat": ["2025-01", "2025-02"], "A": [10, 11], "B": [12, 13]}
        )

        with TemporaryDirectory() as tmp:
            output_path = plot_graph(
                df,
                month_col_key="Monat",
                month_labels=["Januar", "Februar"],
                stacked=False,
                output_format="png",
                title="Title",
                ylabel="kWh",
                outdir=tmp,
            )

            self.assertEqual(output_path, str(Path(tmp) / "chart.png"))
            self.assertTrue(Path(output_path).exists())

    def test_stacked_output_is_written(self) -> None:
        df = pl.DataFrame(
            {"Monat": ["2025-01", "2025-02"], "A": [10, 11], "B": [12, 13]}
        )

        with TemporaryDirectory() as tmp:
            output_path = plot_graph(
                df,
                month_col_key="Monat",
                month_labels=["Januar", "Februar"],
                stacked=True,
                output_format="pdf",
                title="Title",
                ylabel="kWh",
                outdir=tmp,
            )

            self.assertEqual(output_path, str(Path(tmp) / "chart.pdf"))
            self.assertTrue(Path(output_path).exists())


if __name__ == "__main__":
    unittest.main()
