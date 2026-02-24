import argparse as arg
from datetime import date, datetime
from os import path

import polars as pl

from verbrauchsinfo.io import read_files
from verbrauchsinfo.plotting import plot_graph


GERMAN_MONTH_NAMES = {
    1: "Januar",
    2: "Februar",
    3: "Maerz",
    4: "April",
    5: "Mai",
    6: "Juni",
    7: "Juli",
    8: "August",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Dezember",
}


def _parse_month_value(value: object) -> tuple[int, int] | None:
    if value is None:
        return None

    if isinstance(value, date):
        return (value.year, value.month)

    if isinstance(value, datetime):
        return (value.year, value.month)

    if isinstance(value, str):
        text = value.strip()

        for fmt in (
            "%Y-%m",
            "%Y/%m",
            "%Y.%m",
            "%m/%Y",
            "%m-%Y",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d.%m.%Y",
            "%d/%m/%Y",
        ):
            try:
                parsed = datetime.strptime(text, fmt)
                return (parsed.year, parsed.month)
            except ValueError:
                continue

    return None


def build_month_labels(df: pl.DataFrame, month_col_key: str) -> list[str]:
    parsed_keys: list[tuple[int, int]] = []
    for month in df[month_col_key].to_list():
        parsed = _parse_month_value(month)
        if parsed is None:
            raise ValueError(
                "Month values must be valid date-like values to render German month labels."
            )
        parsed_keys.append(parsed)

    years = {year for year, _ in parsed_keys}
    include_year = len(years) > 1

    labels: list[str] = []
    for year, month in parsed_keys:
        month_name = GERMAN_MONTH_NAMES[month]
        if include_year:
            labels.append(f"{month_name} {year}")
        else:
            labels.append(month_name)
    return labels


def sort_months(df: pl.DataFrame, month_col_key: str, mode: str) -> pl.DataFrame:
    if mode == "preserve":
        return df

    months = df[month_col_key].to_list()
    parsed_keys: list[tuple[int, int]] = []
    for month in months:
        parsed = _parse_month_value(month)
        if parsed is None:
            raise ValueError(
                "Unable to sort months chronologically. Use --month-sort preserve or provide month values in supported formats."
            )
        parsed_keys.append(parsed)

    order = sorted(range(len(months)), key=lambda idx: parsed_keys[idx])
    return df[order]


def validate_dataframe(df: pl.DataFrame, month_col_key: str) -> None:
    if month_col_key not in df.columns:
        raise ValueError(
            f'Missing month column [column="{month_col_key}"]. Use --month-column to select the right field.'
        )

    tenant_cols = [c for c in df.columns if c != month_col_key]
    if not tenant_cols:
        raise ValueError("No tenant columns found in CSV")

    if df[month_col_key].is_null().any():
        raise ValueError("Month column contains empty values")

    duplicate_count = df.group_by(month_col_key).len().filter(pl.col("len") > 1).height
    if duplicate_count > 0:
        raise ValueError("Month column contains duplicate values")

    month_values = df[month_col_key].to_list()
    for month_value in month_values:
        if _parse_month_value(month_value) is None:
            raise ValueError(
                "Month column must contain date-like values (for example: YYYY-MM, YYYY/MM, MM/YYYY, YYYY-MM-DD, DD.MM.YYYY)"
            )

    for tenant in tenant_cols:
        try:
            df[tenant].cast(pl.Float64, strict=True)
        except Exception as exc:
            raise ValueError(
                f'Tenant column must be numeric [column="{tenant}"]'
            ) from exc


def main() -> None:
    parser = arg.ArgumentParser(
        prog="verbrauch",
        description="Calculates gas heating usage",
        epilog="Use with care",
    )
    parser.add_argument(
        "-f",
        "--folder",
        required=True,
        help="Path to the folder containing the CSV files",
    )
    parser.add_argument(
        "--stacked",
        action="store_true",
        help="Plot stacked bars instead of grouped bars",
    )
    parser.add_argument(
        "--output",
        choices=["pdf", "png"],
        default="pdf",
        help="Output format (default: pdf)",
    )
    parser.add_argument(
        "--month-column",
        default="Monat",
        help='Month column name (default: "Monat")',
    )
    parser.add_argument(
        "--month-sort",
        choices=["preserve", "chronological"],
        default="preserve",
        help="Month ordering mode (default: preserve)",
    )
    parser.add_argument(
        "--title",
        default="Verbrauchsinfo Heizung",
        help="Chart title",
    )
    parser.add_argument(
        "--ylabel",
        default="Verbrauch (kWh)",
        help="Y-axis label",
    )
    parser.add_argument(
        "--outdir",
        default="charts",
        help="Output directory for generated chart",
    )
    args = parser.parse_args()

    pl.Config.set_tbl_cols(10)
    pl.Config.set_tbl_rows(20)

    if not path.isdir(args.folder):
        raise ValueError(
            'Provided folder does not exist [folder="{0}"]'.format(args.folder)
        )

    try:
        df = read_files(args.folder)
    except Exception as exc:
        raise ValueError(
            f'Failed to read CSV files from provided folder [folder="{args.folder}"]. Error: {exc}'
        ) from exc

    if df.is_empty():
        raise ValueError("No data found in CSV files")

    validate_dataframe(df, month_col_key=args.month_column)
    df = sort_months(df, month_col_key=args.month_column, mode=args.month_sort)
    month_labels = build_month_labels(df, month_col_key=args.month_column)

    print(df)
    output_path = plot_graph(
        df,
        month_col_key=args.month_column,
        month_labels=month_labels,
        stacked=args.stacked,
        output_format=args.output,
        title=args.title,
        ylabel=args.ylabel,
        outdir=args.outdir,
    )
    print(f"Saved chart to: {output_path}")
