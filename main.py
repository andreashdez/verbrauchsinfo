from os import path, makedirs

import argparse as arg
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def read_files(resource_path: str) -> pl.DataFrame:
    """Read all CSV files in the provided folder path and concatenate them into a DataFrame."""
    return pl.read_csv(path.join(resource_path, "*.csv"))


def plot_graph(
    df: pl.DataFrame,
    month_col_key: str = "Monat",
    stacked: bool = False,
    output_format: str = "pdf",
):
    months = df[month_col_key].to_list()
    tenant_cols = [c for c in df.columns if c != month_col_key]

    if len(tenant_cols) == 0:
        raise ValueError("No tenant columns found in CSV")

    n_bars = len(tenant_cols)

    x = np.arange(len(months))
    group_gap = 0.2
    width = (1 - group_gap) / n_bars

    group_centers = x + group_gap / 2 + (n_bars * width) / 2
    group_edges = x[:-1] + n_bars * width + group_gap

    colors = plt.get_cmap("tab10").colors

    with plt.rc_context(
        {
            "font.size": 9,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    ):
        fig, ax = plt.subplots(figsize=(16, 9))

        if not stacked:
            for i, tenant in enumerate(tenant_cols):
                values = df[tenant].to_numpy()
                offset = i * width + (width / 2) + (group_gap / 2)
                base = colors[i % len(colors)]
                rects = ax.bar(
                    x + offset,
                    values,
                    width,
                    label=tenant,
                    facecolor=mcolors.to_rgba(base, 0.5),
                    edgecolor=base,
                    linewidth=2,
                )
                ax.bar_label(rects, rotation=90, padding=4)
        else:
            bottoms = np.zeros(len(months))
            for i, tenant in enumerate(tenant_cols):
                values = df[tenant].to_numpy()
                base = colors[i % len(colors)]
                rects = ax.bar(
                    x + 0.5,
                    values,
                    bottom=bottoms,
                    width=1 - group_gap,
                    label=tenant,
                    facecolor=mcolors.to_rgba(base, 0.5),
                    edgecolor=base,
                    linewidth=2,
                )
                bottoms += values

        if stacked:
            ax.set_xticks(x + 0.5, months)
        else:
            ax.set_xticks(group_edges, [])
            ax.set_xticks(group_centers, labels=months, minor=True)
            ax.tick_params(axis="x", which="minor", length=0, pad=6)

        ax.set_axisbelow(True)
        ax.yaxis.grid(
            True,
            which="major",
            linestyle="-",
            linewidth=0.8,
            color="0.8",
        )
        for edge in group_edges:
            ax.axvline(edge, color="0.8", linewidth=0.8)

        ax.legend(
            loc="upper left",
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            borderpad=0.6,
            borderaxespad=0.8,
            ncols=1,
        )
        ax.set_ylabel("Verbrauch (kWh)", fontdict={"fontsize": 12})
        ax.set_title("Verbrauchsinfo Heizung", fontdict={"fontsize": 14})

        plt.tight_layout()

        makedirs("charts", exist_ok=True)
        filename = f"chart.{output_format}"
        plt.savefig(
            path.join("charts", filename), dpi=300 if output_format == "png" else None
        )


def main():
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
    args = parser.parse_args()
    pl.Config.set_tbl_cols(10)
    pl.Config.set_tbl_rows(20)
    if not path.isdir(args.folder):
        raise ValueError(
            'Provided folder does not exist [folder="{0}"]'.format(args.folder)
        )

    try:
        df = read_files(args.folder)
    except Exception as e:
        raise ValueError(
            f'Failed to read CSV files from provided folder [folder="{args.folder}"]. Error: {e}'
        )

    if df.is_empty():
        raise ValueError("No data found in CSV files")

    print(df)
    plot_graph(df, stacked=args.stacked, output_format=args.output)


if __name__ == "__main__":
    main()
