from os import makedirs, path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def plot_graph(
    df: pl.DataFrame,
    month_col_key: str,
    month_labels: list[str],
    stacked: bool,
    output_format: str,
    title: str,
    ylabel: str,
    outdir: str,
) -> str:
    months = month_labels
    tenant_cols = [c for c in df.columns if c != month_col_key]

    n_bars = len(tenant_cols)
    x = np.arange(len(months))
    group_gap = 0.2
    width = (1 - group_gap) / n_bars

    group_centers = x + group_gap / 2 + (n_bars * width) / 2
    group_edges = x[:-1] + n_bars * width + group_gap

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))

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
                ax.bar(
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
        ax.set_ylabel(ylabel, fontdict={"fontsize": 12})
        ax.set_title(title, fontdict={"fontsize": 14})

        plt.tight_layout()

        makedirs(outdir, exist_ok=True)
        filename = f"chart.{output_format}"
        output_path = path.join(outdir, filename)
        plt.savefig(output_path, dpi=300 if output_format == "png" else None)
        plt.close(fig)

    return output_path
