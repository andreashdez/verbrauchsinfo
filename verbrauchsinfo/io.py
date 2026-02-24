from os import path

import polars as pl


def read_files(resource_path: str) -> pl.DataFrame:
    """Read all CSV files in the provided folder path."""
    return pl.read_csv(path.join(resource_path, "*.csv"))
