"""Gather every cell's tables and the grid configuration into one DuckDB database.

The view ``exemplar`` holds the first cell, which the report's figures draw.
"""

from contextlib import redirect_stderr
from pathlib import Path

import duckdb
import pandas as pd

with (
    Path(snakemake.log[0]).open("w", buffering=1) as log,
    redirect_stderr(log),
    duckdb.connect(snakemake.output[0]) as con,
):
    for table in snakemake.params.tables:
        con.read_parquet(list(snakemake.input[table])).create(table)
    con.from_df(snakemake.params.cells).cross(
        con.from_df(pd.DataFrame([snakemake.params.fixed]))
    ).create("cells")
    con.sql("CREATE VIEW exemplar AS FROM cells ORDER BY cell LIMIT 1")
