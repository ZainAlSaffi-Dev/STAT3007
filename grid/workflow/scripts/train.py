"""Train one cell of the grid and store its two tables as Parquet."""

from contextlib import redirect_stderr
from pathlib import Path

import duckdb

from deeplearning.training import Cell, train

cell = snakemake.params.cell
with (
    Path(snakemake.log[0]).open("w", buffering=1) as log,
    redirect_stderr(log),
):
    frames = train(Cell(**cell, **snakemake.config["fixed"]))
    for table, frame in zip(snakemake.params.tables, frames, strict=True):
        duckdb.from_df(frame.assign(**cell)).to_parquet(snakemake.output[table])
