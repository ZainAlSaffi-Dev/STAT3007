"""Train the cells of one stage fused and store their tables as Parquet."""

from contextlib import redirect_stderr
from dataclasses import fields
from pathlib import Path

import duckdb

from deeplearning.training import Cell, Setting, train

cells = snakemake.params.cells
settings = cells[[field.name for field in fields(Setting)]]
# The cells of a stage share every column but their settings.
shared = cells.drop(columns=settings.columns).drop_duplicates().squeeze().to_dict()
with (
    Path(snakemake.log[0]).open("w", buffering=1) as log,
    redirect_stderr(log),
):
    frames = train(
        Cell(**shared, settings=settings.to_dict("records"), **snakemake.params.fixed)
    )
    for table in snakemake.params.tables:
        duckdb.from_df(frames[table].assign(**shared)).to_parquet(
            snakemake.output[table]
        )
