"""Place the upper end of the eta*lambda range from the scan.

The scan block of ``config/grid.yaml`` gives the sources. The upper end is the
geometric midpoint of the last grokking and the first other scan point, the
boundary rule of Pracher et al. (2026, app:protocol-mlp-calibration).
"""

from contextlib import redirect_stderr
from pathlib import Path

import duckdb
import numpy as np
import yaml

with Path(snakemake.log[0]).open("w", buffering=1) as log, redirect_stderr(log):
    phases = (
        duckdb.read_parquet(snakemake.input[0])
        .df()
        .set_index("eta_lambda")
        .phase.sort_index()
    )
    other = phases[phases != "grokking"]
    if other.empty:
        msg = "every scan point groks, so the upper end lies above the grid"
        raise ValueError(msg)
    first = float(other.index.min())
    grokking = phases[(phases == "grokking") & (phases.index < first)]
    if grokking.empty:
        msg = "no scan point below the first other one groks"
        raise ValueError(msg)
    last = float(grokking.index.max())
    Path(snakemake.output[0]).write_text(
        yaml.safe_dump(
            {"last": last, "first": first, "upper": float(np.sqrt(last * first))}
        ),
        encoding="utf-8",
    )
