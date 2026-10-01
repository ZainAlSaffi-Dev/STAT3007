"""Place the upper end of the eta*lambda range from the scan and redesign.

The scan block of ``config/grid.yaml`` gives the sources. The upper end is the
geometric midpoint of the last grokking and the first other scan point, the
boundary rule of Pracher et al. (2026, app:protocol-mlp-calibration). Park's
(1978, Sec. 4.1) points are placed again on the range up to it, and the lower
end and the join keep the eta*lambda of the first placement, which depends on
the baseline alone.
"""

from contextlib import redirect_stderr
from pathlib import Path

import duckdb
import numpy as np
import yaml

with Path(snakemake.log[0]).open("w", buffering=1) as log, redirect_stderr(log):
    first_design = yaml.safe_load(
        Path(snakemake.input.design).read_text(encoding="utf-8")
    )
    scan = yaml.safe_load(Path(snakemake.input.scan).read_text(encoding="utf-8"))
    phases = (
        duckdb.read_parquet(snakemake.input.phases)
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
    last = float(grokking.index.max()) if not grokking.empty else scan["highest"]
    upper = float(np.sqrt(last * first))
    lower, join_level = first_design["levels"]["eta_lambda"][:2]
    low, high = np.log(snakemake.params.design["range"][0]), np.log(upper)
    centre = (low + high) / 2
    join = float((np.log(join_level) - centre) / (high - centre))
    if not -1 < join < 1:
        msg = (
            f"the join {join} lies outside (-1, 1) of the range up to {upper}, "
            "so the design has fewer than four distinct levels"
        )
        raise ValueError(msg)
    midpoint = (1 + join) / 2
    Path(snakemake.output[0]).write_text(
        yaml.safe_dump(
            {
                "last": last,
                "first": first,
                "upper": upper,
                "join": join,
                "levels": {
                    "x": [-1.0, join, midpoint, 1.0],
                    "eta_lambda": [
                        lower,
                        join_level,
                        float(np.exp(centre + midpoint * (high - centre))),
                        upper,
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
