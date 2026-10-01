"""List the scan points above the highest stage-2 level that groks.

The scan block of ``config/grid.yaml`` gives the sources. A level groks when
its Kaplan-Meier median time to the design level exists within the budget
(Kalbfleisch and Prentice 2002, Sec. 1.4.1), the rule the design applies to the
baseline, and the points come from the lambda grid of Pracher et al. (2026),
continued below its lower end at its own log step when the highest grokking
level lies below the grid, so the scan covers everything above that level.
"""

from contextlib import redirect_stderr
from pathlib import Path

import duckdb
import numpy as np
import yaml
from lifelines import KaplanMeierFitter

design = snakemake.params.design
scan = snakemake.params.scan
with Path(snakemake.log[0]).open("w", buffering=1) as log, redirect_stderr(log):
    events = duckdb.read_parquet(list(snakemake.input)).df()
    grokked = events[(events.event == "grokked") & (events.level == design["level"])]
    medians = grokked.groupby("eta_lambda").apply(
        lambda runs: (
            KaplanMeierFitter().fit(runs.time, runs.observed).median_survival_time_
        )
    )
    grokking = medians[np.isfinite(medians)]
    if grokking.empty:
        msg = "no stage-2 level groks, so the scan has no lower end"
        raise ValueError(msg)
    highest = float(grokking.index.max())
    low, high = scan["lambda"]
    ratio = (high / low) ** (1 / (scan["count"] - 1))
    below = np.floor(np.log(highest / (snakemake.params.eta_0 * low)) / np.log(ratio))
    eta_lambda = snakemake.params.eta_0 * np.concatenate(
        [
            low * ratio ** np.arange(below + 1, 0),
            np.geomspace(low, high, scan["count"]),
        ]
    )
    Path(snakemake.output[0]).write_text(
        yaml.safe_dump(
            {
                "highest": highest,
                "eta_lambda": eta_lambda[eta_lambda > highest].tolist(),
            }
        ),
        encoding="utf-8",
    )
