"""Place the stage-2 eta*lambda levels from the baseline's grokking time.

The design block of ``config/grid.yaml`` gives the sources: Park (1978,
Sec. 4.1) for the points, Berger and Wong (2009, Eq. 2.1) for the rescaling
and Pracher et al. (2026) for the join. The baseline's time is the
Kaplan-Meier median of Kalbfleisch and Prentice (2002, Sec. 1.4.1), which
holds under right censoring.
"""

from contextlib import redirect_stderr
from pathlib import Path

import duckdb
import numpy as np
import yaml
from lifelines import KaplanMeierFitter

design = snakemake.params.design
with Path(snakemake.log[0]).open("w", buffering=1) as log, redirect_stderr(log):
    events = duckdb.read_parquet(snakemake.input[0]).df()
    grokked = events[
        (events.event == "grokked")
        & (events.level == design["level"])
        & (events.alpha == snakemake.params.alpha)
    ]
    t_0 = KaplanMeierFitter().fit(grokked.time, grokked.observed).median_survival_time_
    low, high = np.log(design["range"])
    centre = (low + high) / 2
    join = (np.log(design["constant"] * design["updates"] / t_0) - centre) / (
        high - centre
    )
    if not -1 < join < 1:
        msg = (
            f"the join {join} lies outside (-1, 1), so the design has fewer than "
            "four distinct levels"
        )
        raise ValueError(msg)
    points = np.array([-1, join, (1 + join) / 2, 1])
    Path(snakemake.output[0]).write_text(
        yaml.safe_dump(
            {
                "t_0": float(t_0),
                "join": float(join),
                "levels": {
                    "x": points.tolist(),
                    "eta_lambda": np.exp(centre + points * (high - centre)).tolist(),
                },
            }
        ),
        encoding="utf-8",
    )
