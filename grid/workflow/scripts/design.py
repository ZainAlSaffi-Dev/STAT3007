"""Place the stage-2 eta*lambda levels at the break the scan estimates.

The design block of ``config/grid.yaml`` gives the sources: Park (1978,
Sec. 4.1) for the points, Berger and Wong (2009, Eq. 2.1) for the rescaling of
the range, from the scan's lower end to the upper end of the boundary, and
Berger and Wong (2009, Sec. 5.4) for placing the second stage at the first
stage's estimate, the break of the broken line fitted to the scan runs.
"""

from contextlib import redirect_stderr
from pathlib import Path

import duckdb
import numpy as np
import yaml

with Path(snakemake.log[0]).open("w", buffering=1) as log, redirect_stderr(log):
    upper = yaml.safe_load(Path(snakemake.input.boundary).read_text(encoding="utf-8"))[
        "upper"
    ]
    with duckdb.connect(snakemake.input.join, read_only=True) as join:
        [(psi,)] = join.sql("SELECT psi FROM join_estimate").fetchall()
    lower = float(snakemake.params["lower"])
    low, high = np.log(lower), np.log(upper)
    centre = (low + high) / 2
    x = float((psi - centre) / (high - centre))
    if not -1 < x < 1:
        msg = (
            f"the break {x} lies outside (-1, 1), so the design has fewer than "
            "four distinct levels"
        )
        raise ValueError(msg)
    points = np.array([-1, x, (1 + x) / 2, 1])
    Path(snakemake.output[0]).write_text(
        yaml.safe_dump(
            {
                "lower": lower,
                "upper": float(upper),
                "join": x,
                "levels": {
                    "x": points.tolist(),
                    "eta_lambda": (
                        lower ** ((1 - points) / 2) * upper ** ((1 + points) / 2)
                    ).tolist(),
                },
            }
        ),
        encoding="utf-8",
    )
