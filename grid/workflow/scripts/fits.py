"""Fit the report's timing models to the event times of every run.

Each event and level is fitted separately, and every run is kept, censored at
the budget when its event did not happen (Nelson 1990, Ch. 7, Sec. 6.2). The
table ``survival`` holds the Kaplan-Meier estimate of each cell with
Greenwood's exponential interval, and ``medians`` its median with the
median's interval (Kalbfleisch and Prentice 2002, Sec. 1.4.1). ``aft`` holds the log-normal
accelerated failure time model in two arms: the output scales without weight
decay, with indicators of alpha against the baseline cell's, fitted to the
values of alpha with at least one event, since the estimate for a level
without failures does not exist (Nelson 1990, Ch. 5, Eq. 2.6 and Sec. 3.3),
and the runs of the view ``design_runs``, with the single covariate
log(eta*lambda). ``aalen``
holds the cumulative regression functions of Aalen (1989) on eta*lambda over
the runs of the baseline cell and of the second stage, eta*lambda = 0
included. Indicators of alpha
would stop the estimator when the first alpha group leaves the risk set, the
final censoring time at which the design loses rank (Aalen 1989, Sec. 4.1);
alpha is compared in the AFT arm. The intervals come from the
bias-corrected and accelerated bootstrap (Efron and Tibshirani 1993) over
seeds, each drawn with all its runs and their censoring, which keeps the runs
that share a split and initial weights together (Davison and Hinkley 1997,
Secs. 3.5.2 and 3.8). Every seed has one run in each of these cells, the
balanced structure for which Davison and Hinkley (1997, Sec. 3.8) justify
resampling whole groups, while the scan trains a single seed. ``cox`` holds the relative risk model with time-dependent covariates of
Kalbfleisch and Prentice (2002, Eq. 6.14) on the scale and shape of the
sum-of-logits kernel and the alignment of the tangent kernel of all outputs
(Baratin et al. 2021).
"""

from contextlib import redirect_stderr
from functools import partial
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from formulaic import model_matrix
from lifelines import CoxTimeVaryingFitter, KaplanMeierFitter, LogNormalAFTFitter
from lifelines.utils import median_survival_times
from scipy.stats import bootstrap

from deeplearning.survival import cumulative_regression

with (
    Path(snakemake.log[0]).open("w", buffering=1) as log,
    redirect_stderr(log),
    duckdb.connect(snakemake.input[0], read_only=True) as grid,
    duckdb.connect(snakemake.output[0]) as con,
):
    [(base,)] = grid.sql("SELECT alpha FROM exemplar").fetchall()
    alpha = f"C(alpha, contr.treatment(base={base!r}))"
    runs = grid.sql("""
        SELECT cell, seed, event, level, time, observed, alpha, eta_lambda, stage
        FROM events NATURAL JOIN cells
    """).df()

    survival = []
    medians = []
    for (cell, event, level), group in runs.groupby(["cell", "event", "level"]):
        km = KaplanMeierFitter().fit(
            group.time, group.observed, label="survival", ci_labels=["lower", "upper"]
        )
        survival.append(
            km.survival_function_.join(km.confidence_interval_)
            .rename_axis("time")
            .reset_index()
            .assign(cell=cell, event=event, level=level)
        )
        lower, upper = median_survival_times(km.confidence_interval_).iloc[0]
        medians.append(
            {
                "cell": cell,
                "event": event,
                "level": level,
                "median": km.median_survival_time_,
                "lower": lower,
                "upper": upper,
            }
        )
    con.from_df(pd.concat(survival, ignore_index=True)).create("survival")
    con.from_df(pd.DataFrame(medians)).create("medians")

    without_decay = runs[runs.eta_lambda == 0]
    arms = {
        "alpha": (
            without_decay[
                without_decay.groupby(["event", "level", "alpha"])[
                    "observed"
                ].transform("any")
            ],
            alpha,
        ),
        "decay": (grid.sql("FROM design_runs").df(), "log(eta_lambda)"),
    }
    aft = [
        LogNormalAFTFitter()
        .fit(
            group[["time", "observed", "alpha", "eta_lambda"]],
            "time",
            "observed",
            formula=formula,
        )
        .summary.reset_index()
        .assign(arm=arm, event=event, level=level)
        for arm, (rows, formula) in arms.items()
        for (event, level), group in rows.groupby(["event", "level"])
    ]
    con.from_df(pd.concat(aft, ignore_index=True)).create("aft")

    aalen = []
    baseline = runs[(runs.alpha == base) & (runs.stage != "scan")]
    for (event, level), group in baseline.groupby(["event", "level"]):
        design = model_matrix("eta_lambda", group)
        seeds, cluster = np.unique(group.seed, return_inverse=True)
        fit = partial(
            cumulative_regression,
            torch.tensor(design.to_numpy(dtype=np.float64)),
            torch.tensor(group.time.to_numpy()),
            torch.tensor(group.observed.to_numpy()),
            torch.tensor(cluster),
        )

        def statistic(
            sample: np.ndarray,
            axis: int,
            fit: partial[torch.Tensor] = fit,
            seeds: np.ndarray = seeds,
        ) -> np.ndarray:
            """The cumulative functions of resamples of seeds.

            Args:
                sample: The seeds drawn, one resample along ``axis``.
                axis: The axis of ``sample`` holding each resample's draws.
                fit: Aalen's estimator on this event and level's runs.
                seeds: The distinct seeds.

            Returns:
                The estimates at the event times of all runs, with the
                resamples along the last axis.
            """
            counts = (np.moveaxis(sample, axis, -1)[..., None] == seeds).sum(-2)
            return (
                fit(torch.from_numpy(counts).to(torch.float64))
                .movedim((-2, -1), (0, 1))
                .numpy()
            )

        estimate = pd.DataFrame(
            statistic(seeds, axis=-1),
            index=np.unique(group.time[group.observed]),
            columns=design.columns,
        )
        interval = bootstrap((seeds,), statistic, vectorized=True).confidence_interval
        # melt lists each covariate's times in turn, the column-major order.
        aalen.append(
            estimate.melt(
                var_name="covariate", value_name="estimate", ignore_index=False
            )
            .assign(
                low=interval.low.ravel(order="F"),
                high=interval.high.ravel(order="F"),
                event=event,
                level=level,
            )
            .rename_axis("time")
            .reset_index()
        )
    con.from_df(pd.concat(aalen, ignore_index=True)).create("aalen")

    intervals = grid.sql("""
        SELECT
            concat_ws('/', cell, seed) AS run,
            event, level, start, stop, occurred, S_sum, R_sum, A_full
        FROM intervals
        WHERE stage != 'scan'
    """).df()
    cox = [
        CoxTimeVaryingFitter()
        .fit(
            group.drop(columns=["event", "level"]),
            id_col="run",
            event_col="occurred",
            start_col="start",
            stop_col="stop",
        )
        .summary.reset_index()
        .assign(event=event, level=level)
        for (event, level), group in intervals.groupby(["event", "level"])
    ]
    con.from_df(pd.concat(cox, ignore_index=True)).create("cox")
