"""Fit the report's timing models to the event times of every run.

Each event and level is fitted separately, and every run is kept, censored at
the budget when its event did not happen (Nelson 1990, Ch. 7, Sec. 6.2). The
table ``survival`` holds the Kaplan-Meier estimate of each cell with
Greenwood's exponential interval, and ``medians`` its median with the
median's interval (Kalbfleisch and Prentice 2002, Sec. 1.4.1). ``aft`` holds the log-normal
accelerated failure time model in two arms: the output scales without weight
decay, with indicators of alpha against the baseline cell's, and the runs of
the view ``design_runs``, with the single covariate log(eta*lambda). ``aalen``
holds the cumulative regression functions of Aalen (1989) on the alpha
indicators and eta*lambda over all runs, with intervals from the
bias-corrected and accelerated bootstrap over seeds (Efron and Tibshirani
1993). ``cox`` holds the relative risk model with time-dependent covariates of
Kalbfleisch and Prentice (2002, Eq. 6.14) on the scale, shape and alignment of
the sum-of-logits kernel.
"""

from contextlib import redirect_stderr
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from lifelines import (
    AalenAdditiveFitter,
    CoxTimeVaryingFitter,
    KaplanMeierFitter,
    LogNormalAFTFitter,
)
from lifelines.utils import median_survival_times
from scipy.stats import bootstrap

with (
    Path(snakemake.log[0]).open("w", buffering=1) as log,
    redirect_stderr(log),
    duckdb.connect(snakemake.input[0], read_only=True) as grid,
    duckdb.connect(snakemake.output[0]) as con,
):
    base = grid.sql("SELECT alpha FROM exemplar").fetchone()[0]
    alpha = f"C(alpha, contr.treatment(base={base!r}))"
    runs = grid.sql("""
        SELECT cell, seed, event, level, time, observed, alpha, eta_lambda
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

    arms = {
        "alpha": (runs[runs.eta_lambda == 0], alpha),
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

    def cumulative(group: pd.DataFrame) -> pd.DataFrame:
        """Aalen's cumulative regression functions, one column per covariate.

        Args:
            group: The runs of one event and level.

        Returns:
            The estimates at each event time of the fit.
        """
        fitted: pd.DataFrame = (
            AalenAdditiveFitter()
            .fit(
                group[["time", "observed", "alpha", "eta_lambda"]],
                "time",
                "observed",
                formula=f"{alpha} + eta_lambda",
            )
            .cumulative_hazards_
        )
        return fitted

    aalen = []
    for (event, level), group in runs.groupby(["event", "level"]):
        estimate = cumulative(group)
        by_seed = dict(tuple(group.groupby("seed")))

        def statistic(
            sample: np.ndarray,
            estimate: pd.DataFrame = estimate,
            by_seed: dict[int, pd.DataFrame] = by_seed,
        ) -> np.ndarray:
            """The cumulative functions of a resample of seeds.

            Args:
                sample: The seeds drawn.
                estimate: The fit to every seed, whose event times the
                    resample's step functions are read at.
                by_seed: The runs of each seed.

            Returns:
                The estimates at the event times of ``estimate``, zero before
                the resample's first event.
            """
            resampled = cumulative(pd.concat([by_seed[s] for s in sample]))
            return (
                resampled.reindex(estimate.index, method="ffill").fillna(0).to_numpy()
            )

        interval = bootstrap(
            (np.array(list(by_seed)),), statistic, vectorized=False
        ).confidence_interval
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
            event, level, start, stop, occurred, S_sum, R_sum, A_sum
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
