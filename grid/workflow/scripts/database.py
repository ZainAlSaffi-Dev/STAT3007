"""Gather every cell's tables and the grid configuration into one DuckDB database.

The view ``exemplar`` holds the first cell, which the report's figures draw.
The views ``target_power``, ``spectral_summary`` and ``intervals`` compute the
report's spectral and timing quantities in SQL, and each carries the equations
it implements in its DuckDB comment, as the ``events`` table carries the
lifelines reading of its columns. The tables ``design``, ``boundary`` and
``design_levels`` hold the stage-2 design and its sources, and ``phases`` the
phase of every run. The view ``design_runs`` holds the runs that the timing
models in log(eta*lambda) are fitted to.
"""

from contextlib import redirect_stderr
from pathlib import Path

import duckdb
import pandas as pd
import yaml

with (
    Path(snakemake.log[0]).open("w", buffering=1) as log,
    redirect_stderr(log),
    duckdb.connect(snakemake.output[0]) as con,
):
    # Scan runs measure no kernel, so their metrics lack the kernel columns.
    for table in snakemake.params.tables:
        con.read_parquet(list(snakemake.input[table]), union_by_name=True).create(table)
    con.from_df(snakemake.params.cells).cross(
        con.from_df(pd.DataFrame([snakemake.params.fixed]))
    ).create("cells")
    con.sql("CREATE VIEW exemplar AS FROM cells ORDER BY cell LIMIT 1")
    designed = yaml.safe_load(Path(snakemake.input.design).read_text(encoding="utf-8"))
    designed.pop("levels")
    con.from_df(pd.DataFrame([snakemake.params.design | designed])).create("design")
    bounded = yaml.safe_load(Path(snakemake.input.boundary).read_text(encoding="utf-8"))
    con.from_df(pd.DataFrame(bounded.pop("levels"))).create("design_levels")
    con.from_df(pd.DataFrame([bounded])).create("boundary")
    con.sql("""
        COMMENT ON TABLE design IS
        'The first placement of the stage-2 design. level is the test accuracy
        of the grokking time; constant and updates are the inverse-decay
        asymptote of the boundary calibration of Pracher et al. (2026), panel
        (a), and the updates after which it was fitted; range is the
        eta*lambda range rescaled to [-1, 1] by Berger and Wong (2009),
        Equation 2.1; t_0 is the baseline''s Kaplan-Meier median time to level,
        Kalbfleisch and Prentice (2002), Section 1.4.1; join is the rescaled
        ln(constant * updates / t_0).'
    """)
    con.sql("""
        COMMENT ON TABLE boundary IS
        'The upper end of the eta*lambda range, the geometric midpoint upper of
        the last grokking scan point last and the first other scan point first,
        Pracher et al. (2026), app:protocol-mlp-calibration, and the join
        rescaled on the range from the lower end of design up to it.'
    """)
    con.sql("""
        COMMENT ON TABLE design_levels IS
        'The stage-2 eta*lambda levels at the points x of the D-optimal design
        of Park (1978), Section 4.1, for a constant joined to a quadratic: -1,
        the join, the midpoint of the join and 1, and 1, on the range up to the
        upper end of boundary.'
    """)
    con.sql("""
        COMMENT ON TABLE phases IS
        'One row per cell and seed: the phase of the run at the final
        checkpoint, Pracher et al. (2026), app:protocol-mlp: grokking,
        memorization, forgetting or no fitting, against the threshold of
        cells.'
    """)
    con.sql("""
        CREATE VIEW target_power AS
        SELECT
            spectra.* EXCLUDE ("order"),
            cell,
            row_number() OVER mode AS rank,
            sum(power) OVER (mode ROWS UNBOUNDED PRECEDING)
            / sum(power) OVER (PARTITION BY cell, seed, step, kernel)
                AS cumulative_share
        FROM spectra NATURAL JOIN cells
        WINDOW mode AS (
            PARTITION BY cell, seed, step, kernel ORDER BY "order" DESC
        )
    """)
    con.sql("""
        COMMENT ON VIEW target_power IS
        'One row per eigenvector of each centred kernel. rank 1 is the largest
        eigenvalue, the reverse of the ascending order of torch.linalg.eigh that
        the spectra column "order" keeps. cumulative_share is the power of the
        centred target along ranks 1 to rank over its total power, the
        cumulative power distribution C of Canatar, Bordelon and Pehlevan
        (2021), Equation 6.'
    """)
    con.sql("""
        CREATE VIEW spectral_summary AS
        WITH normalised AS (
            SELECT
                *,
                abs(eigenvalue)
                / sum(abs(eigenvalue)) OVER (PARTITION BY cell, seed, step, kernel)
                    AS mu
            FROM target_power
        )
        SELECT
            * EXCLUDE (eigenvalue, power, rank, cumulative_share, mu),
            exp(-sum(mu * ln(mu)) FILTER (mu > 0)) AS erank,
            sum(eigenvalue) FILTER (rank <= 2 * (p - 1)) / sum(eigenvalue)
                AS trace_ratio,
            any_value(cumulative_share) FILTER (rank = 2 * (p - 1))
                AS target_share
        FROM normalised
        GROUP BY ALL
    """)
    con.sql("""
        COMMENT ON VIEW spectral_summary IS
        'One row per cell, seed, step and kernel. erank is the effective rank
        of Roy and Vetterli (2007), Definition 1, which Baratin et al. (2021),
        Equation 5, cites: exp(-sum mu_j log mu_j), mu_j = sigma_j / sum_i
        sigma_i over the singular values sigma of the centred kernel, with 0 log
        0 = 0. For a symmetric matrix these are the absolute eigenvalues, the
        route numpy.linalg.svd takes with hermitian=True. trace_ratio is T_k of
        Baratin et al., Equation 6, at k = 2(p - 1), the sum of the
        k largest eigenvalues over the sum of all, as Section 3.2 reads T_40.
        target_share is cumulative_share of target_power at rank 2(p - 1).'
    """)
    con.sql("""
        COMMENT ON TABLE events IS
        'One row per cell, seed, event and level, with the duration and event
        columns of a lifelines fitter: time is the step of the event, or the
        step budget when the event did not happen, and observed is false for
        such a right-censored run.'
    """)
    con.sql("""
        CREATE VIEW intervals AS
        SELECT
            * EXCLUDE (step, time, observed),
            step AS start,
            least(lead(step) OVER run, time) AS stop,
            observed AND lead(step) OVER run IS NULL AS occurred
        FROM metrics
        NATURAL JOIN cells
        NATURAL JOIN events
        WHERE step < time
        WINDOW run AS (PARTITION BY cell, seed, event, level ORDER BY step)
    """)
    con.sql("""
        COMMENT ON VIEW intervals IS
        'One row per cell, seed, event, level and checkpoint before the event
        time, in the long format of the lifelines CoxTimeVaryingFitter: the
        interval runs from the checkpoint start to the next checkpoint, or to
        the event or censoring time for the last one, and occurred marks the
        interval that ends in the event. Each row carries the metrics measured
        at its start, which makes the covariates left-continuous, as the
        relative risk model with time-dependent covariates of Kalbfleisch and
        Prentice (2002), Equation 6.14, requires.'
    """)
    con.sql("""
        CREATE VIEW design_runs AS
        FROM events NATURAL JOIN cells
        WHERE eta_lambda BETWEEN (SELECT range[1] FROM design)
            AND (SELECT upper FROM boundary)
    """)
    con.sql("""
        COMMENT ON VIEW design_runs IS
        'One row per cell, seed, event and level of the runs with eta*lambda in
        the design range, from the lower end of design to the upper end of
        boundary, stage-2 and scan runs alike, all at the baseline cell''s
        alpha. The timing models in log(eta*lambda) are fitted to them alone,
        since an accelerated test model is valid over a limited range of
        stress, Nelson (1990), Chapter 1.'
    """)
