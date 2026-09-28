"""Gather every cell's tables and the grid configuration into one DuckDB database.

The view ``exemplar`` holds the first cell, which the report's figures draw.
The views ``target_power``, ``spectral_summary`` and ``event_times`` compute
the report's spectral and timing quantities in SQL, and each carries the
equations it implements in its DuckDB comment.
"""

from contextlib import redirect_stderr
from pathlib import Path

import duckdb
import pandas as pd

with (
    Path(snakemake.log[0]).open("w", buffering=1) as log,
    redirect_stderr(log),
    duckdb.connect(snakemake.output[0]) as con,
):
    for table in snakemake.params.tables:
        con.read_parquet(list(snakemake.input[table])).create(table)
    con.from_df(snakemake.params.cells).cross(
        con.from_df(pd.DataFrame([snakemake.params.fixed]))
    ).create("cells")
    con.sql("CREATE VIEW exemplar AS FROM cells ORDER BY cell LIMIT 1")
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
        WITH tolerated AS (
            SELECT
                *,
                max(abs(eigenvalue)) OVER kernel_matrix
                * count(*) OVER kernel_matrix
                * (nextafter(1::DOUBLE, 2::DOUBLE) - 1) AS tolerance
            FROM target_power
            WINDOW kernel_matrix AS (PARTITION BY cell, seed, step, kernel)
        ),
        normalised AS (
            SELECT
                *,
                if(
                    eigenvalue > tolerance,
                    eigenvalue
                    / sum(eigenvalue) FILTER (eigenvalue > tolerance)
                        OVER (PARTITION BY cell, seed, step, kernel),
                    NULL
                ) AS mu
            FROM tolerated
        )
        SELECT
            * EXCLUDE (eigenvalue, power, rank, cumulative_share, tolerance, mu),
            exp(-sum(mu * ln(mu))) AS erank,
            count(mu) AS positive,
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
        exp(-sum mu_j log mu_j), mu_j = lambda_j / sum_i lambda_i, of Baratin et
        al. (2021), Equation 5, which assumes strictly positive eigenvalues. A
        centred kernel has a zero eigenvalue along the constant vector, so the
        sums run over the positive eigenvalues above the default tolerance of
        numpy.linalg.matrix_rank, S.max() * max(M, N) * eps, with eps the gap
        from 1 to the next double as numpy.finfo defines it; positive counts
        them. trace_ratio is T_k of Equation 6 at k = 2(p - 1), the sum of the
        k largest eigenvalues over the sum of all, as Section 3.2 reads T_40.
        target_share is cumulative_share of target_power at rank 2(p - 1).'
    """)
    con.sql("""
        CREATE VIEW event_times AS
        SELECT *, coalesce(step, budget) AS time, step IS NOT NULL AS observed
        FROM events
    """)
    con.sql("""
        COMMENT ON VIEW event_times IS
        'One row per cell, seed, event and level, as the duration and event
        columns of a lifelines fitter: time is the step of the event, or the
        step budget when the event did not happen, and observed is false for
        such a right-censored run.'
    """)
