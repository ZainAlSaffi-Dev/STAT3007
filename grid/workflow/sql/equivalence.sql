-- Compares a candidate run's stage tables with a reference run's, for the
-- DuckDB command line client, which opens the database it writes. REFERENCE and
-- CANDIDATE name the two directories, read through getenv, which only that
-- client provides; a table function reads each path through getvariable (SET
-- VARIABLE).
SET VARIABLE events_reference = getenv('REFERENCE') || '/events.parquet';
SET VARIABLE events_candidate = getenv('CANDIDATE') || '/events.parquet';
SET VARIABLE phases_reference = getenv('REFERENCE') || '/phases.parquet';
SET VARIABLE phases_candidate = getenv('CANDIDATE') || '/phases.parquet';
SET VARIABLE metrics_reference = getenv('REFERENCE') || '/metrics.parquet';
SET VARIABLE metrics_candidate = getenv('CANDIDATE') || '/metrics.parquet';
SET VARIABLE spectra_reference = getenv('REFERENCE') || '/spectra.parquet';
SET VARIABLE spectra_candidate = getenv('CANDIDATE') || '/spectra.parquet';

-- Events and phases hold steps and labels, so mathematically equivalent runs
-- agree on them row for row, and error() stops the client when they do not.
SELECT error('events differ between the runs')
WHERE (
    SELECT count(*) FROM (
        (FROM read_parquet(getvariable('events_reference'))
        EXCEPT ALL FROM read_parquet(getvariable('events_candidate')))
        UNION ALL
        (FROM read_parquet(getvariable('events_candidate'))
        EXCEPT ALL FROM read_parquet(getvariable('events_reference')))
    )
) > 0;

SELECT error('phases differ between the runs')
WHERE (
    SELECT count(*) FROM (
        (FROM read_parquet(getvariable('phases_reference'))
        EXCEPT ALL FROM read_parquet(getvariable('phases_candidate')))
        UNION ALL
        (FROM read_parquet(getvariable('phases_candidate'))
        EXCEPT ALL FROM read_parquet(getvariable('phases_reference')))
    )
) > 0;

-- The measurements are floating point, so each column's rows and greatest
-- absolute and relative differences are kept as they come out.
CREATE TABLE differences AS
    SELECT
        'metrics' AS tbl,
        metric AS column_name,
        count(*) AS n,
        max(abs(reference - candidate)) AS max_abs,
        max(abs(reference - candidate) / nullif(abs(reference), 0)) AS max_rel
    FROM (
        UNPIVOT (FROM read_parquet(getvariable('metrics_reference')))
        ON COLUMNS('^(train|test)_(loss|acc)$|^weight_norm$|^parameter_movement$|^[SRDA]_')
        INTO NAME metric VALUE reference
    )
    NATURAL JOIN (
        UNPIVOT (FROM read_parquet(getvariable('metrics_candidate')))
        ON COLUMNS('^(train|test)_(loss|acc)$|^weight_norm$|^parameter_movement$|^[SRDA]_')
        INTO NAME metric VALUE candidate
    )
    GROUP BY metric
    UNION ALL
    SELECT
        'spectra',
        quantity,
        count(*),
        max(abs(reference - candidate)),
        max(abs(reference - candidate) / nullif(abs(reference), 0))
    FROM (
        UNPIVOT (FROM read_parquet(getvariable('spectra_reference')))
        ON eigenvalue, power
        INTO NAME quantity VALUE reference
    )
    NATURAL JOIN (
        UNPIVOT (FROM read_parquet(getvariable('spectra_candidate')))
        ON eigenvalue, power
        INTO NAME quantity VALUE candidate
    )
    GROUP BY quantity
    ORDER BY ALL;
