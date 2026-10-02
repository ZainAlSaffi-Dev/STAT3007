# The break of the first stage of the sequential design (Berger and Wong 2009,
# Sec. 5.4), which the second stage's design is placed at.
#
# The scan runs below the upper end of the range are fitted with the broken
# line of Muggeo (2003), mu(x) = a0 + a1 x + b (x - psi)_+, with x =
# log(eta*lambda), in the censored normal regression of censReg on log time to
# the design level, right censored at the step budget, as broken_line.R fits the
# stage-2 runs. psi gets the delta method interval of confint.segmented.

library(censReg)
library(DBI)
library(duckdb)
library(segmented)

log <- file(snakemake@log[[1]], open = "wt")
sink(log)
sink(log, type = "message")

scan <- dbConnect(duckdb())
runs <- dbGetQuery(
  scan,
  "
  SELECT ln(time) AS y, observed, ln(steps) AS budget, ln(eta_lambda) AS x
  FROM read_parquet(?)
  WHERE event = 'grokked' AND level = ? AND eta_lambda <= ?
  ",
  params = list(
    snakemake@input[["events"]],
    snakemake@params[["level"]],
    snakemake@params[["upper"]]
  )
)
dbDisconnect(scan)

budget <- unique(runs$budget)
stopifnot(length(budget) == 1, all(runs$observed == (runs$y < budget)))
start <- if (all(runs$observed)) {
  lm(y ~ x, data = runs)
} else {
  censReg(y ~ x, left = -Inf, right = budget, data = runs)
}
fit <- segmented(start, seg.Z = ~x)
psi <- confint.segmented(fit, "x", .coef = coef(fit), .vcov = vcov(fit))

out <- dbConnect(duckdb(snakemake@output[[1]]))
dbWriteTable(out, "join_estimate", data.frame(
  psi = psi[1, 1],
  psi_low = psi[1, 2],
  psi_high = psi[1, 3],
  runs = nrow(runs)
))
dbDisconnect(out)
