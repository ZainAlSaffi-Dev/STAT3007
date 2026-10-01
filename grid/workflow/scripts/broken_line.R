# Broken-line fits of the log-normal accelerated failure time model.
#
# Muggeo (2003) fits mu(x) = a0 + a1 x + b (x - psi)_+ in any regression model
# with a linear predictor, and segmented.default runs his iteration on a
# survreg fit through its coef, update and logLik methods (segmented manual).
# Each event and level is fitted twice: with a free left slope, and with the
# left slope fixed at zero by leaving x out of the starting fit, Muggeo's
# model 2. The rows are the database view design_runs, with x =
# log(eta*lambda). psi gets the delta method interval of confint.segmented, b
# the Wald statistic of the survreg summary, and each fit its AIC, the
# criterion Muggeo compares the two by. a0 and a1 are kept with psi and b, the
# coefficients that draw the fitted broken line, as in Muggeo (2008, Fig. 1).

library(DBI)
library(duckdb)
library(segmented)
library(survival)

log <- file(snakemake@log[[1]], open = "wt")
sink(log)
sink(log, type = "message")

grid <- dbConnect(duckdb(snakemake@input[[1]], read_only = TRUE))
runs <- dbGetQuery(grid, "
  SELECT event, level, time, observed, ln(eta_lambda) AS x FROM design_runs
")
dbDisconnect(grid)

starts <- list(free = Surv(time, observed) ~ x, flat = Surv(time, observed) ~ 1)
fits <- list()
for (d in split(runs, runs[c("event", "level")], drop = TRUE)) {
  for (model in names(starts)) {
    fit <- segmented(
      survreg(starts[[model]], data = d, dist = "lognormal"),
      seg.Z = ~x
    )
    psi <- confint.segmented(fit, "x", .coef = coef(fit), .vcov = vcov(fit))
    wald <- summary(fit)$table["U1.x", ]
    fits[[length(fits) + 1]] <- data.frame(
      event = d$event[1],
      level = d$level[1],
      model = model,
      a0 = coef(fit)[["(Intercept)"]],
      a1 = if (model == "free") coef(fit)[["x"]] else 0,
      psi = psi[1, 1],
      psi_low = psi[1, 2],
      psi_high = psi[1, 3],
      b = wald[["Value"]],
      b_se = wald[["Std. Error"]],
      b_z = wald[["z"]],
      b_p = wald[["p"]],
      aic = AIC(fit),
      x_min = min(d$x),
      x_max = max(d$x)
    )
  }
}

out <- dbConnect(duckdb(snakemake@output[[1]]))
dbWriteTable(out, "broken_line", do.call(rbind, fits))
dbDisconnect(out)
