# Broken-line fits of the log-normal accelerated failure time model.
#
# Muggeo (2003) fits mu(x) = a0 + a1 x + b (x - psi)_+ in any regression model
# with a linear predictor, and segmented.default runs his iteration on a fit
# through its coef, update and logLik methods (segmented manual). The fit is
# the censored normal regression of censReg (censReg manual) on log time, right
# censored at the step budget, which is the log-linear model of Kalbfleisch and
# Prentice (2002) with a normal error, the log-normal model. segmented.default
# reads the coefficients of a censReg fit from its estimate component (segmented
# NEWS, 1.2-0). censReg refuses rows with no censored observation, and without
# its censored terms its log-likelihood (censReg vignette, Eq. 5) is that of the
# normal linear model, which lm fits and segmented.lm takes (segmented manual).
# Each event and level is fitted twice: with a free left slope,
# and with the left slope fixed at zero by leaving x out of the starting fit,
# Muggeo's model 2. The rows are the database view design_runs, with x =
# log(eta*lambda). psi gets the delta method interval of confint.segmented, b
# the Wald statistic of the censReg summary, and each fit its AIC, the
# criterion Muggeo compares the two by. a0 and a1 are kept with psi and b, the
# coefficients that draw the fitted broken line, as in Muggeo (2008, Fig. 1).

library(censReg)
library(DBI)
library(duckdb)
library(segmented)

log <- file(snakemake@log[[1]], open = "wt")
sink(log)
sink(log, type = "message")

grid <- dbConnect(duckdb(snakemake@input[[1]], read_only = TRUE))
runs <- dbGetQuery(grid, "
  SELECT event, level, ln(time) AS y, observed, ln(steps) AS budget,
    ln(eta_lambda) AS x
  FROM design_runs
")
dbDisconnect(grid)

starts <- list(free = y ~ x, flat = y ~ 1)
fits <- list()
for (d in split(runs, runs[c("event", "level")], drop = TRUE)) {
  budget <- unique(d$budget)
  stopifnot(length(budget) == 1, all(d$observed == (d$y < budget)))
  for (model in names(starts)) {
    start <- if (all(d$observed)) {
      lm(starts[[model]], data = d)
    } else {
      censReg(starts[[model]], left = -Inf, right = budget, data = d)
    }
    fit <- segmented(start, seg.Z = ~x)
    psi <- confint.segmented(fit, "x", .coef = coef(fit), .vcov = vcov(fit))
    wald <- coef(summary(fit))["U1.x", ]
    fits[[length(fits) + 1]] <- data.frame(
      event = d$event[1],
      level = d$level[1],
      model = model,
      a0 = coef(fit)[["(Intercept)"]],
      a1 = if (model == "free") coef(fit)[["x"]] else 0,
      psi = psi[1, 1],
      psi_low = psi[1, 2],
      psi_high = psi[1, 3],
      b = wald[[1]],
      b_se = wald[[2]],
      b_z = wald[[3]],
      b_p = wald[[4]],
      aic = AIC(fit),
      x_min = min(d$x),
      x_max = max(d$x)
    )
  }
}

out <- dbConnect(duckdb(snakemake@output[[1]]))
dbWriteTable(out, "broken_line", do.call(rbind, fits))
dbDisconnect(out)
