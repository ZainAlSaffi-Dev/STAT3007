#!/usr/bin/env bash
set -o pipefail
Rscript --no-environ -e 'Sys.setenv(R_REMOTES_NO_ERRORS_FROM_WARNINGS="false"); remotes::install_version("censReg", version = "0.5-38", repos = "https://cloud.r-project.org", upgrade = "never")'
