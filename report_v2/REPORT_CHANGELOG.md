# Changelog for the report in report_v2

This file records the decisions behind `overleaf/main.tex`, what was cut from it
and where each cut thing can be found again. It was kept in red draft notes at
the end of `main.tex` until 4 November 2026, when it moved here so that the
report could be submitted. The reference checks stayed in `main.tex` as a
commented block at the end of the file. The criteria self-assessment that stood
beside this changelog was deleted on the same day and is in the git history of
`main.tex`.

When you cut, move or decide something, add it here with a pointer for
restoring it.

## Decisions taken on 5 October 2026

- All results come from `data/`: 720 runs, 5 seeds, 200,000 steps, α ∈ {0.5, 1,
  2}, six widths and eight weight decays. The golden report used the
  grid-v0.1.0 databases (12 seeds, 100,000 steps) instead.
- The alignment A_t is the centred alignment of the summed-logit kernel on the
  test pairs (`A_sum` in `data/`). The golden report on main uses the tangent
  kernel of all p logits, after Baratin et al. The statistics S_t, R_t and A_t
  now share one kernel, and the prediction notebooks already use this A_t.
- The main text uses one grokking level, 0.95. The levels 0.80 and 0.90 move to
  Appendix E.
- The reproduction of Kumar et al. (2024) keeps a short section with one figure.
  The comparison setting by setting moves to Appendix B.
- Tier 4' was first drafted as the rate-addition model. It was reopened the same
  day as a decision between five options (A to E), because that model rests on
  assumptions of our own.
- Each section gained a red draft contribution block that states, as if the
  preliminary findings hold, what is new and why it matters.
- The word "rotation" is kept for now and will become "shape" later. (Done on 25
  October 2026; see below.)
- In Tier 3 the model given the settings is a reference, since a general network
  does not report its laziness or its effective regularisation.
- The planning checks changed one claim. The alignment does not pass the placebo
  test, so Tier 1 claims a threshold for rotation only.

## Decisions taken on 11 October 2026 (Jasper Chong)

- Tier 4' (Section 4.5, with its options A to E and its figure) is commented out
  in `main.tex`, and items E11, E17 and E18 were not run. To restore it, remove
  the leading comment sign from each line between the markers TIER 4' COMMENTED
  OUT and END OF TIER 4', and restore contribution 5 of the Introduction.
- (12 October 2026, Jasper Chong.) A new Section 4.5, Tier 4, holds a TODO
  placeholder for an extension experiment on MNIST digits. It is separate from
  the commented-out theory section Tier 4'. The MNIST runs would be new training
  outside `data/`.
- The spread in Tier 1 is the standard deviation, not the coefficient of
  variation. To restore the CV, set `SPREAD = 'cv'` in
  `report_v2/code/E03_placebo.py`; its JSON output keeps the CV values.
- Every figure, table and number now comes from the scripts in
  `report_v2/code/` through `numbers.tex`, `figures/` and `tables/`. The red
  planning values are gone except those of E14.
- The Tier 2 intervals resample whole cells and then seeds within cells. This
  settles the open decision on the intervals. Seed-only intervals are in
  `E06.json` for comparison.
- E14 (then Appendix D, dropped on 17 October 2026) was not rerun. It needs the
  grid-v0.1.0 databases and the R scripts of branch `feat/grid`.
- Kumar et al. App. 8.1 and 8.3, Sec. 3 and Fig. 2 (arXiv v3) were read for
  Appendix B. The memorisation steps differ from their figure; the grokking
  steps agree at α = 0.5 and α = 1.
- Claims changed by the full runs were marked in red with "Differs from draft"
  in the draft contribution blocks and the abstract. In short: the rotation
  threshold at a fixed width is not sufficient; the interval of γ includes zero;
  weight decay is one of the settings that set the scale; across widths no
  statistic passes only at level 0.95. These markers were removed on 12 October
  2026 (below).

## Changes made on 12 October 2026

- The red draft contribution blocks were folded into the prose and deleted,
  together with the `draftcontrib` environment, the `\differs` macro and the
  commented block inside Tier 4'. Their text is in the git history of `main.tex`
  before this date. The abstract and the Conclusion were written from the final
  numbers.
- The "why it matters" and "what is new" points that the numbers support now sit
  in the Introduction (a paragraph after the contributions), Related Work (the
  Predictions paragraph and Kim's scope), §4.1 (the gate and the baseline
  split), Tier 1 (the alignment hypothesis and the critical norm), Tier 2 (Kim's
  scope and the width-dependent decay edge) and Tier 3 (scoring, progress
  measure).
- Claims changed. (1) The Tier 2 claim and the Conclusion said that weight decay
  acts through the scale. Weight decay explains 0.26 of the spread of S_t, about
  as much as α and width, so the text now says that at a fixed width weight
  decay shrinks the scale. The story in `report_v2/CLAUDE.md` still says weight
  decay sets the scale. (2) The baseline claim "rotates by a few per cent"
  became "R_t stays below 0.09", the bound on every baseline checkpoint. (3) The
  reconciling claim of the Introduction block was dropped.
- Claims added from the final numbers. Tier 1 now reports that S_t pooled over
  all runs that grok tightens 1.69-fold, with the caveat that it keeps growing.
  Tier 3 now states in the prose that 0.96 to 0.99 of the spread lies between
  cells, so the statistics do not separate the seeds of a cell, and its claim is
  limited to held-out cells.
- The Introduction no longer carries a "search the literature" TODO. No sentence
  claims a first. The search is listed under novelty in the criteria table.
- The criteria table was rewritten against the five criteria and the required
  content of the task sheet.
- The red TODO around the interval sentence of §3.5 was removed, since the
  intervals resample cells (decision of 11 October).
- Page limit. Built with the red markup hidden, the main text ended five lines
  into page 13. Words that repeat other sections were cut: Kim's scope in Tier 2
  (kept in Related Work), a closing sentence of §4.1, a scoring sentence of
  Tier 3 (kept in §3.5), and parts of the Introduction and Conclusion. The
  surface figure (E9, commented out on 17 October) went from 0.66 to 0.6 of the
  line width. Two lines still spill onto page 13. The next cuts would be to
  write the Limitations as prose, or to merge the mechanism figure into the map
  figure.
- The code and the numbers were verified. The tangent kernel recomputed by
  autograd from the saved weights of 8 runs matches S_t, R_t and A_t to a
  relative 3e-10. A minimal retraining of the baseline run reproduces its
  accuracy curve exactly over 6,000 steps. Independent code reproduces 133
  macros of `numbers.tex`; the only gaps are Monte Carlo ones (a permutation p
  and a bootstrap bound, both within their noise). A clean rerun of
  `make_all.py` reproduces `numbers.tex`, every table and every JSON output
  exactly.
- One claim was corrected. The text said that at levels 0.80 and 0.90 the scale
  and the alignment tighten across widths. At 0.90 the scale has p = 0.085, so
  only the alignment does. Tier 1 and Appendix E now say so.

## Changes made on 17 October 2026 (Jasper Chong's notes)

- Tier 3' is renamed Tier 3 everywhere in `report_v2/`: in `main.tex`,
  `report_v2/CLAUDE.md`, `code/README.md` and `code/models.py`. The commented-out
  theory section Tier 4' keeps its name.
- Each experiment section opens with "Finding." in place of "Claim.".
- §4.1 was cut to the result, the figure and the split of D_t² at grokking for
  α = 1. The paragraph of event-step medians was dropped, since the event table
  of Appendix B gives them. The gap in memorisation time and the rest of the
  split moved to Appendix B. A TODO there asked whether the report needs that
  appendix; it was settled on 4 November 2026 (below). The earlier text is in
  the git history of `main.tex` before this date.
- A cell is now defined in §3.2, and the captions of the settings table and the
  map figure use the same wording.
- Figure E5 (the grokking time against weight decay at each width) is commented
  out of Tier 2. To restore it, remove the comment signs below the marker FIGURE
  E5 COMMENTED OUT.
- Tier 3 was simplified. It now opens with the column of Table 1 that it answers
  and runs two tests, held-out cells (Table 5) and an unseen α (Fig. 5). Figure
  E8 was redrawn by `code/E08_ladder.py` with five bars: each statistic alone,
  the trees on all three at t\* and the recurrent network on their history. Its
  overlapping label and its cut-off axis label were fixed. The linear model on
  all three, the model with interactions and kernel ridge are no longer in the
  figure or the text, but `E08.json` keeps their scores. The rerun reproduced
  `E08.json` exactly, and `numbers.tex` did not change. §3.5 and Appendix D were
  trimmed to match.
- Figure E9 (the surface over S_t and R_t) and the product coefficient γ_SR are
  commented out of Tier 3. The claim that the best models let rotation and scale
  interact was dropped from the finding. The comment block below the marker
  FIGURE E9 AND THE PRODUCT COEFFICIENT COMMENTED OUT holds the old text and
  says how to restore it.
- The criteria table was rewritten against the marking rubric.
- At Jasper Chong's request, Tier 3 does not test the weight decay as a
  predictor, and the report now says why. We set ηλ, while the effective
  regularisation of a general network cannot be measured, and the report looks
  for predictors that carry over to real models. The caption of Table 1, the
  Prediction paragraph of §3.5 and the opening of Tier 3 each say so.
  `code/E08_ladder.py` now also fits the trees to log N alone and to ηλ alone.
  Both rank runs worse than the two settings together, and Tier 3, its finding
  and the Conclusion use this as support for the mixed clock. The rerun
  reproduced every earlier value of `E08.json`. An earlier draft of the same day
  put the weight-decay fits (kernel ridge on ηλ alone and with the statistics,
  and the trees on ηλ with log R_t or S_t) in a new Appendix D. That appendix and
  its code were removed as confusing. They are not in the git history, so they
  would need to be rewritten.
- Appendix D, the cross-check from the second pipeline (E14), was dropped. It
  held only a TODO that quoted the golden report. Its text is in the git history
  of `main.tex` before this date.

## Changes made on 20 October 2026 (Samuel Rieger)

- Tier 4 (§4.5) now holds the MNIST extension, built from the sweep of 19
  October 2026 in `mnist/sweep/` (560 runs, trained by Zac Kienzle on the
  H100s). Its figure, table and numbers come from `report_v2/code/E19_mnist.py`,
  which reads `mnist/sweep/` from the repository. This is the one exception to
  the rule that every result comes from `data/`; `report_v2/CLAUDE.md` records
  it.
- New: §3.6 (the second task, in Methods), Appendix G (its settings and the
  check rows of its fit), a contribution in the Introduction, a sentence on Liu,
  Michaud and Tegmark (2023) in Related Work, and that paper in the weight-decay
  row of Table 1.
- Claims changed. The abstract, the end of the Introduction and the Conclusion
  no longer say that rotation matters most or that each clock describes one part
  of the time. They say that weight decay sets the time through the weight norm
  on both tasks, and that width enters only on modular addition, where the
  kernel must change shape before the task can be solved. The second half of
  that is our reading, and the text says so.
- To make room: the Limitations are prose; the paragraph on seed-only intervals
  moved from Tier 2 to Appendix E; the opening paragraph of Tier 3 lost two
  sentences. Every change was marked in blue with a bracketed note; the group
  reviewed the blue text the same day and the markup was then removed.
- The MNIST events keep the report's definitions, except that the grokking level
  is 0.80 and the budget is 500,000 steps, since no MNIST run reaches 0.95. §3.6
  says so.
- Later on 20 October: the MNIST text was reworded to say how the delay arises
  there, following Liu, Michaud and Tegmark (2023), Secs. 2, 4 and 5: the
  enlarged initial function is kept by the uncentred predictor and weight decay
  must shrink it, whereas the centred predictor of §3.1 removes it. §3.6 cites
  the group's pilot of 12 October 2026 (`mnist/exploration/results/mnist_pilot/`,
  one seed), in which the centred predictor on MNIST showed no delay at any α.
  Whether a one-seed pilot may be cited in the main text is still open.
- Also later on 20 October, for the tutor's feedback on the proposal (more
  datasets, stated assumptions, expected outcomes): the Introduction says why a
  second task was added, the Predictions paragraph of §2 states what each clock
  predicts for MNIST, the Tier 4 finding recalls those predictions, and the
  Limitations state the assumption behind the comparison.
- Last on 20 October, for the page limit: the MNIST text was cut to the two
  points that matter, how the delay is induced on MNIST against modular
  addition, and what that does to the Tier 1 and Tier 2 tests. The abstract, the
  contribution, Related Work, §3.6, Tier 4, the Limitations and the Conclusion
  are shorter; the MNIST figure is narrower. Nothing was dropped from the
  record: the check rows, width 100, the no-decay runs, the alignment, the
  placebo ratios and the pilot are in Appendix G. The entry for Liu, Michaud and
  Tegmark (2023) in `ref.bib` has a `shortauthor` field so that it cites as "Liu
  et al."; the other three-author entries still cite in full, which
  `maxcitenames=2` in the preamble would change for all of them.
- Panel (c) of the MNIST figure is now the time and outcome map of Fig. 2 at
  s = 7, in place of rotation against width. Drawing it showed that width 100
  does reach the level in most runs with decay, so the caption, the Limitations
  and Appendix G no longer say that it never groks; they say it is later than
  the wider networks, with little delay to time, and off the law.
- A second cut of about 16 lines from the MNIST additions: the MNIST predictions
  sentence in §2 became the clause "on either task" before Table 1; the
  pure-decay check, the placebo reason, the s = 3 accuracy, the check-row
  sentence and the note on Tier 3 moved from Tier 4 to Appendix G; §3.6, the
  Limitations paragraph and the Conclusion lost a clause each. The two-route
  contrast, the finding and the width-100 caveat are unchanged.
- The Limitations and the Conclusion were re-marked so that only the MNIST
  additions were blue. Limitations paragraph 1 is the group's bullet list as
  prose, word for word, with "modular addition" added; the proposal items are
  back in that paragraph. The Conclusion keeps the original sentences, with "On
  modular addition" before "rotation matters most", the MNIST sentences in place
  of "Each published clock therefore describes one part of it", "on two tasks"
  after "the crossed grid", and the frozen-kernel test added to the next step.
- The abstract was rewritten at about half its length, with the group's
  permission: one sentence per result, no numbers, the MNIST result in two
  sentences, and the commented-out line removed.
- To save space, the settings table moved from §3.1 to a new Appendix F just
  before the MNIST appendix, and the map figure and the ladder figure were
  narrowed from 0.72 to 0.56 of the line width. Nothing in the table or the
  figures changed.

## Changes made on 25 October 2026 (Jasper Chong's decisions)

- R_t is now called "shape", or "shape change" where "shape" alone reads badly,
  in place of "rotation". The symbol R_t stays. Jasper Chong took this decision
  on 25 October 2026 once the group agreed, and it replaces the rule of 5
  October that the word "rotation" stays for now. The abstract, every section,
  the captions, Table 1 ("Shape clock"), Table 3, the title of §3.3 and the
  commented-out blocks were changed, and the red TODO about the rename at the
  top of `main.tex` was removed. The sentence that R_t is the kernel distance of
  Fort et al. (2020) stays. The quotes of earlier text in this changelog and in
  the comment above the E9 figure keep the word "rotation".
- `code/common.py` now labels R_t as "shape R_t". Figures E1, E2, E7 and E9 were
  redrawn by rerunning their scripts on the extract in `code/data/`, and copied
  to `figures/` as `make_all.py` does. Their JSON outputs did not change. The
  panel titles of E2 now say "SD ratio" in place of "placebo ratio". E9 is still
  commented out.
- The "tightening ratio" of Tier 1 is renamed the "SD ratio" in §3.5, in the
  caption and the column head of Table 2, and in the caption of the thresholds
  figure. It is the same quantity: the median placebo SD over 2,000 permutations
  divided by the SD at the true grokking steps. No number changed. The macros of
  `numbers.tex` and the field `tightening` in `E03.json` keep their names.
- The Prediction paragraph of §3.5 now says why Tier 3 uses log R_t. Zain
  Al-Saffi wrote to the group that R_t and log R_t were both tried, with other
  non-linear forms, and that log R_t predicted best of the forms of R_t. The
  paragraph also says that R_t at t\* spans more than a decade across widths:
  over the runs with ηλ ≤ 1e-3, its median falls from 0.031 at N = 50 to 0.00037
  at N = 1600. A red TODO asked Zain to confirm which forms were compared and
  whether log S_t was tried; it was settled on 4 November 2026 (below).
- The Introduction now ends with the main finding, the mixed clock, and the
  Conclusion was rewritten around it. On modular addition shape is the most
  informative statistic, but it is not enough on its own and it is not a fixed
  threshold. The time depends on both width and weight decay, so neither the
  shape clock nor the weight-decay clock holds alone. On MNIST weight decay
  alone sets the time. The closing claim that weight decay sets the time through
  the weight norm on both tasks is kept. The reading that width enters only
  where the task forces the kernel to change shape is now labelled as ours in
  the abstract, the Introduction, Tier 4 and the Conclusion.
- To make room, the Introduction paragraph that listed the cited works a second
  time was cut, since its third paragraph lists them. What it said the report
  adds is in the contributions and the Conclusion. Its text is in the git
  history of `main.tex` before this date.
- The TODO after the accelerated failure time equation, which asked for more
  explanation, was answered with two sentences on how to read the model. The
  TODO itself was left in the file by mistake and was removed on 4 November
  2026.
- §3.6 was reworded in shorter sentences with the same content, and a broken
  sentence of Tier 4 ("the kernel shrinks too where at grokking") was mended.
- Page check. Built with the red markup hidden, the Conclusion ends on page 12,
  with the Contribution heading after it. The appendices run from a third of the
  way down page 13 to most of the way down page 18, so they take about 5.5
  pages. With the red markup in, the Conclusion is on page 13 and the appendices
  run from page 14 to page 19. Each of these facts is the same as before the
  changes of this day, so the appendices were already about half a page over
  their limit of 5 pages.

## Changes made on 4 November 2026 (preparing for submission)

Asked for by Jasper Chong, after the seminar was given. Two improvements from
the slides come back into the report, the draft scaffolding goes, and the report
fits its page limits.

- **From the slides.** The Prediction paragraph of §3.5 now says why Tier 3 uses
  log R_t in plain terms: at t\* the values of R_t cluster near zero, 46% of the
  runs that grok are below 0.01 and the largest is 0.13, so taking the logarithm
  spreads them out and predicts better. Kernel ridge on all three statistics
  reaches 0.47 with R_t and 0.55 with log R_t. This answers the red TODO to Zain
  of 25 October for R_t. The report does not claim that log S_t was tried,
  because `E10.json` holds no such fit. The slides' per-model comparison stays in
  the talk.
- **From the seminar's questions.** The Introduction gained a paragraph on why
  the grokking time is worth predicting: the tangent kernel can be measured on
  any network trained by gradient descent, while the settings the published
  clocks need are not always known; an estimate of the time lets training stop
  once the network has generalised, which saves compute; and it says when during
  training a capability appears, which an audit has to establish. The closing
  paragraph of Tier 3 states the limit of that use, since the settings reference
  still ranks runs better.
- **Red markup.** The note block after the title, all 16 page budgets and seven
  TODOs are gone. The `\budget`, `\pending`, `\placeholderfigure` and
  `draftnote` macros stay, because the commented-out Tier 4' section uses them.
  Four TODOs are left for the group: the contribution lines, the
  acknowledgement, the AI-use text and the consent choice. One new TODO asks for
  the link to the public code repository once it exists.
- **TODOs settled.** The notation table is not added, since it costs about a
  quarter of a page. The identity of Appendix A was checked against the golden
  report, `docs/report_golden.tex` §A.1 (`\label{sec:identity}`), which gives the
  same derivation including the Cauchy-Schwarz equality case. The TODO after the
  accelerated failure time equation was already answered on 25 October. The E5
  figure stays commented out and its TODO about the seminar is gone, with the
  reason moved into the comment. Appendix B stays, trimmed.
- **Draft notes.** This file holds the changelog, which was section 2 of the
  notes. The reference checks, section 3, are a commented block at the end of
  `main.tex`. The criteria self-assessment, section 1, was deleted and is in the
  git history of `main.tex` before this date.
- **References.** Dyer and Gur-Ari (2020), "Asymptotics of Wide Networks from
  Feynman Diagrams", was checked against arXiv 1909.11304 (v1, 25 September
  2019, the only version) and the ICLR 2020 programme, added to `ref.bib` as
  `dyerAsymptoticsWideNetworks2020`, and cited in §2 where the text previously
  named it without a citation. The Kalbfleisch and Prentice edition was
  confirmed: the 2002 Wiley-Interscience printing is the second edition, ISBN
  978-0-471-36357-6, so `edition = 2` is right. The Khanh or Truong name order
  is still the group's decision.
- **For the marking criteria.** §3.1 says where the code is and that
  `make_all.py` rebuilds every figure, table and number. The Limitations say
  what two small tasks can and cannot show. "Its test accuracy rises
  significantly long afterwards" in the Introduction became "rises long
  afterwards", since the first reads as statistical significance.
- **Page limits.** The additions above and the float packing pushed the
  Conclusion onto page 14, so the following were done to bring it back to page
  12. The Introduction lost the paragraph "The shape matters for the
  lazy-to-rich account of grokking", whose point is made two paragraphs earlier
  by "A distance does not say how the kernel changed". The baseline figure went
  to 0.92 and the thresholds figure to 0.85 of the line width. Appendix B's
  prose was cut from twenty lines to ten, with no fact dropped except the sweep
  at α = 1.5, which Table 5 still records. Appendix G lost the sentence that the
  three widths of the main range memorise within a few thousand steps of each
  other, which nothing else rests on. The settings tables of Appendices B, F and
  G went from \small to \footnotesize, and the MNIST fit table gained
  `\tabcolsep` of 4pt, which removed the last overfull box of the build.
- **Page check.** The Conclusion ends on page 12. The appendices run from the
  lower half of page 13 to the lower part of page 18, about 5.3 pages against
  the limit of 5, which Jasper Chong confirmed on 4 November 2026 comes from the
  marking sheet. Every remaining appendix page is the reproduction detail that
  the correctness and replication criterion is scored on, so rather than cut it
  here, a red TODO at the head of the appendices lists the candidates in order of
  least loss. The whole document is 20 pages. The build has no errors, no
  undefined references or citations and no overfull or underfull boxes.
- **Two TODOs added.** One at the head of the appendices, to read the marking
  sheet again for the appendix page limit and, if it is 5 pages, to cut about
  half a page. The repository's copy of `project.pdf` gives no appendix limit. One on the consent page, to confirm the report
  still matches the course template, the font size, the page size and the lines
  marked "Don't change this". The template TODO sits on the consent page rather
  than page 1 because three red lines before Table 1 push it over a page break
  and cost most of a page of whitespace.

## Changes made on 9 November 2026 (Jasper Chong)

- The code folder can now be submitted as a zip on its own. `E19_mnist.py` used
  to read the MNIST sweep tables from `mnist/sweep/` at the top of the
  repository, so a zip of `code/` alone failed at E19. `extract.py` now copies
  `runs.parquet`, `report_kernel.parquet` and `manifest.json` from there into
  `code/data/` with the prefix `mnist_`, and their hashes are in
  `code/data/SHA256SUMS.json`. E19 reads those copies. No number changed. To go
  back, point `E19_mnist.py` at `REPO / 'mnist' / 'sweep'` again.
- `make_numbers.py` now also writes `code/out/numbers.tex`, and it creates the
  Overleaf folder if that folder is missing. Before this, an unzipped copy of
  `code/` failed at that step because `../overleaf/` did not exist.
- A rebuild of a copy of `code/` alone gave the same JSON numbers, the same
  `numbers.tex`, and the same figures and tables by hash. The MNIST figure was
  the one exception: it differs slightly in its fonts under matplotlib 3.10.8,
  and it was committed with 3.11.2.
- `code/README.md` was rewritten in plainer prose. Its content is the same,
  apart from the new data location and the run time.

## Cut from the golden report

Each item can be restored from the golden report's source,
`docs/report_golden.tex` (added in commit 7da1996), and its figure files in
`docs/figures/`.

- Section 3.4, Spectral measures: effective rank, trace ratios and the target
  power along eigenvectors. Section 4.3: the cumulative target power figure
  (`docs/figures/spectrum.tex`).
- Section 3.5, Event times: the Kaplan-Meier estimate with Greenwood's interval,
  the phase labels, the additive model of Aalen and the relative risk model with
  time-dependent covariates. Section 4.3: the table of that model. Appendix B,
  Model checks: the probability plot and the cumulative regression functions
  (`docs/figures/probability.tex`, `aalen.tex`).
- Section 3.6, Choice of levels: optimal designs, the censored information
  matrix, the broken line and the sequential design. Section 4.2.2: the scan,
  the design table, the broken-line figure, the survival figure and the
  accelerated failure time table of the weight-decay arm
  (`docs/figures/phases.tex`, `broken.tex`, `survival.tex`). A short summary
  stood in Appendix D until it was dropped on 17 October 2026.
- Section 4.2.1: the α sweep with α = 1.5 and 12 seeds
  (`docs/figures/alpha_sweep.pdf`).
- Section 3.2: the first-logit kernel as a second output.
- Related Work: the paragraph on kernel alignment and spectral bias, silent
  alignment, the early and late dichotomy, Omnigrok, and provable grokking in
  ridge regression.
- The pipeline itself: branch `feat/grid`, folder `grid/` (Snakemake workflow,
  DuckDB databases, R scripts).

## Planned in the proposal and not run

The clamped-norm arm with ρ ∈ [1, 2], the arm over three primes and three
training fractions, the cross-entropy arm, the statistic yᵀK⁻¹y, and events
defined by loss thresholds. The proposal is `docs/G2_proposal.tex`.

## Superseded analyses

`repoduced-code/tier1_v2.ipynb` and `tier2_width_decay.ipynb` use the old data,
with one split shared by every seed, the trace kernel and test accuracy 1.0.
Their methods are ported, and their numbers are not used.

## Open decisions for the group

- Keep the title, or change it to give the answer.
- Keep the mechanism figure as its own figure, or merge it into the map figure.
- Which theory direction, if any, Tier 4' takes (options A to E in the
  commented-out Section 4.5 of `main.tex`).
- Confirm the softer Tier 2 headline of 12 October 2026 ("at a fixed width,
  weight decay shrinks the scale"), and update the story in
  `report_v2/CLAUDE.md` to match.
- Whether the group accepts the curvature learning-rate rule of the MNIST runs
  (§3.6) as part of the setup. Every strong MNIST cell uses it. The only
  fixed-rate grokking in the exploration was weak, at s = 3 (20 October 2026).
- Whether a one-seed pilot may be cited in the main text (§3.6, 20 October
  2026).
- Whether the first author of the two Khanh papers is Khanh or Truong. The bib
  follows arXiv, which parses the surname as Khanh.
- The four TODOs left in `main.tex`: the contribution lines, the
  acknowledgement, the AI-use text and the consent choice. The link to the
  public code repository is a fifth, once the repository exists.

## Settled decisions

- Whether to add the notation table (open since 5 October 2026). Settled on 4
  November 2026: no, it costs about a quarter of a page.
- Whether the report needs Appendix B for the reproduction (open since 17
  October 2026). Settled on 4 November 2026: keep it, trimmed. It is the
  evidence for the correctness and replication criterion.
- Whether to use figure E5 in the seminar (open since 17 October 2026). It was
  not used, and the figure stays commented out.
- Whether to keep the reading that width enters only where the task forces the
  kernel to change shape (open since 20 October 2026). Settled by Jasper Chong
  on 25 October 2026: keep it, labelled as the group's own reading, which rests
  on Mohamadi et al. (2024). The abstract, the Introduction, Tier 4 and the
  Conclusion label it so.
- Whether to rename "rotation" to "shape" (a red TODO since 5 October 2026).
  Settled by Jasper Chong on 25 October 2026 once the group agreed: R_t is
  called "shape".
