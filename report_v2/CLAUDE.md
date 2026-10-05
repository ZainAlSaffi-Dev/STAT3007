# CLAUDE.md for report_v2

This file is for any assistant that edits the report in this folder or runs the experiments behind it. Read the root `CLAUDE.md` first. Its four rules still apply here: never invent a source, write plain full sentences, cross-check anything taken from a paper, and make every LaTeX change compile. This file adds the rules that are specific to `report_v2`.

`report_v2` is the restructured G2 report. It was set up on 5 October 2026 to give the golden report one line of argument, to use the data in `data/` only, and to cut the measures and models that the story does not need. Where this file and the golden report disagree about the content of this report, this file wins inside `report_v2`. The root `CLAUDE.md` records this precedence. The golden report itself is `docs/report_golden.pdf`, built from `docs/report_golden.tex`. The decisions are listed below, and none of them should be reverted without asking the user.

## What is in this folder

```
report_v2/
  CLAUDE.md            this file
  EXPERIMENTS_TODO.md  the work list: every figure (E1 to E18) and number (N1 to N9)
  overleaf/            upload this folder to Overleaf as it is
    main.tex           the report
    ref.bib            only the entries main.tex cites, each one verified
    figures/           PDFs made by the scripts; empty until they exist
  code/                does not exist yet; one script per item will live here
```

The `overleaf/` folder must stay self-contained. Nothing in it may refer to a file outside it.

## The story the report tells

Every edit should serve this argument. If a change does not help a reader follow it, leave the change out.

1. The research question is what sets the grokking time: the shape of the tangent kernel, its scale, or the weight-decay rate.
2. The literature times grokking by a scale quantity (the weight norm or the logit scale) or by the weight-decay rate. In the kernel-flow equation of Lewkowycz and Gur-Ari, weight decay only rescales the kernel, and only a finite-width term can rotate it. Our hypothesis is that rotation sets the clock.
3. Table 1 of the report sets out what each account predicts. Each test answers one column of it.
   - Tier 1 asks whether grokking happens at a fixed value of a statistic.
   - Tier 2 asks how the grokking time scales with width and with weight decay.
   - Tier 3' asks which early statistic predicts the grokking time.
4. All three tests give the same answer. Rotation matters most but is not enough on its own, and the time is set by rotation and scale together. Width sets the rotation and weight decay sets the scale. This is the "mixed clock". Tier 4' is a stretch theory section. Its direction is not decided: §4.5 of the report lists options A to E, with the costs and risks of each. Do not write the section until the group chooses.

## Decisions that differ from the golden report

These were taken by Jasper Chong on 5 October 2026. They are recorded in the changelog in the red draft notes at the end of `main.tex`.

- All results come from `data/`. That is the sweep of 3 October 2026 with 720 runs, 5 seeds and 200,000 steps. The golden report used the grid-v0.1.0 databases on branch `feat/grid` (12 seeds, 100,000 steps), which appear only in Appendix D as a cross-check.
- The report uses three statistics on one kernel. The kernel is the tangent kernel of the sum of the logits divided by √p, on the 53 test pairs of each run.
  - S is the ratio of Frobenius norms to step 0.
  - R is one minus the uncentred cosine to step 0. It is called "rotation" for now.
  - A is the centred alignment with Y Yᵀ, which is `A_sum` in `data/`.
  - The golden report's A, which uses the tangent kernel of all p logits, is not used.
- The variation D appears only through the identity D² = S² − 2S(1 − R) + 1, to show that it mixes scale and rotation.
- The main text uses grokking level 0.95 only. Levels 0.80 and 0.90 go in Appendix F.
- The following are cut from the main text: the spectral measures, Kaplan-Meier with Greenwood's interval, the Aalen model, the time-varying Cox model, the broken line, optimal design and the phase taxonomy. The changelog says where to find each one if the group wants it back.
- In Tier 3', the model given the settings (width and weight decay) is a reference that knows more than any network would report. It is not a competitor. The reason is that a general network does not report its laziness or its effective regularisation, while its tangent kernel can be measured.
- "Rotation" stays as the word for R until the group agrees to switch to "shape". A red TODO at the top of `main.tex` records this. Do not do the rename unless asked.

## What the data allow the report to claim

These limits come from checks on `data/`. A claim that goes past them is wrong, even if it reads better.

- **Tier 1.** Rotation passes the placebo test only at a fixed width. Read at another run's grokking step, its spread at N = 100 widens 1.9 times, against 1.3 times for scale. Across widths no statistic passes.
- **Alignment.** A barely moves during training (median 0.398 at step 0 and 0.406 at grokking) and fails the placebo test. Do not write that alignment is a threshold or a clock.
- **Tier 2.** The exponents depend on which weight-decay cells enter the fit. Always report the main range (ηλ from 1e-5 to 1e-4) together with the check rows. If the intervals resample seeds only, say that they hold the cells fixed.
- **Tier 3'.** Rotation is the best single statistic, scale adds information that rotation lacks, and alignment adds little.
  - The statistics are almost fully set by the cell, and they carry no signal about which seed of a cell groks first.
  - The settings reference predicts better than any kernel model.
  - Do not claim that the kernel gives an earlier warning than the settings do.
- **Tier 4'.** Whatever option the group chooses, label as the group's own any model or derivation that no source gives. Option A, the rate-addition model 1/T = r0 N^(−q) + c·ηλ, is the clearest case. No source paper derives it.
  - Two parts come from sources. The weight-decay term of the kernel-flow equation only rescales the kernel (Lewkowycz and Gur-Ari, Eq. S5). The term that rotates the kernel, T_t, is of order 1/N at initialisation, which they credit to Dyer and Gur-Ari. Kim's law makes the decay route's rate grow in proportion to ηλ.
  - The rest is ours. That T_t stays of order 1/N during training, that the width route and the decay route act side by side so their rates add, and the form r0 N^(−q) are our assumptions.
  - So the text must present the model as a hypothesis that the data test. It must not cite it to Lewkowycz and Gur-Ari or to Kim, and it must not treat a good fit as proof of the mechanism.

## Editing main.tex

**Build.** Run this from `report_v2/overleaf/`:

```
tectonic main.tex
```

The build must finish with no errors, undefined citations or undefined references. It should also have no overfull boxes. Read the log, then delete `main.log` and `main.blg` so the folder stays clean for Overleaf.

**Overleaf limits.**
- No PythonTeX, no shell escape, no `\input` of a file outside `overleaf/`, and no packages that Overleaf lacks.
- Biblatex runs with the bibtex backend, as the template sets.
- Do not put an `@` inside a comment in `ref.bib`. BibTeX reads it as the start of an entry.

**Template.** The preamble down to `%%% Add additional packages or macros below` is `proposal_template.tex` verbatim. Never change a line marked `% Don't change this`, the font size or the page size. Add packages and macros only below that line.

**Page limits.**
- The main text must end by page 12. References and appendices do not count toward that.
- The appendices may take at most 5 pages.
- Each section shows its budget with `\budget{...}`. On 5 October 2026 the main text ended on page 11 with placeholders alone, and on page 15 once the red draft contribution blocks and the Tier 4' options were added. Those red blocks are temporary, so judge the budget by the text that will stay.
- If an edit pushes past a budget, cut words before you cut content, and tell the user.

**Red markup for the group.** All of it is deleted before submission, and not before.

| Macro | Meaning |
|---|---|
| `\TODO{...}` | A task. |
| `\pending{value}{ID}` | A planning value from a read-only check. Item ID in `EXPERIMENTS_TODO.md` recomputes it. |
| `\budget{...}` | The page budget of a section. |
| `\placeholderfigure[height]{...}` | A red box that stands in for a figure. The text names its E item. |
| `draftnote` environment | A red block. The notes after the consent page use it. |
| `draftcontrib{where}` environment | A red draft of what a section adds and why it matters, written as if the preliminary findings hold. It ends with a note to verify against the full runs. Fold what survives into the prose and delete the block. These blocks carry the narrative and the case for significance and novelty, so update them whenever a finding changes. |
| `\file{...}` | A file path that can break across lines. |

**Replacing a placeholder.** When the figure for item E*k* exists as `overleaf/figures/E<k>_<name>.pdf`, replace its `\placeholderfigure{...}` with `\includegraphics[width=\linewidth]{figures/E<k>_<name>.pdf}`. Then rewrite the caption and remove its `\TODO`.

**Finalising a number.**
- Replace `\pending{value}{ID}` with the recomputed value only after the item's script has run.
- If the value changes enough to alter a claim, change the claim, tell the user, and add a line to the changelog. Never adjust a number to fit the text.
- Once `overleaf/numbers.tex` exists, use its macros in place of typed numbers.

**Keep the draft notes current.**
- When you cut, move or decide something, add it to the changelog with a pointer for restoring it.
- When you add a reference, add a row to the reference-check table.
- When an open decision is settled, move it out of the open list and say who settled it.

**Writing.** Rule 2 of the root `CLAUDE.md` applies to every sentence, caption, table cell and figure label. Before you finish, run these from `report_v2/overleaf/`:

```
grep -nE -- '---|—|–' main.tex
grep -nwiE 'groundbreaking|remarkable|elegant|powerful|crucial|seamless|dramatically|robustly|fundamental|key|deep|simply|of course' main.tex
```

The first grep should print nothing. `--` is allowed only inside page ranges such as `795--828`. The second grep has one known hit, "the proposal's key is", where "key" is a bibliography key. Also avoid "not X but Y" sentences. Use `\cref` and `\Cref` for cross-references. The figures, tables and equations all have labels such as `fig:map`, `tab:scaling` and `eq:identity`.

## References

- `ref.bib` holds only the entries that `main.tex` cites. Every entry was checked on 5 October 2026, and the red reference-check table in `main.tex` records how.
- To add a source, fetch its arXiv abstract page or publisher page. Confirm the title, the authors and their name order, and the year. If you cite a section, theorem or equation number, open the PDF of the version you name and confirm it.
- Copy an entry from `docs/ref.bib`, the golden report's bibliography, when it is there. Otherwise write a new one with a comment saying what it was checked against.
- Then add a row to the reference-check table. If a source cannot be verified, cite it with `\TODO{UNVERIFIED: ...}` and tell the user. Never cite from memory, and never swap in a source that looks close.
- These items are still open:
  - The Khanh papers' first author may have the family name Truong. The bib follows arXiv, which parses it as Khanh. The group decides.
  - Dyer and Gur-Ari (2020) is named in the text only as Lewkowycz and Gur-Ari's source. It is not in `ref.bib` until someone checks it.
  - The edition of Kalbfleisch and Prentice was set to 2 and should be confirmed against the book.
  - Kumar et al. Appendix 8.3 still has to be compared with Table 2 of the report (item E12).
- The kernel-flow equation of Lewkowycz and Gur-Ari is Eq. S5, and T_t is Eq. S6, in both arXiv v1 and v2. A note in the proposal says S8 and S9 for v2. That note is wrong.

## Running experiments

`EXPERIMENTS_TODO.md` is the work list. Pick an item, follow its setup exactly, and keep to the ground rules at its top. In short:

- **Data.**
  - Read from `data/` only. It is about 11 GB and is not in git, so never commit it or any large file from it.
  - Do not use `data/runs_seed42/` or `data/seed42_tables/`, where every seed shared one split.
  - The column `eta_kappa` is ηλ.
  - The statistics are `S_sum`, `R_sum` and `A_sum` in `checkpoints.parquet`, `S_t_at_*` and so on in `report_runs.csv`, and `S_t`, `R_t` and `A_t` in `report_kernel.parquet`.
- **Events.**
  - Memorisation is the first checkpoint with training accuracy ≥ 0.99.
  - Grokking at level ℓ is the first checkpoint with test accuracy ≥ ℓ, counted from step 0.
  - A run that misses a level is kept and censored at 200,000 steps.
  - These are the definitions of the root `CLAUDE.md` and the golden report §3.5. Do not change them without telling the user.
- **Code layout.** Write one script per item as `report_v2/code/E<k>_<name>.py`, with no arguments and with seeds fixed at the top. Write the figure to `overleaf/figures/E<k>_<name>.pdf` and the numbers to `report_v2/code/out/E<k>.json`. Shared loaders, event definitions and colours go in `report_v2/code/common.py`, with one colour per statistic used by every figure. The last section of `EXPERIMENTS_TODO.md` describes the planned clean repository, so build toward it.
- **Reuse before writing.** Most items port a method from an existing notebook or script, which the item names. Examples are the placebo test in `repoduced-code/tier1_v2.ipynb` §7, the scaling fits in `repoduced-code/tier2_consistency.py`, and the prediction models in `model_fitting/3_kernel_only.ipynb` and `4_nonlinear_exploration.ipynb`. The old notebooks used old data and old definitions, so take their methods and never their numbers.
- **Environment.** Use Python 3.11 or later with the packages listed in `pyproject.toml` at the repository root. The fits need `lifelines` and `xgboost`, so check that both are installed before running them. Record the interpreter and the package versions with every output.
- **After a run.**
  - Mark the item done in `EXPERIMENTS_TODO.md`, with the date and what was run.
  - Replace the matching `\pending` values in `main.tex`.
  - Report what was run, what was measured and the numbers, as the root `CLAUDE.md` asks.
  - If a run failed or a value moved far from its planning value, say so plainly.
- **Long runs.** Do not start new training runs without the user's agreement. The data needed for every item already exist.

## Before you finish

1. `tectonic main.tex` builds cleanly from `report_v2/overleaf/`, and the log files are removed.
2. The main text ends by page 12, and the appendices fit in 5 pages.
3. The two greps above show nothing new.
4. Every new citation is in `ref.bib` and in the reference-check table.
5. The changelog, the open decisions and `EXPERIMENTS_TODO.md` match what you changed.
6. You have told the user about any claim that changed.

Do not commit, push or merge unless the user asks. Ask before you change the narrative, a decision listed above, or anything outside `report_v2/`.
