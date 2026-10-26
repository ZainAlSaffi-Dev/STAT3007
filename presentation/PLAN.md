# Plan for the G2 seminar

This file is the plan for the seminar talk. It sets out what the talk has to do, the story it tells, every slide with its time, the animations and figures still to make, and the decisions the group still has to take. The deck itself is `slides.qmd`, and `README.md` explains how to build it.

The deck is a skeleton. Every slide exists, with its message, its speaker notes and a time budget. Most visuals are still placeholders. Each placeholder carries a storyboard, so anyone can pick one up and build it.

## 1. What the talk has to do

The project brief (`project.pdf`, page 3) sets these requirements.

- The talk is most likely about 10 minutes. The exact length is announced closer to the date.
- It must convey the problem, its significance, the steps taken, the main findings, the limitations and the conclusion.
- It is marked on whether those points are clear and whether the claims are supported with enough technical detail. A talk without technical detail reads as opinion.
- The slides are uploaded as a `.ppt` or a `.pdf` file. We build the deck in Quarto, so we upload the PDF that `export_pdf.py` writes.
- The brief asks every deliverable to state how AI tools were used (page 1). The closing slide has a draft statement for the group to agree.

The topic is hard for this audience. Most of the class will not know the neural tangent kernel, lazy and rich training, or grokking. The talk therefore builds up slowly, explains one idea per slide, and spends about a third of its time on the background before any result appears.

## 2. Which document the deck follows

The deck follows `report_v2/overleaf/main.tex`, the restructured report, for its story, its claims and its numbers. That report holds the final results: 720 runs, three statistics on the tangent kernel of the summed logits over the 53 test pairs of each run, and the three tests. Every number on a slide is a macro from `report_v2/overleaf/numbers.tex`, copied into `_variables.yml` by `make_numbers.py`. The definitions of $S_t$ and $R_t$ are the same in the golden report and in `report_v2`. The alignment $A_t$ follows `report_v2`, which measures it on the summed-logit kernel.

The draft that was pasted into the chat on 14 October is the early skeleton of `docs/report.tex` from 20 September (commit `6e9ec1e`). It is older than both the golden report and `report_v2`, and some of its definitions changed afterwards. In that draft the scale is a log ratio, $S_t = \log(\lVert K_t \rVert_F / \lVert K_0 \rVert_F)$, all three statistics use the centred kernel, and the grokking time is a gap between two loss thresholds. The report now uses the plain ratio, the uncentred kernel for $S_t$ and $R_t$, and test accuracy 0.95 counted from step 0. The deck uses the current definitions. The early draft is still useful for its framing of the conflation and the three clocks, and the deck keeps that framing.

## 3. The story in five acts

The talk asks one question: what sets the grokking time? Its answer is a **mixed clock**. On modular addition the shape of the kernel is the most informative statistic, but it is not enough on its own, and width and weight decay both set the time. Each results slide ends with a "For the clock" line that says what it adds to that answer. The story was rebuilt on 25 October 2026 from the first round of feedback (Jasper, Zain and a third reviewer).

| Act | Question on the act slide | What the audience should leave with | Core time |
|---|---|---|---|
| 1. Grokking | Why does a network take so much longer to generalise than to fit? | Grokking is a long delay on a standard task, modular addition. | 1:05 |
| 2. The tangent kernel | What does it mean for a network to learn features? | The kernel is a snapshot of the features a network learns with. It can grow or change shape, a distance cannot tell which, and three numbers can. | 2:15 |
| 3. What sets the clock? | Which kind of kernel movement sets the grokking time? | Every published clock is scale-like. Theory says weight decay only rescales the kernel. Our hypothesis is that shape sets the clock, and 720 runs test it. | 1:50 |
| 4. Results | Which clock does the grokking time follow? | Shape is the most tied to grokking at a fixed width but is no threshold. Both width and weight decay set the time. Shape predicts best, and scale adds to it. On MNIST weight decay alone sets the time. | 3:15 |
| 5. What we found | So what sets the grokking time? | A mixed clock, with shape as its best single reading. | 1:00 |

The times include 5 seconds for each act slide and the 15 seconds of the title. The render log prints these totals every time the deck builds.

## 4. Slide by slide

The talk has 19 slides and five act slides, 24 in all, for a core of 9:40. Four handovers add about 20 seconds, so the talk should land near 10:00, the middle of the marked window of 9.5 to 10.5 minutes. The speaker notes of every slide hold its message and its script as dot points. `make_script.py` collects them into `script.md`.

| # | Act | Slide title (the message) | Visual | Budget |
|---|---|---|---|---|
| 1 | Opening | Title: Scale or Shape? | title art | 0:15 |
| 2 | Grokking | act slide | | 0:05 |
| 3 | Grokking | A network can fit its training data long before it generalises | clip GrokkingTimeline | 0:35 |
| 4 | Grokking | The task is modular addition, a standard test bed for grokking | still ModularAdditionTable | 0:25 |
| 5 | The tangent kernel | act slide | | 0:05 |
| 6 | The tangent kernel | The tangent kernel shows which inputs the network treats alike | text and formula | 0:35 |
| 7 | The tangent kernel | Grokking may be the move from a fixed kernel to one that learns features | still LazyVersusRich | 0:30 |
| 8 | The tangent kernel | A kernel can grow or change shape, and a distance cannot tell which | clip ScaleVersusRotation | 0:35 |
| 9 | The tangent kernel | Three numbers say how the kernel moved | three cards | 0:30 |
| 10 | What sets the clock? | act slide | | 0:05 |
| 11 | What sets the clock? | Every published clock is scale-like | three cards | 0:25 |
| 12 | What sets the clock? | In theory, weight decay rescales and width slows the change of shape | still KernelFlow | 0:30 |
| 13 | What sets the clock? | Our hypothesis: the shape of the kernel sets the grokking time | table, one row per click | 0:30 |
| 14 | What sets the clock? | We trained 720 networks across laziness, width and weight decay | table | 0:20 |
| 15 | Results | act slide | | 0:05 |
| 16 | Results | Tier 1: at a fixed width, shape is the statistic most tied to grokking | figure E3 | 0:45 |
| 17 | Results | No statistic is a fixed threshold, so shape is not the whole clock | figure E2 | 0:25 |
| 18 | Results | Tier 2: the grokking time follows both width and weight decay | the fit table (E5 in the backup) | 0:45 |
| 19 | Results | Tier 3: shape is the best single predictor, and scale adds what shape lacks | bar chart | 0:40 |
| 20 | Results | Tier 4: on MNIST, weight decay alone sets the time | figure E19 and a setup table | 0:35 |
| 21 | What we found | act slide | | 0:05 |
| 22 | What we found | Limitations | text | 0:20 |
| 23 | What we found | Grokking runs on a mixed clock, and shape is its best single reading | three cards | 0:30 |
| 24 | What we found | Thank you | | 0:05 |

The backup slides come after the closing slide and do not count toward the time or the slide numbers. They start with the AI-use statement, the baseline (clip BaselineKernel), the kernel of one network (clip NTKFromScratch), the split of the distance on a plane (clip IdentityLevelSets) and the placebo test step by step (clip PlaceboTest). The older backup slides follow.

## 5. Getting to ten minutes

The changes of 25 October 2026 brought the core from 10:00 to 9:40, so that the handovers fit inside the window.

1. Slides 9 and 11 of the old deck merged into one slide on the distance, which saved 0:10.
2. The baseline slide moved to the backup, which saved 0:30. Its two facts moved: the reproduction of Kumar et al. to the setup slide and the 90% scale share to the distance slide.
3. The slide on the published accounts moved from Act 1 to Act 3, after the kernel is defined.
4. Slides that lost text lost time: the tangent kernel 0:40 to 0:35, the three numbers 0:35 to 0:30, the published clocks 0:30 to 0:25 and the setup 0:25 to 0:20. Results kept their time.

If a timed run-through is still long, cut in this order: merge slide 17 into slide 16 with E2 in the backup (0:15), fold slide 11 into the columns of slide 13 (0:20), then speak the act slides in passing. If it is short, give the time to Tier 2.

A possible split for five speakers:

| Turn | Slides | Time |
|---|---|---|
| 1 | 1 to 4: the title, grokking and the task | 1:20 |
| 2 | 5 to 9: the kernel, lazy and rich, the distance, the three numbers | 2:15 |
| 3 | 10 to 14: the published clocks, the theory, the hypothesis and the grid | 1:50 |
| 4 | 15 to 18: Tier 1, its limits and Tier 2 | 2:00 |
| 5 | 19 to 24: Tier 3, MNIST, the limitations and the conclusion | 2:15 |

## 6. How the deck explains the kernel

Act 2 carries most of the explaining, so it follows a fixed order and one picture.

1. **Start from something familiar.** A gradient step on one input moves the output on another input. The size of that move is the tangent kernel. This needs no infinite width and no theory, only the chain rule.
2. **Name the two regimes.** In the lazy regime the kernel stays fixed. In the rich regime it changes and the network learns features. The output scale $\alpha$ moves a network between them.
3. **Show why a distance is not enough.** The usual evidence for feature learning is the distance $D_t$ between the kernel now and at the start. A kernel that only grows can travel the same distance as one that changes shape.
4. **Give one number to each kind of movement.** Scale $S_t$ for growth, shape $R_t$ for change of shape, and alignment $A_t$ for agreement with the labels.
5. **Show the identity.** $D_t^2 = (S_t - 1)^2 + 2 S_t R_t$, so a large distance can be pure growth.
6. **Bring in the theory.** In the kernel-flow equation of Lewkowycz and Gur-Ari, weight decay only rescales the kernel. A change of shape needs a term that shrinks with width.

The kernel is drawn as an ellipse throughout: its axes are the eigenvectors and their lengths are the eigenvalues. Growth makes the ellipse bigger, a change of shape turns or stretches it, and alignment is how well its long axis lines up with the labels. The title slide introduces the same three pictures, so the audience has seen them before Act 2 starts. Since 25 October 2026 the deck calls $R_t$ "shape" in every slide, figure and clip.

Colour carries meaning in every slide, figure and clip: scale is blue, shape is orange and alignment is aqua. Accuracy curves are grey for training and ink for test, so they never read as a statistic. Width uses the violet of the report figures. The colours come from `report_v2/code/common.py`.

## 7. Animations

The deck has thirteen animations, and every one of them renders. The filter in `filters/g2.lua` swaps each placeholder for its clip as soon as the clip exists, so a scene goes into the deck by being rendered. The rules for writing a scene are in `animations/CLAUDE.md`, and `animations/README.md` describes each scene. The clips were built on 16 October 2026 on the machine that holds the full sweep. On 18 October 2026 they were sped up three times and committed, so the lengths below are those of the committed clips.

| Scene | Slide | Shown as | Kind | Data | Change on 25 October 2026 |
|---|---|---|---|---|---|
| `GrokkingTimeline` | 3 | clip | measured | baseline $\alpha = 1$, seed 0, and its events in `report_runs.csv` | a "generalised" label, and each label in the colour of its curve |
| `ModularAdditionTable` | 4 | still | data | the split of data seed 42 | shown as a still |
| `LazyVersusRich` | 7 | still | schematic, then measured | the baseline at $\alpha$ = 0.5, 1 and 2, seed 0 | shown as a still |
| `ScaleVersusRotation` | 8 | clip | schematic | exact 2 by 2 kernels | labels "change of scale" and "change of shape", at column size |
| `KernelFlow` | 12 | still | schematic | Eqs. S5 and S6 of Lewkowycz and Gur-Ari, checked in arXiv v2 | redrawn as two panels: the scale part shrinks the kernel, and the shape part turns it at N = 100 and less at N = 1600 (our prediction) |
| `BaselineKernel` | backup | clip | measured | the five baseline seeds | moved to the backup with its slide |
| `NTKFromScratch` | backup | clip | schematic, then measured | the weights of the baseline at step 0 | moved to the backup |
| `IdentityLevelSets` | backup | clip | identity, then measured | the baseline, seed 0 | "shape part"; moved to the backup |
| `PlaceboTest` | backup | clip | measured | the 78 runs of width 100 that grok | replaced on slide 16 by the figure E3 |
| `TwoClocks` | backup | clip | fit and predictions | the main row of `E06.json` | "shape clock" |
| `KernelAsArrows` | backup | clip | schematic | arrows drawn for the picture | none |
| `KernelOnTheTorus` | backup | clip | measured | the weights of the baseline | none |
| `GridOfRuns` | none | | design | the settings and counts of `report_runs.csv` | dropped from the deck |

Only two slides still play a clip, slides 3 and 8, where the motion carries the message. The attribute `still="true"` shows the last frame of a clip on the others.

These points were settled while building the clips.

- `animations/common/notation.py` now holds the report's definitions: the plain-ratio $S_t$, the uncentred $R_t$, the centred $A_t$ and $D_t^2 = (S_t - 1)^2 + 2S_tR_t$.
- `animations/common/palette.py` now draws training accuracy in grey and test accuracy in ink, as the report does, and every Text uses Helvetica Neue, the deck's fallback font.
- The seed 0 curves in `model_fitting/data/curves/` match `report_v2/code/data/checkpoints_slim.parquet` to within 4.8e-7. `animations/extract_deck_data.py` checks this every time it runs.
- `TurnAndAim` and `AimOverTraining` stay out of the deck, because their identity needs the centred $R_t$. They still render.
- The clip `IdentityLevelSets`, now in the backup, plots the scale part against the shape part on one scale, so that lines of equal $D_t$ are diagonals and the baseline path visibly stays close to the growth axis.
- `TwoClocks` draws two slices through the centre of the fit in place of three planes in 3D, which Manim's renderer could not show readably.
- On the torus slide, now in the backup, the change of the report's uncentred kernel falls on the shared-input cross and rises most along the sum line and the difference line. The scene asserts this before it draws the lines.
- No clip pauses in the middle. If a run-through shows that a clip outruns its speaker, end a Manim section there with `self.next_section()` and render with `--save_sections`. Each section is then its own clip, which the slide can name with the `clip` attribute.

## 8. Figures

The report figures were drawn for an A4 page, so their text is small on a projector. The results slides now show slide versions of them, drawn by `make_figures.py` into `assets/figures/` on 16 October 2026. Each one is 16 by 9, its type matches the clips (Helvetica Neue, the deck's fallback font, or Nimbus Sans where it is not installed, with formulas in the TeX fonts), and it keeps to the message of its slide. The script reads the committed data and the saved numbers of `report_v2/code`, so the numbers on the figures are those of the report.

| Figure | Slide | Source | State |
|---|---|---|---|
| E1 baseline | none | `report_v2/code/E01_baseline.py` | replaced by the clip `BaselineKernel`, now in the backup |
| E2 statistics at grokking | 17 | `E02_thresholds.py` | one hue per statistic, "shape" for $R_t$, and the SD ratio of the placebo test on each panel; drawn for a column of 62% since 25 October 2026 |
| E3 placebo boxes | 16 | `E03_placebo.py` and the committed checkpoints | new on 25 October 2026: for each statistic, a box at each run's own grokking step and a box at another run's step, with the SD ratio and $p$. The script recomputes the ratios and checks them against `E03.json` |
| E4 time and outcome map | backup | `E04_map.py` | no counts or phase letters, since every cell at $\alpha = 1$ has 0 or 5 seeds that grok |
| E5 scaling | backup | `E05_scaling.py` | since 25 October 2026: no fitted lines, a shaded fit range, and the two reference slopes named by clock |
| E7 mechanism | backup | `E07_mechanism.py` | the two maps stacked, with "shape" for $R_t$ |
| E8 prediction ladder | backup | `E08_ladder.py` | the mean bars only, the models in plain words |
| E9 surface | none | `E09_surface.py` | dropped on 18 October 2026 with the claim that the two statistics interact |
| Tier 3 subsets | 19 | `E10.json` through `_variables.yml` | the filter draws the bars, all at once since 25 October 2026 |
| E19 MNIST | 20 | `mnist/sweep/runs.parquet` and `E19.json` | panels (a) the time against weight decay and (b) the weight norm at grokking |

This plan first suggested a slide style inside the report scripts, so that both versions come from one place. The slide versions went into a separate script in `presentation/` instead, so that no file in `report_v2` changed. The cost is that the drawing code exists twice, and a change to a report figure has to be made in `make_figures.py` as well.

The report figures on the slides with one figure across the slide (E2 and E7) had been stretched sideways, because Quarto sizes a lone image to fill the slide and the theme then cut only its height. The theme now keeps the shape of every figure with `object-fit: contain`.

## 9. What the slides may claim

The limits in `report_v2/CLAUDE.md` apply to every slide and every speaker note. In short:

- Shape is the statistic most tied to the grokking step only at a fixed width. Across widths no statistic passes the placebo test at level 0.95.
- Reaching the usual shape change does not make a network grok. Never say it does.
- The alignment barely changes between step 0 and grokking. Never call it a threshold or a clock.
- The interval of $\gamma$ includes zero, so its sign is not claimed.
- The Tier 2 exponents hold for the range in which every seed groks. Say so whenever $b$ is quoted.
- The kernel statistics do not predict better than the settings, and they give no earlier warning.
- The rate-addition model and every other step of Tier 4′ are our own. No source paper derives them, and Tier 4′ is not in the deck.

On MNIST (Tier 4), the limits come from `report_v2/code/out/numbers/E19.json`.

- The width exponent is zero and the decay exponent is minus one. Never say that width lengthens the time on MNIST.
- The law $T = \ln(s/n_c)/(\eta\lambda)$ is from Liu, Michaud and Tegmark (2023), Section 2. Do not present it as ours.
- Width sets the shape change and the shape change plays no part in the time. Do not say that the shape change is absent on MNIST.
- Width 100 does not follow the law and is left out of the fits. Do not say it never groks.
- The reading that width enters only where the task forces the kernel to change shape is the group's own. Label it so, and cite Mohamadi et al. (2024) for the result it rests on.
- The curvature learning-rate rule sets the memorisation step. State it rather than hide it. Whether the group accepts the rule is still open in the report.

Every citation uses an entry from `ref.bib`, which copies the verified entries of the report unchanged. A new source has to be checked as the root `CLAUDE.md` asks before it goes into `ref.bib`.

## 10. Open decisions for the group

- **The date and the length of the slot.** The brief says week 13 and about ten minutes. Confirm both, and whether questions are inside the slot.
- **Speakers.** Who presents which part, and how many of us speak.
- **The title.** The report lists this as open: keep "Scale or Shape?" or change it to give the answer.
- **"Rotation" or "shape".** Settled on 25 October 2026 by Jasper. The deck and the report both call $R_t$ "shape".
- **MNIST.** Settled on 22 October 2026. The sweep finished on 19 October, so slide 18 presents the results.
- **Optional slides.** Which of the eleven survive.
- **Laptop or room computer.** The HTML deck plays clips but needs our own laptop and, for the formulas, a network connection. The PDF works anywhere but shows clips as stills. Bring both.
- **The AI-use statement.** It moved from the closing slide to the first backup slide on 25 October 2026. It should match the statement in the report.
- **One reference title.** The Lewkowycz and Gur-Ari entry in the verified bibliography escapes its formula, so the reference list printed `$l_2$` literally. The arXiv abstract page, checked on 16 October 2026, writes the formula as $L_2$ in math with a capital L. The deck's `ref.bib` now does the same. The copies in `report_v2/overleaf/ref.bib` and `docs/ref.bib` still escape it, and changing them is the group's call.

## 11. Schedule

Count back from the day of the talk, T.

| When | What |
|---|---|
| Now | The group reads this plan and the skeleton, and settles the decisions in section 10. |
| T − 10 days | The four priority 1 animations render at low quality, and the notation and palette fixes of section 7 are in. |
| T − 7 days | All core animations render at high quality. The slide versions of E1 and E5 exist. First timed run-through. |
| T − 5 days | The cut to ten minutes is made. Speaker notes are rewritten by each speaker in their own words. |
| T − 3 days | Second timed run-through with every speaker. MNIST slide final. |
| T − 1 day | Final render, final PDF export, and one check of the PDF on another computer. Copy the deck, the clips and the PDF to a USB drive. |
