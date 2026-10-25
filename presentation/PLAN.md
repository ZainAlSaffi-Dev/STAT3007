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

The talk asks one question: what sets the grokking time? It answers in five acts. Each act opens on a dark slide with a question, and a small label at the top of every slide shows which act the audience is in.

| Act | Question on the act slide | What the audience should leave with | Core time |
|---|---|---|---|
| 1. Grokking | Why does a network take so much longer to generalise than to fit? | Grokking is a long delay. The literature disagrees about what sets it. | 1:25 |
| 2. The tangent kernel | What does it mean for a network to learn features? | The kernel is the network's similarity between inputs. It can grow or change shape, and a distance cannot tell the two apart. Theory says weight decay only rescales it. | 3:05 |
| 3. The experiment | How can one experiment tell the clocks apart? | Each account predicts a different result, and 720 runs cross the three knobs. | 1:00 |
| 4. Results | Which clock does the grokking time follow? | The distance is mostly growth. Rotation is tied to grokking at a fixed width but is not enough. Both width and weight decay set the time. Rotation is the best early predictor, and scale adds to it. | 3:30 |
| 5. What we found | So what sets the grokking time? | Weight decay sets the time through the weight norm on both tasks, and width enters only on modular addition. | 1:05 |

The times include 5 seconds for each act slide. The render log prints these totals every time the deck builds.

## 4. Slide by slide

The talk has 21 slides and five act slides, for 10:00 in all, since the MNIST slide grew from 0:20 to 0:30 on 22 October 2026 when the sweep results went in. On 18 October 2026 the nine optional slides moved to the backup, behind the closing slide, so they stay ready for questions. The speaker notes of every slide hold its message, a draft of what to say, and the budget with the running total.

| # | Act | Slide title (the message) | Visual | Budget |
|---|---|---|---|---|
| 1 | Opening | Title: Scale or Shape? | Title art, schematic | 0:15 |
| 2 | Grokking | A network can fit its training data long before it generalises | clip GrokkingTimeline | 0:35 |
| 3 | Grokking | The task is addition on a clock with 23 hours | clip ModularAdditionTable | 0:25 |
| 4 | Grokking | What sets the length of the delay is disputed | three cards | 0:20 |
| 5 | The tangent kernel | The tangent kernel says how a step on one input moves the output on another | clip NTKFromScratch | 0:40 |
| 6 | The tangent kernel | In the lazy regime the kernel stays fixed during training | clip LazyVersusRich | 0:25 |
| 7 | The tangent kernel | A kernel can grow, or it can change shape | clip ScaleVersusRotation | 0:25 |
| 8 | The tangent kernel | Three numbers describe how the kernel moved | three cards, one per statistic | 0:35 |
| 9 | The tangent kernel | The distance mixes growth and rotation | clip IdentityLevelSets | 0:20 |
| 10 | The tangent kernel | In theory, weight decay only rescales the kernel | clip KernelFlow | 0:25 |
| 11 | The experiment | Each account predicts a different result | table | 0:30 |
| 12 | The experiment | We trained 720 networks across laziness, width and weight decay | clip GridOfRuns | 0:25 |
| 13 | Results | Baseline: the network memorises early, groks late, and its kernel mostly grows | clip BaselineKernel | 0:30 |
| 14 | Results | Tier 1: at a fixed width, rotation is the statistic most tied to grokking | clip PlaceboTest | 0:45 |
| 15 | Results | Reaching that rotation is not enough to grok | text | 0:20 |
| 16 | Results | Tier 2: the grokking time depends on both width and weight decay | slide figure E5 | 0:40 |
| 17 | Results | Tier 3: rotation is the best single predictor, and scale adds what rotation lacks | live bar chart from E10 | 0:40 |
| 18 | Results | Tier 4: on MNIST the time is set by weight decay alone | slide figure E19 | 0:30 |
| 19 | What we found | Limitations, including the three things MNIST changes at once | text | 0:15 |
| 20 | What we found | Weight decay sets the time, through the weight norm | three cards | 0:30 |
| 21 | What we found | Thank you, with the AI-use statement | text | 0:05 |

The backup slides come after the closing slide and do not count toward the time or the slide numbers. First come the nine slides that were optional: the distance as the usual evidence, the arrows (clip KernelAsArrows), the pattern on the torus (clip KernelOnTheTorus), the three tests, the statistics at grokking (E2), the two clocks (clip TwoClocks), the decay edge (E4), width sets the rotation (E7) and the unseen $\alpha$ (E8). Then come the identity behind the split, the placebo test in numbers, the Tier 2 fit with its checks, Tier 3 on held-out cells and on an unseen $\alpha$ in numbers, how the prediction models were scored, the baseline beside Kumar et al., and the references.

## 5. Getting to ten minutes

The talk comes to 9:50. Five speakers hand over four times, which costs about 20 seconds, so the group may still want to save a little. These cuts were made on 18 October 2026.

1. The nine optional slides moved to the backup. Together they took 3:50.
2. The identity starts in its split form, on one slide. This saved 0:10.
3. The MNIST slide showed the setup and said the results were pending, which saved 0:10. That cut was undone on 22 October 2026, when the sweep results replaced the placeholder and the slide went to 0:30. The talk is now at 10:00, so the next cut in this list has to be made.
4. The limitations slide keeps the three points that bear on the claims, and the other two are in its notes. This saved 0:10.
5. Every deck clip plays three times faster, so each lasts 5 to 10 seconds. `SPEED` in `animations/common/layout.py` sets the factor.

If the talk still runs long, merge slide 15 into slide 14, with its two main sentences as a fragment under the tightening ratios, which saves 0:10, and speak the act slides in passing. To bring a backup slide back, move it to its place in the talk and give it a `time` attribute.

A possible split for five speakers follows the acts. Each act slide goes to the speaker who opens that act. The group may prefer fewer speakers, which saves handover time.

| Turn | Slides | Time |
|---|---|---|
| 1 | 1 to 4: the title and grokking | 1:40 |
| 2 | 5 to 8: the kernel, lazy and rich, growth and shape, the three numbers | 2:10 |
| 3 | 9 to 12: the identity, the theory, the predictions and the grid | 1:45 |
| 4 | 13 to 16: the baseline, Tier 1 and Tier 2 | 2:20 |
| 5 | 17 to 21: Tier 3, MNIST, the limitations and the conclusion | 1:55 |

## 6. How the deck explains the kernel

Act 2 carries most of the explaining, so it follows a fixed order and one picture.

1. **Start from something familiar.** A gradient step on one input moves the output on another input. The size of that move is the tangent kernel. This needs no infinite width and no theory, only the chain rule.
2. **Name the two regimes.** In the lazy regime the kernel stays fixed. In the rich regime it changes and the network learns features. The output scale $\alpha$ moves a network between them.
3. **Show why a distance is not enough.** The usual evidence for feature learning is the distance $D_t$ between the kernel now and at the start. A kernel that only grows can travel the same distance as one that changes shape.
4. **Give one number to each kind of movement.** Scale $S_t$ for growth, rotation $R_t$ for change of shape, and alignment $A_t$ for agreement with the labels.
5. **Show the identity.** $D_t^2 = (S_t - 1)^2 + 2 S_t R_t$, so a large distance can be pure growth.
6. **Bring in the theory.** In the kernel-flow equation of Lewkowycz and Gur-Ari, weight decay only rescales the kernel. A change of shape needs a term that shrinks with width.

The kernel is drawn as an ellipse throughout: its axes are the eigenvectors and their lengths are the eigenvalues. Growth makes the ellipse bigger, a change of shape turns or stretches it, and alignment is how well its long axis lines up with the labels. The title slide introduces the same three pictures, so the audience has seen them before Act 2 starts.

Colour carries meaning in every slide, figure and clip: scale is blue, rotation is orange and alignment is aqua. Accuracy curves are grey for training and ink for test, so they never read as a statistic. Width uses the violet of the report figures. The colours come from `report_v2/code/common.py`.

## 7. Animations

The deck has thirteen animations, and every one of them renders. The filter in `filters/g2.lua` swaps each placeholder for its clip as soon as the clip exists, so a scene goes into the deck by being rendered. The rules for writing a scene are in `animations/CLAUDE.md`, and `animations/README.md` describes each scene. The clips were built on 16 October 2026 on the machine that holds the full sweep. On 18 October 2026 they were sped up three times and committed, so the lengths below are those of the committed clips.

| Priority | Scene | Slide | Kind | Data | Length |
|---|---|---|---|---|---|
| 1 | `GrokkingTimeline` | 2 | measured | baseline $\alpha = 1$, seed 0, and its events in `report_runs.csv` | 6.6 s |
| 1 | `NTKFromScratch` | 5 | schematic, then measured | the weights of the baseline at step 0, kernel on all 529 pairs | 10.0 s |
| 1 | `ScaleVersusRotation` | 7 | schematic | exact 2 by 2 kernels | 8.3 s |
| 1 | `PlaceboTest` | 14 | measured | the 78 runs of width 100 that grok, with the 2,000 shuffles of E03 | 8.3 s |
| 2 | `LazyVersusRich` | 6 | schematic, then measured | the baseline at $\alpha$ = 0.5, 1 and 2, seed 0 | 8.3 s |
| 2 | `IdentityLevelSets` | 9 | identity, then measured | the baseline, seed 0 | 6.7 s |
| 2 | `KernelFlow` | 10 | schematic | Eqs. S5 and S6 of Lewkowycz and Gur-Ari, checked in arXiv v2 | 6.6 s |
| 2 | `ModularAdditionTable` | 3 | data | the split of data seed 42 | 5.0 s |
| new | `GridOfRuns` | 12 | design | the settings and counts of `report_runs.csv` | 5.0 s |
| new | `BaselineKernel` | 13 | measured | the five baseline seeds; replaces report figure E1 | 8.3 s |
| 3 | `TwoClocks` | backup | fit and predictions | the main row of `E06.json` | 6.7 s |
| 3 | `KernelAsArrows` | backup | schematic | arrows drawn for the picture | 6.6 s |
| 3 | `KernelOnTheTorus` | backup | measured | the weights of the baseline, kernel on all 529 pairs | 10.0 s |

These points were settled while building the clips.

- `animations/common/notation.py` now holds the report's definitions: the plain-ratio $S_t$, the uncentred $R_t$, the centred $A_t$ and $D_t^2 = (S_t - 1)^2 + 2S_tR_t$.
- `animations/common/palette.py` now draws training accuracy in grey and test accuracy in ink, as the report does, and every Text uses Helvetica Neue, the deck's fallback font.
- The seed 0 curves in `model_fitting/data/curves/` match `report_v2/code/data/checkpoints_slim.parquet` to within 4.8e-7. `animations/extract_deck_data.py` checks this every time it runs.
- `TurnAndAim` and `AimOverTraining` stay out of the deck, because their identity needs the centred $R_t$. They still render.
- Slide 12 plots the scale part against the rotation part on one scale, so that lines of equal $D_t$ are diagonals and the baseline path visibly stays close to the growth axis.
- `TwoClocks` draws two slices through the centre of the fit in place of three planes in 3D, which Manim's renderer could not show readably.
- On the torus slide, now in the backup, the change of the report's uncentred kernel falls on the shared-input cross and rises most along the sum line and the difference line. The scene asserts this before it draws the lines.
- No clip pauses in the middle. If a run-through shows that a clip outruns its speaker, end a Manim section there with `self.next_section()` and render with `--save_sections`. Each section is then its own clip, which the slide can name with the `clip` attribute.

## 8. Figures

The report figures were drawn for an A4 page, so their text is small on a projector. The results slides now show slide versions of them, drawn by `make_figures.py` into `assets/figures/` on 16 October 2026. Each one is 16 by 9, its type matches the clips (Helvetica Neue, the deck's fallback font, or Nimbus Sans where it is not installed, with formulas in the TeX fonts), and it keeps to the message of its slide. The script reads the committed data and the saved numbers of `report_v2/code`, so the numbers on the figures are those of the report.

| Figure | Slide | Source | State |
|---|---|---|---|
| E1 baseline | 18 | `report_v2/code/E01_baseline.py` | replaced by the clip `BaselineKernel` |
| E2 statistics at grokking | 20 | `E02_thresholds.py` | done: one hue per statistic in place of the shading by weight decay, and the tightening ratio on each panel |
| E4 time and outcome map | 24 | `E04_map.py` | done: no counts or phase letters, since every cell at $\alpha = 1$ has 0 or 5 seeds that grok |
| E5 scaling | 22 | `E05_scaling.py` | done. The report dropped this figure on 17 October 2026 and suggested it for the seminar, so the speaker notes say how to read it |
| E7 mechanism | 25 | `E07_mechanism.py` | done: the two maps stacked, without the number in each cell; the slide now has the figure and the text in two columns |
| E8 prediction ladder | 27 | `E08_ladder.py` | done: the mean bars only, the models in plain words, and the labels above the plot. Redrawn on 18 October 2026 with the rungs of the report figure of 17 October: each statistic alone, the trees and the recurrent network |
| E9 surface | none | `E09_surface.py` | dropped on 18 October 2026, since the report cut the figure and the claim that rotation and scale interact on 17 October |
| Tier 3 subsets | 26 | `E10.json` through `_variables.yml` | done: the filter draws the bars |
| GridDiagram | 16 | none | replaced by the clip `GridOfRuns` |
| E19 MNIST | 28 | `mnist/sweep/runs.parquet` and `E19.json` | done 22 October 2026: panels (a) the time against weight decay and (b) the weight norm at grokking, without the report's grid map, which does not read on a projector. The placeholder `MnistResult` is gone |

This plan first suggested a slide style inside the report scripts, so that both versions come from one place. The slide versions went into a separate script in `presentation/` instead, so that no file in `report_v2` changed. The cost is that the drawing code exists twice, and a change to a report figure has to be made in `make_figures.py` as well.

The report figures on the slides with one figure across the slide (E2 and E7) had been stretched sideways, because Quarto sizes a lone image to fill the slide and the theme then cut only its height. The theme now keeps the shape of every figure with `object-fit: contain`.

## 9. What the slides may claim

The limits in `report_v2/CLAUDE.md` apply to every slide and every speaker note. In short:

- Rotation is the statistic most tied to the grokking step only at a fixed width. Across widths no statistic passes the placebo test at level 0.95.
- Reaching the usual rotation does not make a network grok. Never say it does.
- The alignment barely changes between step 0 and grokking. Never call it a threshold or a clock.
- The interval of $\gamma$ includes zero, so its sign is not claimed.
- The Tier 2 exponents hold for the range in which every seed groks. Say so whenever $b$ is quoted.
- The kernel statistics do not predict better than the settings, and they give no earlier warning.
- The rate-addition model and every other step of Tier 4′ are our own. No source paper derives them, and Tier 4′ is not in the deck.

On MNIST (Tier 4), the limits come from `report_v2/code/out/numbers/E19.json`.

- The width exponent is zero and the decay exponent is minus one. Never say that width lengthens the time on MNIST.
- The law $T = \ln(s/n_c)/(\eta\lambda)$ is from Liu, Michaud and Tegmark (2023), Section 2. Do not present it as ours.
- Width sets the rotation and the rotation plays no part in the time. Do not say that rotation is absent on MNIST.
- Width 100 does not follow the law and is left out of the fits. Do not say it never groks.
- The reading that width enters only where the task forces the kernel to change shape is the group's own. Label it so, and cite Mohamadi et al. (2024) for the result it rests on.
- The curvature learning-rate rule sets the memorisation step. State it rather than hide it. Whether the group accepts the rule is still open in the report.

Every citation uses an entry from `ref.bib`, which copies the verified entries of the report unchanged. A new source has to be checked as the root `CLAUDE.md` asks before it goes into `ref.bib`.

## 10. Open decisions for the group

- **The date and the length of the slot.** The brief says week 13 and about ten minutes. Confirm both, and whether questions are inside the slot.
- **Speakers.** Who presents which part, and how many of us speak.
- **The title.** The report lists this as open: keep "Scale or Shape?" or change it to give the answer.
- **"Rotation" or "shape".** The report keeps "rotation" for $R_t$ for now. The deck follows the report and says once that $R_t$ measures any change of shape. If the report switches, change the deck too.
- **MNIST.** Settled on 22 October 2026. The sweep finished on 19 October, so slide 18 presents the results.
- **Optional slides.** Which of the eleven survive.
- **Laptop or room computer.** The HTML deck plays clips but needs our own laptop and, for the formulas, a network connection. The PDF works anywhere but shows clips as stills. Bring both.
- **The AI-use statement.** The closing slide has a draft. It should match the statement in the report.
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
