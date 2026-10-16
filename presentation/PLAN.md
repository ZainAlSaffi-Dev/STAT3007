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
| 5. What we found | So what sets the grokking time? | A mixed clock, with honest limits. | 1:05 |

The times include 5 seconds for each act slide. The render log prints these totals every time the deck builds.

## 4. Slide by slide

"Core" slides make up the ten-minute version. "Optional" slides are in the deck so the group can see them, and they are the first to go. Each optional slide carries an amber "optional" tag next to its act label until the group decides. The speaker notes of every slide hold its message, a draft of what to say, and the budget with the running total.

| # | Act | Slide title (the message) | Visual | Status | Budget |
|---|---|---|---|---|---|
| 1 | Opening | Title: Scale or Shape? | Title art, schematic | core | 0:15 |
| 2 | Grokking | A network can fit its training data long before it generalises | clip GrokkingTimeline | core | 0:35 |
| 3 | Grokking | The task is addition on a clock with 23 hours | clip ModularAdditionTable | core | 0:25 |
| 4 | Grokking | What sets the length of the delay is disputed | three cards | core | 0:20 |
| 5 | The tangent kernel | The tangent kernel says how a step on one input moves the output on another | clip NTKFromScratch | core | 0:40 |
| 6 | The tangent kernel | In the lazy regime the kernel stays fixed during training | clip LazyVersusRich | core | 0:25 |
| 7 | The tangent kernel | The usual evidence is a distance, and a distance hides how the kernel moved | formula | optional | 0:20 |
| 8 | The tangent kernel | A kernel can grow, or it can change shape | clip ScaleVersusRotation | core | 0:25 |
| 9 | The tangent kernel | Three numbers describe how the kernel moved | three cards, one per statistic | core | 0:35 |
| 10 | The tangent kernel | Seen as arrows, the three numbers are a length and two angles | clip KernelAsArrows | optional | 0:25 |
| 11 | The tangent kernel | The distance mixes growth and rotation (first step) | formula | core | 0:10 |
| 12 | The tangent kernel | The distance mixes growth and rotation (second step) | clip IdentityLevelSets | core | 0:20 |
| 13 | The tangent kernel | On this task, the change of shape has a visible pattern | clip KernelOnTheTorus | optional | 0:30 |
| 14 | The tangent kernel | In theory, weight decay only rescales the kernel | clip KernelFlow | core | 0:25 |
| 15 | The experiment | Each account predicts a different result | table | core | 0:30 |
| 16 | The experiment | We trained 720 networks across laziness, width and weight decay | clip GridOfRuns | core | 0:25 |
| 17 | The experiment | Three tests, one for each column of the table | three cards | optional | 0:15 |
| 18 | Results | Baseline: the network memorises early, groks late, and its kernel mostly grows | clip BaselineKernel | core | 0:30 |
| 19 | Results | Tier 1: at a fixed width, rotation is the statistic most tied to grokking | clip PlaceboTest | core | 0:45 |
| 20 | Results | The statistics at the grokking step of every run | slide figure E2 | optional | 0:25 |
| 21 | Results | Reaching that rotation is not enough to grok | text | core | 0:20 |
| 22 | Results | Tier 2: the grokking time depends on both width and weight decay | slide figure E5 | core | 0:40 |
| 23 | Results | What the two clocks would look like | clip TwoClocks | optional | 0:25 |
| 24 | Results | A wider network groks later and stops grokking at a smaller weight decay | slide figure E4 | optional | 0:30 |
| 25 | Results | Width sets the rotation, and at a fixed width weight decay shrinks the scale | slide figure E7 | optional | 0:30 |
| 26 | Results | Tier 3′: rotation is the best single predictor, and scale adds what rotation lacks | live bar chart from E10 | core | 0:40 |
| 27 | Results | Growth shortens the grokking time more when the kernel has also rotated | slide figure E9 | optional | 0:25 |
| 28 | Results | The kernel statistics do not beat knowing the settings | slide figure E8 | optional | 0:30 |
| 29 | Results | Tier 4: does the split hold beyond modular addition? | figure MnistResult, to make | core | 0:30 |
| 30 | What we found | Limitations | text | core | 0:25 |
| 31 | What we found | Rotation matters most, and the clock is mixed | three cards | core | 0:30 |
| 32 | What we found | Thank you, with the AI-use statement | text | core | 0:05 |

The backup slides come after the closing slide and do not count toward the time or the slide numbers. They hold the identity behind the split, the placebo test in numbers, the Tier 2 fit with its checks, Tier 3′ in numbers, how the prediction models were scored, the baseline beside Kumar et al., and the references.

## 5. Getting to ten minutes

The core slides come to 10:20, and all slides together to 14:35. Five speakers hand over four times, which costs about 20 seconds. A safe target for the core is therefore about 9:30. These cuts get there, in this order.

1. Drop the first step of the identity (slide 11) and start on the second. This saves 0:10, and the equation appears at once in its split form.
2. Shorten the task slide (slide 3) to the table animation and one sentence. This saves 0:05.
3. If the MNIST sweep has no results by the talk, reduce slide 29 to two sentences on the setup. This saves 0:10.
4. Merge slide 21 into slide 19, with its two main sentences as a fragment under the tightening ratios. This saves 0:10.
5. Speak the act slides in passing, with no pause. This saves about 0:10 in all.

If the slot turns out to be longer, add optional slides back in this order: 25 (width sets rotation), 28 (the settings reference), 23 (the two clocks), 7 (the distance), 10 (the arrows), 13 (the torus), 24 (the decay edge), 27 (the interaction) and 20 (the statistics at grokking). To hide a slide without deleting it, add `visibility="hidden"` to its heading.

A possible split for five speakers follows the acts. It keeps each turn near two minutes, and each act slide goes to the speaker who opens that act. The group may prefer fewer speakers, which saves handover time.

| Turn | Slides | Core time |
|---|---|---|
| 1 | 1 to 4: the title and grokking | 1:40 |
| 2 | 5 to 9: the kernel, lazy and rich, growth and shape, the three numbers | 2:10 |
| 3 | 11 to 16: the identity, the theory, the predictions and the grid | 1:55 |
| 4 | 18 to 22: the baseline, Tier 1 and Tier 2 | 2:20 |
| 5 | 26 to 32: Tier 3′, MNIST, the limitations and the conclusion | 2:15 |

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

The deck has thirteen animations, and every one of them renders. The filter in `filters/g2.lua` swaps each placeholder for its clip as soon as the clip exists, so a scene goes into the deck by being rendered. The rules for writing a scene are in `animations/CLAUDE.md`, and `animations/README.md` describes each scene. The clips were built on 16 October 2026 on the machine that holds the full sweep.

| Priority | Scene | Slide | Kind | Data | Length |
|---|---|---|---|---|---|
| 1 | `GrokkingTimeline` | 2 | measured | baseline $\alpha = 1$, seed 0, and its events in `report_runs.csv` | 19.7 s |
| 1 | `NTKFromScratch` | 5 | schematic, then measured | the weights of the baseline at step 0, kernel on all 529 pairs | 29.9 s |
| 1 | `ScaleVersusRotation` | 8 | schematic | exact 2 by 2 kernels | 25.0 s |
| 1 | `PlaceboTest` | 19 | measured | the 78 runs of width 100 that grok, with the 2,000 shuffles of E03 | 25.0 s |
| 2 | `LazyVersusRich` | 6 | schematic, then measured | the baseline at $\alpha$ = 0.5, 1 and 2, seed 0 | 24.7 s |
| 2 | `IdentityLevelSets` | 12 | identity, then measured | the baseline, seed 0 | 20.0 s |
| 2 | `KernelFlow` | 14 | schematic | Eqs. S5 and S6 of Lewkowycz and Gur-Ari, checked in arXiv v2 | 20.0 s |
| 2 | `ModularAdditionTable` | 3 | data | the split of data seed 42 | 15.1 s |
| new | `GridOfRuns` | 16 | design | the settings and counts of `report_runs.csv` | 14.9 s |
| new | `BaselineKernel` | 18 | measured | the five baseline seeds; replaces report figure E1 | 24.9 s |
| 3 | `TwoClocks` | 23 | fit and predictions | the main row of `E06.json` | 20.0 s |
| 3 | `KernelAsArrows` | 10 | schematic | arrows drawn for the picture | 19.9 s |
| 3 | `KernelOnTheTorus` | 13 | measured | the weights of the baseline, kernel on all 529 pairs | 30.0 s |

These points were settled while building the clips.

- `animations/common/notation.py` now holds the report's definitions: the plain-ratio $S_t$, the uncentred $R_t$, the centred $A_t$ and $D_t^2 = (S_t - 1)^2 + 2S_tR_t$.
- `animations/common/palette.py` now draws training accuracy in grey and test accuracy in ink, as the report does, and every Text uses Helvetica Neue, the deck's fallback font.
- The seed 0 curves in `model_fitting/data/curves/` match `report_v2/code/data/checkpoints_slim.parquet` to within 4.8e-7. `animations/extract_deck_data.py` checks this every time it runs.
- `TurnAndAim` and `AimOverTraining` stay out of the deck, because their identity needs the centred $R_t$. They still render.
- Slide 12 plots the scale part against the rotation part on one scale, so that lines of equal $D_t$ are diagonals and the baseline path visibly stays close to the growth axis.
- `TwoClocks` draws two slices through the centre of the fit in place of three planes in 3D, which Manim's renderer could not show readably.
- On slide 13 the change of the report's uncentred kernel falls on the shared-input cross and rises most along the sum line and the difference line. The scene asserts this before it draws the lines.
- No clip pauses in the middle. If a run-through shows that a clip outruns its speaker, end a Manim section there with `self.next_section()` and render with `--save_sections`. Each section is then its own clip, which the slide can name with the `clip` attribute.

## 8. Figures

The report figures were drawn for an A4 page, so their text is small on a projector. The results slides now show slide versions of them, drawn by `make_figures.py` into `assets/figures/` on 16 October 2026. Each one is 16 by 9, its type matches the clips (Helvetica Neue, the deck's fallback font, with formulas in the TeX fonts), and it keeps to the message of its slide. The script reads the committed data and the saved numbers of `report_v2/code`, so the numbers on the figures are those of the report.

| Figure | Slide | Source | State |
|---|---|---|---|
| E1 baseline | 18 | `report_v2/code/E01_baseline.py` | replaced by the clip `BaselineKernel` |
| E2 statistics at grokking | 20 | `E02_thresholds.py` | done: one hue per statistic in place of the shading by weight decay, and the tightening ratio on each panel |
| E4 time and outcome map | 24 | `E04_map.py` | done: no counts or phase letters, since every cell at $\alpha = 1$ has 0 or 5 seeds that grok |
| E5 scaling | 22 | `E05_scaling.py` | done |
| E7 mechanism | 25 | `E07_mechanism.py` | done: the two maps stacked, without the number in each cell; the slide now has the figure and the text in two columns |
| E8 prediction ladder | 28 | `E08_ladder.py` | done: the mean bars only, the models in plain words, and the labels above the plot |
| E9 surface | 27 | `E09_surface.py` | done |
| Tier 3′ subsets | 26 | `E10.json` through `_variables.yml` | done: the filter draws the bars |
| GridDiagram | 16 | none | replaced by the clip `GridOfRuns` |
| MnistResult | 29 | `mnist/sweep/`, once it has run | waits on the sweep, which had not run on 16 October 2026 |

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

Every citation uses an entry from `ref.bib`, which copies the verified entries of the report unchanged. A new source has to be checked as the root `CLAUDE.md` asks before it goes into `ref.bib`.

## 10. Open decisions for the group

- **The date and the length of the slot.** The brief says week 13 and about ten minutes. Confirm both, and whether questions are inside the slot.
- **Speakers.** Who presents which part, and how many of us speak.
- **The title.** The report lists this as open: keep "Scale or Shape?" or change it to give the answer.
- **"Rotation" or "shape".** The report keeps "rotation" for $R_t$ for now. The deck follows the report and says once that $R_t$ measures any change of shape. If the report switches, change the deck too.
- **MNIST.** Present results if the sweep finishes in time, or present the setup and say the results are pending.
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
