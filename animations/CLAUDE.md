# CLAUDE.md for the animations

This file governs everything under `animations/`. It adds to the rules in the
root `CLAUDE.md`; it does not replace them. Rule 1 on sources, Rule 2 on plain
language and its punctuation subsection, and Rule 3 on cross-checking all apply
here. Read that file first.

## What this folder is for

The group needs two kinds of animation. One kind explains the maths to the five
of us, where being right matters more than looking good. The other kind is short
clips for the presentation, which is built with Quarto in `presentation/`. The
deck finds a rendered clip in `out/mp4/` by its scene name. The two live in
`explainer/` and `slides/`. A still frame from either can go into the
report, so every scene exports one.

## Honesty

A curve that looks like a measurement must be a measurement. Load it from a
saved run through `common/data.py`. Do not paste an array of numbers into a
scene and do not draw a loss curve by hand.

A schematic is allowed when the point is conceptual. An ellipse standing for a
kernel is a schematic. Put the word "schematic" on screen when the picture is
one, so that nobody quotes it as a result.

Every number on screen traces back to something. For a measurement, name the run
and the step in a comment next to the line that draws it. For a number from a
paper, follow root Rule 1 and check it against the paper before it appears.
Write the source in a comment as author, year and section.

Notation comes from `common/notation.py`, which follows the restructured report,
`report_v2/overleaf/main.tex`, as the deck does. The report writes the grokking
step as `T` and the weight decay as `\eta\lambda`. A scene that invents its own
symbols gives the group a second vocabulary to learn. If a scene needs a symbol
that `notation.py` does not have, add it there first.

The three statistics are those of the report. They are measured on the tangent
kernel of the summed logits over the square root of p, on the 53 test pairs of
a run. `S_t` is a plain ratio of norms, so 1 means no change. `R_t` is one minus
the uncentred cosine to step 0. `A_t` is the centred alignment with the label
kernel. In the data they are the columns `S_sum`, `R_sum` and `A_sum`. The keys
`S_c` and `R_c` of the older saved runs hold the centred terms of the September
draft, which only the explainer scenes use.

## Text on screen

Root Rule 2 and its punctuation subsection apply to every title, caption, label
and axis name. No em dashes anywhere, including axis labels. No hyperbole.

A caption is one short sentence. If it wraps to a second line at `CAPTION` size
on a 1920 by 1080 frame, it is too long for a slide.

Formulas go through `MathTex` using the strings in `common/notation.py`. Do not
approximate a formula with unicode in a `Text` object. MacTeX provides the
`latex` and `dvisvgm` binaries that Manim calls. Tectonic cannot stand in for
them, because Manim invokes those two programs directly.

## Technical rules

Render at 1920 by 1080 and 30 frames per second. `manim.cfg` sets that, along
with the white background and the output folders. Do not override it per scene.

The background is white, so the Manim default of white strokes is invisible.
Give every mobject a colour from `common/palette.py`. Use `INK` where black
would be used.

One scene class per idea, named for the idea. `ScaleVersusRotation` is a good
name. `Scene2` is not.

Fix the seed inside the scene body for anything random, so the clip renders the
same way twice.

Use `-ql` while iterating. Use `-qh` for the version that goes into the deck.
Rendered video lands in `out/mp4/`, which is in `.gitignore` because the files
are large. Still frames land in `out/stills/` and are committed, because the
report uses them.

Every scene exports at least one still. Render it with `-s`, which saves the
last frame, so make the last frame the one worth keeping.

Anything two scenes share goes into `common/`. A colour goes into `palette.py`,
a symbol into `notation.py`, a list of runs into `data.py`, a drawing helper
into `layout.py` and a kernel computation into `kernels.py`. A scene file holds
only what is particular to that scene.

A scene is not finished until it renders from start to end without an error.
When you report a scene as done, say which command you ran and how long the
render took. If it failed, say that instead.

## Clips for the deck

A clip for the deck is named after the `scene` attribute of its placeholder in
`presentation/slides.qmd`, and it lives in `slides/`. The filter of the deck
plays the clip as soon as `out/mp4/<scene>.mp4` exists.

The slide heading states the message, so a clip has no title and no caption
bar. Short labels sit next to what they name.

A clip in a column of a slide is shown at about half size. Its labels use
`LABEL_COLUMN`, and a clip across the whole slide uses `LABEL_WIDE`. Check a
clip by looking at its still scaled down to the width of its column.

The clip lasts about as long as the `length` attribute of its placeholder. The
clip stops on its last frame, which is also the still in the PDF, so the last
frame shows the message of the slide.

A number on screen is read with `data.deck_number`, so the clip prints the same
string as the slide. Colours follow `palette.py`, where accuracy is grey for
training and ink for test, so that it never reads as a statistic.

## Where things are

- Run data: `repoduced-code/results/*.json`, loaded through `common/data.py`.
  `data.available()` lists them. `data.DENSE_RUNS` names the three runs that
  save the kernel every 250 steps.
- The probe kernels: `repoduced-code/results/*_kernels.npz`. These are ignored by
  git, so they have to be regenerated locally before any scene can animate the
  kernel matrix itself. `repoduced-code/kernel_snapshots.py` writes the dense
  ones.
- The feature dataset: `model_fitting/data/`. The seed 0 curves of every cell,
  `curves/<run_id>.npz`, and the runs table, `runs.csv`, are committed and are
  loaded through `data.load_curves` and `data.load_runs_table`. The columns
  are defined in `model_fitting/data/data_dictionary.md`. The curves match the
  report extract, which `extract_deck_data.py` checks.
- The report extract: `report_v2/code/data/`. `data.load_report_runs` reads its
  runs table, with the events of every run. `data.report_numbers` reads the
  results of a report item from `report_v2/code/out/numbers/`.
- The deck data: `data/`, written once by `extract_deck_data.py` with the root
  environment. It holds the 120 runs at width 100 at every checkpoint, the
  split of data seed 42, and the weights of the baseline run with seed 0 at its
  48 saved steps. `data.load_n100`, `data.load_split` and `data.load_weights`
  read them, and `kernels.sum_kernel` rebuilds the report's kernel from the
  weights.
- The numbers the slides print: `presentation/_variables.yml`, read through
  `data.deck_number`.
- A starting point for a new scene: `scene_template.py`, which renders as it
  stands.
- A check that every scene still renders: `uv run python render_all.py`.
- Metric definitions: `report_v2/overleaf/main.tex`, Section `sec:statistics`
  and Equation `eq:statistics`.
- The identity that ties D to S and R: the same file, Equation `eq:identity`.

## Which paper backs which idea

Check these before writing a caption that makes a claim.

- The laziness knob alpha, the centred predictor and the learning rate rescaling:
  Kumar et al. (2024), arXiv:2310.06110v3, Appendix 8.1, Equation 7.
- Alpha and the initialisation scale acting through one product: the same paper,
  Appendix 8.2 and Figure 7.
- The modular addition baseline: the same paper, Appendix 8.3.
- Large alpha giving lazy training, and the parameters moving by order one over
  alpha: Chizat, Oyallon and Bach (2019), arXiv:1812.07956v5, Section 2.1 and
  Theorem 2.2.
- Weight decay rescaling the kernel without turning it: Lewkowycz and Gur-Ari
  (2020), arXiv:2006.08643, Theorems 1 and 2, and Equation S5 for the exact
  evolution.
- Centred alignment: Cortes, Mohri and Rostamizadeh (2012), JMLR 13, pages 795
  to 828.

## Checklist before calling a scene done

1. It renders at `-qh` with no error.
2. Every measured curve came from `common/data.py`, and every number on screen
   from `data.deck_number` or from a loader there.
3. Every schematic says so on screen.
4. Every symbol came from `common/notation.py`.
5. A still frame is in `out/stills/`.
6. No em dash appears in any rendered text.
7. The README table has a row for the scene.
