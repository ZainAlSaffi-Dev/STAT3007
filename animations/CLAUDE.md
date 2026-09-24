# CLAUDE.md for the animations

This file governs everything under `animations/`. It adds to the rules in the
root `CLAUDE.md`; it does not replace them. Rule 1 on sources, Rule 2 on plain
language and its punctuation subsection, and Rule 3 on cross-checking all apply
here. Read that file first.

## What this folder is for

The group needs two kinds of animation. One kind explains the maths to the five
of us, where being right matters more than looking good. The other kind is short
clips for the presentation, which will be built in PowerPoint or LaTeX. The two
live in `explainer/` and `slides/`. A still frame from either can go into the
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

Notation comes from `common/notation.py`, which mirrors the macros in
`docs/report.tex`. The report writes the normalised kernel as `\widehat{K}`, the
centred kernel as `\widetilde{K}`, the grokking time as `t_{\mathrm{grok}}` and
the alignment threshold as `A_{*}`. A scene that invents its own symbols gives
the group a second vocabulary to learn. If a scene needs a symbol that
`notation.py` does not have, add it there first.

The three timing variables are defined on the centred kernel. In a saved run
those are the history keys `S_c` and `R_c`, together with `A_t`. The keys `S_t`
and `R_t` hold the raw versions. Use the centred ones unless the scene is about
the difference.

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

## Where things are

- Run data: `repoduced-code/results/*.json`, loaded through `common/data.py`.
  `data.available()` lists them. `data.DENSE_RUNS` names the three runs that
  save the kernel every 250 steps.
- The probe kernels: `repoduced-code/results/*_kernels.npz`. These are ignored by
  git, so they have to be regenerated locally before any scene can animate the
  kernel matrix itself. `repoduced-code/kernel_snapshots.py` writes the dense
  ones.
- A starting point for a new scene: `scene_template.py`, which renders as it
  stands.
- A check that every scene still renders: `uv run python render_all.py`.
- Metric definitions: `docs/report.tex`, the Kernel metrics part of the Methods
  section.
- The decomposition: `docs/report.tex`, Equation `eq:decomp`.

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
2. Every measured curve came from `common/data.py`.
3. Every schematic says so on screen.
4. Every symbol came from `common/notation.py`.
5. A still frame is in `out/stills/`.
6. No em dash appears in any rendered text.
7. The README table has a row for the scene.
