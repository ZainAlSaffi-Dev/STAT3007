# Animations for the G2 grokking project

Two audiences. `explainer/` holds longer pieces that explain the maths to the
group. `slides/` holds short silent clips for the presentation, which will be
built in PowerPoint or LaTeX. Every scene also writes a still frame, so the
report and the deck can share one picture.

The rules for writing a scene are in `CLAUDE.md` in this folder. Read it and the
root `CLAUDE.md` before adding anything.

## Quick start

These steps are for a Mac. Run them once.

```
brew install --cask mactex-no-gui    # latex and dvisvgm, about 5 GB, asks for a password
brew install cairo pkgconf           # pycairo builds against these
cd animations
uv sync
```

Open a new shell after MacTeX installs, so that `/Library/TeX/texbin` is on the
path. Then check the toolchain:

```
uv run manim -ql smoke.py SmokeTest
```

If that writes `out/mp4/480p30/SmokeTest.mp4`, everything works. The kernel
scenes also need the probe kernels, which git does not store. Write them once
from the repository root environment:

```
cd ../repoduced-code
uv run python kernel_snapshots.py    # about 20 seconds per cell
```

### When the setup fails

Manim needs MacTeX for `MathTex`, because it calls `latex` and `dvisvgm`
directly. Tectonic cannot stand in for them.

`uv sync` builds `pycairo` from source, because it has no macOS wheel. It needs
cairo and pkg-config for the same architecture as the Python that uv picks,
and uv's Python on an Apple Silicon Mac is arm64. If your shell runs under
Rosetta, `brew` is the Intel Homebrew in `/usr/local`, and its cairo will not
link. Use the Apple Silicon Homebrew instead:

```
arch -arm64 /opt/homebrew/bin/brew install cairo pkgconf
PATH=/opt/homebrew/bin:$PATH uv sync
```

Manim 0.21 writes video through PyAV, so rendering works without the ffmpeg
binary.

## Rendering

```
cd animations
uv run manim -ql slides/kernel_on_the_torus.py KernelOnTheTorus       # while working
uv run manim -qh slides/kernel_on_the_torus.py KernelOnTheTorus       # for the deck
uv run manim -qh -s slides/kernel_on_the_torus.py KernelOnTheTorus    # last frame as a still
```

To check that every scene still renders after a change to `common/`:

```
uv run python render_all.py            # every scene at low quality
uv run python render_all.py torus      # only scenes whose name matches
uv run python render_all.py -q h -s    # deck-quality stills of every scene
```

`render_all.py` runs the scenes one at a time. Do not render several scenes in
parallel from separate shells. They share the LaTeX folder `out/Tex`, and one
render can delete the files another is still using.

`manim.cfg` fixes the frame size, the frame rate, the white background and the
output folders. Video goes to `out/mp4/`, which git ignores. Stills go to
`out/stills/`, which git keeps.

## How the folder is laid out

```
animations/
  common/              shared code; every scene imports from here
    data.py            loads saved runs and probe kernels from repoduced-code/results
    kernels.py         numpy versions of the kernel terms, the aim and the torus average
    layout.py          titles, captions, labels and heat maps
    notation.py        LaTeX strings for every symbol, matching docs/report.tex
    palette.py         colours and type sizes
  slides/              short clips for the deck
  explainer/           longer pieces for the group
  scene_template.py    a working scene to copy when starting a new one
  render_all.py        renders every scene and reports which ones fail
  smoke.py             checks that Manim, LaTeX and video output all work
  manim.cfg            frame size, frame rate, background and output folders
```

A scene holds only what is particular to it. Anything two scenes could share
goes into `common/`. A colour goes into `palette.py`, a symbol into
`notation.py`, a list of runs into `data.py`, a drawing helper into
`layout.py`, and a kernel computation into `kernels.py`.

## Adding a scene

1. Copy `scene_template.py` into `slides/` or `explainer/`, with a file name
   that says what it shows.
2. Change the path line to `parents[1]`, because `common/` is now one folder
   up. The comment in the template says this too.
3. Rename the class for the idea it shows, and rewrite the docstring and
   `construct()`.
4. Load every measured curve through `common/data.py`. `data.available()`
   lists the saved runs, and `data.history(run, "step", "S_c")` returns the
   series. Take every symbol from `common/notation.py` and every colour from
   `common/palette.py`.
5. Render it with `-ql` until it looks right, then with `-qh` and `-qh -s`.
6. Add a row to the table below and go through the checklist at the end of
   `CLAUDE.md`.

## State of the data

All saved runs are in `repoduced-code/results/`. They are one JSON file per
run, written by `ntk_lib.train_run`.

- The Tier 0 runs are named `mean_field_*`, `lazy_*` and `early_*`.
- The Tier 1 grid runs are named `ntk_N100_a{alpha}_wd{eta lambda}_s{seed}`.
- The dense runs are named `dense_N100_a1_wd{eta lambda}_s0`. `data.DENSE_RUNS` lists them.

The dense runs come from `repoduced-code/kernel_snapshots.py`. It retrains
three Tier 1 cells for 30,000 steps and saves the probe kernel every 250 steps,
together with the probe pairs. The three cells have alpha 1 and eta lambda 0,
0.0003 and 0.001. Their histories match the saved grid runs at all 61 shared
checkpoints. The losses are identical, and the kernel terms agree to within
1e-6. Their JSON files are committed. Their kernel files are about 27 MB each
and are not, so run the script once on each machine, as in the quick start.

`repoduced-code/term_dependence.py` holds the analysis behind the turn and aim
scenes.

## Scenes

| Scene | Folder | Source | Status | What it shows |
| --- | --- | --- | --- | --- |
| `NTKFromScratch` | explainer | schematic | planned | A nudge to the weights changes the output. The gradient vector is the feature map. `K(x, x')` is the similarity the network currently believes in. |
| `LazyVersusRich` | both | schematic, then `mean_field_alpha1` and `lazy_alpha10` | planned | The one step algebra. The weights move by `eta_0 / alpha`, the output moves by an amount that does not depend on alpha. Large alpha slows feature change and leaves the loss curve alone. |
| `GrokkingTimeline` | slides | `mean_field_alpha1` | planned | The two losses on a log step axis, both crossings marked, the gap labelled as the grokking time. |
| `ScaleVersusRotation` | both | schematic | planned | The kernel as an ellipse. Shrink it with the axes fixed, which is the scale term. Turn it at fixed size, which is the rotation term. Then build Equation `eq:decomp` term by term and show the movement statistic growing while the rotation term stays at zero. |
| `TheConfound` | slides | tier 1 grid runs | planned | Two cells with near equal movement and opposite mechanism. This is the figure the report calls the confound, demonstrated. |
| `ThreeTermsOneAxis` | slides | tier 1 grid runs | planned | The scale, rotation and alignment terms with the two losses on one log step axis, with a moving time cursor. It asks whether the alignment crosses its threshold when the test loss falls. |
| `TwoClocks` | explainer | tier 2 fits, once they exist | waits for the Tier 2 fit | What `a > 0, b ~ 0` looks like next to `a ~ 0, b ~ -1`. |
| `KernelOnTheTorus` | slides | the three dense runs | renders | The probe kernel averaged over offsets (a - a', b - b'). At step 0 it is a cross, which is plain input overlap. The target is the line a + b = a' + b'. Then the change of the normalised kernel in three cells from step 0 to 30,000, with a bar for the kernel size and the live R, A and test accuracy. The change forms an X in all three cells, strongest at eta lambda 0.0003. One arm is the target line. The other is the difference line a - b = a' - b', which the target does not contain. |
| `TurnAndAim` | explainer | identity, then `dense_N100_a1_wd0.0003_s0` | renders | The identity A = A_0 (1 - R) + gamma sqrt(1 - A_0^2) sqrt(R (2 - R)), built term by term. Then the step 0 kernel, the target and the current kernel as unit arrows at their measured angles. Most of the turn goes into directions the target does not see. |
| `AimOverTraining` | slides | the three dense runs | renders | Rotation, aim and alignment against step. The strongest decay turns early with an aim near zero. At step 30,000 the three cells have turned about as far and differ in aim. |
| `ScaleRotationLockstep` | slides | the three dense runs | renders | S against R. Without decay the two rise together. Decay pulls them apart. Exact level sets of D from Equation `eq:decomp` show that one value of D covers both kinds of path. |

"Renders" means the scene renders at `-ql` without an error. None of them has
been reviewed at `-qh` for the deck yet. The planned scenes are free to take.
The schematic ones, `NTKFromScratch`, `LazyVersusRich` and
`ScaleVersusRotation`, need no data. `GrokkingTimeline`, `TheConfound` and
`ThreeTermsOneAxis` can use the Tier 0 and Tier 1 runs that are already
saved.

## What the kernel scenes rest on

The aim gamma is new. It is the cosine between the target and the part of the
normalised centred kernel that is orthogonal to the step 0 kernel, divided by
its largest possible value. The identity above follows from that split, holds
for any kernel, and is our own algebra. It is not from a paper. It says that
scale cannot enter the alignment, and that alignment is fixed by rotation and
aim together. Over the 45 Tier 1 runs that reach test accuracy 1, R at that
event has a spread of 1.50 from largest to smallest. A at the event is
correlated with gamma at 0.98 and with R at 0.11. Along a run without decay, a
straight line in R explains a median 99.3 percent of the variance of S. That is
why one such run cannot separate scale from rotation. `term_dependence.py`
prints all of these numbers.
