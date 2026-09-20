# Animations for the G2 grokking project

Two audiences. `explainer/` holds longer pieces that explain the maths to the
group. `slides/` holds short silent clips for the presentation, which will be
built in PowerPoint or LaTeX. Every scene also writes a still frame, so the
report and the deck can share one picture.

The rules for writing a scene are in `CLAUDE.md` in this folder. Read it and the
root `CLAUDE.md` before adding anything.

## Setting up

```
brew install ffmpeg
brew install --cask mactex
cd animations && uv sync
```

MacTeX is about 5 GB and asks for a password. Manim needs it for `MathTex`,
because it calls `latex` and `dvisvgm` directly. Tectonic cannot stand in for
them. Open a new shell after the install so that `/Library/TeX/texbin` is on the
path.

## Rendering

```
cd animations
uv run manim -ql slides/scale_versus_rotation.py ScaleVersusRotation   # while working
uv run manim -qh slides/scale_versus_rotation.py ScaleVersusRotation   # for the deck
uv run manim -qh -s slides/scale_versus_rotation.py ScaleVersusRotation  # last frame as a still
```

`manim.cfg` fixes the frame size, the frame rate, the white background and the
output folders. Video goes to `out/mp4/`, which git ignores. Stills go to
`out/stills/`, which git keeps.

## State of the data

The saved runs and the shared helper `ntk_lib.py` are on `origin/sam` and are
not merged into this branch yet. The merge is waiting on Sam, because the two
tier notebooks call helpers that are missing from the pushed `ntk_lib.py`:
`load_cell`, `accuracy_crossings`, `cell_name`, `summarise_cell`, `tidy_axes`,
`relative_movement`, `first_step_at_least`, `fmt`, and the colour constants
`BLUE`, `GRAY`, `ORANGE` and `RAMP`. The notebook `tier1_alignment_threshold`
also imports plotly, which is not in the root `pyproject.toml`.

Until that merge lands, `common/data.py` raises with a message saying so. The
three schematic scenes below do not read data and can be built now.

The probe kernels are saved as `results/*_kernels.npz` and are in `.gitignore`,
so they never arrive through a merge. A scene that animates the kernel matrix or
its eigenvectors needs them regenerated locally first.

## Planned scenes

| Scene | Folder | Source | What it shows |
| --- | --- | --- | --- |
| `NTKFromScratch` | explainer | schematic | A nudge to the weights changes the output. The gradient vector is the feature map. `K(x, x')` is the similarity the network currently believes in. |
| `LazyVersusRich` | both | schematic, then `mean_field_alpha1` and `lazy_alpha10` | The one step algebra. The weights move by `eta_0 / alpha`, the output moves by an amount that does not depend on alpha. Large alpha slows feature change and leaves the loss curve alone. |
| `GrokkingTimeline` | slides | `mean_field_alpha1` | The two losses on a log step axis, both crossings marked, the gap labelled as the grokking time. |
| `ScaleVersusRotation` | both | schematic | The kernel as an ellipse. Shrink it with the axes fixed, which is the scale term. Turn it at fixed size, which is the rotation term. Then build Equation `eq:decomp` term by term and show the movement statistic growing while the rotation term stays at zero. |
| `TheConfound` | slides | tier 1 grid runs | Two cells with near equal movement and opposite mechanism. This is the figure the report calls the confound, demonstrated. |
| `ThreeTermsOneAxis` | slides | tier 1 grid runs | The scale, rotation and alignment terms with the two losses on one log step axis, with a moving time cursor. It asks whether the alignment crosses its threshold when the test loss falls. |
| `TwoClocks` | explainer | tier 2 fits, once they exist | What `a > 0, b ~ 0` looks like next to `a ~ 0, b ~ -1`. Build this after the fit has been run. |

The first, second and fourth are schematic and can start now. The rest wait for
the merge.
