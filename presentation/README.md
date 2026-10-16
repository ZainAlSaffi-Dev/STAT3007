# The G2 seminar deck

This folder holds the slides for the seminar talk. They are built with Quarto as a reveal.js deck, which was option 1 in `docs/presentation_options.md`. `PLAN.md` gives the story, every slide with its time, the animations and figures still to make, and the open decisions. Read it first.

The rules in the root `CLAUDE.md` apply to every slide, caption and speaker note. In short: cite only entries from `ref.bib`, write plain full sentences without dashes, and check anything taken from a paper against the paper.

## Building the deck

Install Quarto once from <https://quarto.org/docs/get-started/>. The deck was built with Quarto 1.10.19, and any 1.10 release should work. Then, from the repository root:

```bash
quarto preview presentation/slides.qmd
```

This opens the deck in a browser and reloads it on every save. To write the HTML file without a preview:

```bash
quarto render presentation/slides.qmd
```

The render log ends with lines that start with `[g2]`. They give the time of the core slides and of all slides against the target of ten minutes, the time of each act, and the animations and figures that do not exist yet.

In the browser, press `S` for the speaker view, with the notes, the next slide and a timer. Press `F` for full screen, `Esc` for an overview of all slides, and `M` for the menu.

## Exporting the PDF

The course takes the slides as a `.ppt` or a `.pdf` file, so we upload a PDF. Render the deck, then run:

```bash
python presentation/export_pdf.py
```

It writes `presentation/exports/slides.pdf`, one page per slide, using Chrome, Chromium or Edge in headless mode. If it cannot find a browser, set the `CHROME` environment variable, or run `quarto install chrome-headless-shell`. By hand, open `slides.html?print-pdf` in Chrome and print to PDF with no margins and with background graphics on.

A slide with a clip prints the clip's still, so render the stills before the final export. Optional slides print too, so hide the ones the group cuts before exporting.

## What is where

```
presentation/
  slides.qmd         the deck
  PLAN.md            the plan: story, slides, times, animations, figures, decisions
  _quarto.yml        the format: theme, size, transitions, filter, bibliography
  _variables.yml     every number on the slides; written by make_numbers.py
  make_numbers.py    copies report_v2/overleaf/numbers.tex into _variables.yml
  export_pdf.py      writes the PDF for upload
  filters/g2.lua     act labels, clips, bar charts and the time budget
  theme/g2.scss      colours, fonts and the layout classes
  ref.bib            the references the slides cite, copied from verified bibliographies
  assets/stills/     stills to use in place of animations/out/stills, if needed
  assets/video/      clips to use in place of animations/out/mp4; ignored by git
```

## Writing a slide

Each level 2 heading is a slide, and its title states the message of the slide. Each level 1 heading with the class `act` opens an act, with the act's question in the paragraph below it.

```markdown
## Rotation is tied to grokking at a fixed width {time="0:45"}
```

`time` is the budget of the slide, in minutes and seconds. Add the class `.optional` to mark a slide that can be cut, and `visibility="hidden"` to leave a slide out without deleting it. Put what to say in a `::: {.notes}` block. The filter adds the budget and the running total at the top of the notes.

### Numbers

Never type a result. Every number from the experiments is a variable with the name of its macro in `report_v2/overleaf/numbers.tex`:

```markdown
The width exponent is {{< var EsixMainA >}}.
```

A number that the report sets in math mode, such as a signed value or a power of ten, arrives as inline math, for example `$-0.63$`, and MathJax renders it. Any other number is plain text. No Unicode character stands in for math anywhere in the deck: write a formula, a prime or a minus sign in `$...$`.

When `report_v2/code/make_all.py` changes the numbers, run `python presentation/make_numbers.py` and render again. The comment next to each variable in `_variables.yml` names the JSON file and the key it comes from. Settings such as the widths or the step budget are not results and can be typed.

### Animations

An animation is a div that names its Manim scene and holds its storyboard:

```markdown
::: {.anim scene="NTKFromScratch" status="planned" source="schematic" length="30 s"}
1. A small network with two inputs and their outputs as two bars.
2. One gradient step pulls the output at x towards its target.
:::
```

While no clip exists, the slide shows an amber box with the storyboard. Once the scene is rendered in `animations/`, the filter finds the clip in `animations/out/mp4/` and plays it on the slide, with the still from `animations/out/stills/` as its poster. A file in `assets/video/` or `assets/stills/` with the same name takes precedence. Add `clip="Name"` when the file name differs from the scene, and `loop="true"` for a clip that should loop. The rules for writing a scene are in `animations/CLAUDE.md`.

A div with the class `figure-todo` and a `name` stands for a static figure still to make, in the same amber style.

### Bar charts

A div with the class `g2-bars` draws a bar chart from report variables, so a chart never shows a typed number:

```markdown
::: {.g2-bars .stepwise max="0.6" label="Held-out $R^2$ of $\log T$"}
- [$A_t$]{.is-alignment var="EtenARsq"}
- [$\log R_t$]{.is-rotation var="EtenRRsq"}
- [All three]{var="EtenSRARsq"}
:::
```

The class of each label picks the colour of its bar. `.stepwise` shows one bar per click.

### Colours and layout

The three statistics have one colour each in every slide, figure and clip: scale is blue, rotation is orange and alignment is aqua.

- `[rotation $R_t$]{.g2-key .is-rotation}` puts a swatch of the colour before ink text. Use it in tables and result boxes.
- `\Sc`, `\Rc` and `\Ac` are the coloured symbols for formulas.
- `.c-scale`, `.c-rotation` and `.c-alignment` colour a word or two of text. They use darker steps of each colour so the text stays readable.
- `.g2-cards` lays out two or three cards side by side. Give each card `.g2-card` and a colour class, `.is-scale`, `.is-rotation`, `.is-alignment` or `.is-ink`, and start it with `[Title]{.g2-card-title}`. Do not start a card with a `###` heading, because pandoc would turn the card into a section and reveal.js would count it as a slide.
- `.g2-result` is a grey box for one result. `.g2-lead` is a large sentence that carries the slide. `.g2-small` and `.g2-source` are small print. `.g2-compact` makes a crowded column smaller.
- `![](path){.g2-fig}` places a figure. `[Report figure E1, to restyle for slides]{.g2-todo-chip}` marks an interim figure.

The colours are set in `theme/g2.scss`, and they match `report_v2/code/common.py` and `animations/common/palette.py`. Change them in all three places or not at all.

### Citations

Cite with `[@key]` or `@key`, using the keys in `ref.bib`. The reference list is built on the last backup slide. Do not start a line with `@key.`, because pandoc reads that as a numbered example list. To add a source, check it as the root `CLAUDE.md` asks, copy its entry from `docs/ref.bib` or `report_v2/overleaf/ref.bib` if it is there, and say in a comment where it came from.

## On the day

The HTML deck plays the clips, but it needs the rendered `slides.html`, the `slides_files/` folder, the clips, and a network connection for the formulas and the fonts. Present it from one of our laptops. Bring the PDF as well, on the laptop and on a USB drive, in case the room computer has to be used.
