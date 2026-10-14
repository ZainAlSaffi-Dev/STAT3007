# Options for the presentation

This note compares the tools we could use to build the slides for the G2 grokking talk. The talk needs two things from a tool. It must show equations that look the same as the ones in the report, and it must play the MP4 clips that we render with Manim in `animations/`. I checked each claim below against the tool's own documentation where I could find it, and the sources are listed at the end. Where I could not confirm a detail, I say so.

## What we already have

Our clips are rendered by Manim at 1920 by 1080 pixels and 30 frames per second on a white background. They are written to `animations/out/mp4/`, and a still frame of each scene goes to `animations/out/stills/`. The clips are short and silent, and the README in `animations/` says the deck was planned for PowerPoint or LaTeX. The report itself is written in LaTeX, so most of our equations already exist as LaTeX source that we can copy into the slides.

The project brief says the talk should give enough technical detail to count as a scientific presentation. That favours a tool that makes it easy to show a few equations and a few animations, with little text. The brief also asks for the slides to be uploaded as a `.ppt` or a `.pdf` file, so whichever tool we use must export one of those. (An earlier version of this note said that the brief forbids the talk from summarising the report. The brief says that about the reflective essay.)

## Summary table

| Tool | LaTeX maths | MP4 playback | Runs on | Main risk |
|------|-------------|--------------|---------|-----------|
| Quarto with reveal.js | Full LaTeX through MathJax | Inline or full-screen background | Any modern browser | Needs a browser and the video files on the day |
| Manim Slides | Whatever Manim can draw | Every slide is an animation | Its own player, a browser, or PowerPoint | Text-heavy slides are slow to build |
| Beamer with pdfpc | The same LaTeX as the report | Plays in pdfpc only | Needs pdfpc installed | Most PDF viewers will not play the video |
| Keynote | Native LaTeX equation editor | Drag and drop | Mac only | Group members on Windows cannot edit |
| PowerPoint with IguanaTex | LaTeX through an add-in | Drag and drop | Windows and Mac | The add-in needs MacTeX and Rosetta on our Macs |
| Slidev | LaTeX through KaTeX | A built-in video component | Any modern browser | Needs Node.js and some web knowledge |

## Option 1: Quarto with reveal.js

Quarto turns a Markdown file into a reveal.js slide deck that runs in a web browser. We write each slide as a heading followed by text, and we write maths with the usual `$...$` and `$$...$$` delimiters, which Quarto renders with MathJax. This means the equations from the report can be pasted in with very few changes.

Quarto has two documented ways to show an MP4. The first is the `video` shortcode, which places a player inside the slide, for example `{{< video ../animations/out/mp4/KernelOnTheTorus.mp4 width="100%" height="100%" >}}`. The second is the `background-video` attribute on a slide heading, which fills the whole slide with the clip. The background form also accepts `background-video-loop`, `background-video-muted`, `background-size` and `background-opacity`. Because our clips have a white background, a background video with a short heading on top would look clean.

Quarto has a speaker view that shows the current slide, the next slide, a timer and our notes. It can print the deck to a PDF from Chrome or Firefox, which gives us a static backup. The option `embed-resources: true` packs the deck into one HTML file, but I could not confirm from the Quarto documentation whether it also packs the video files. We should test this before relying on it, or simply carry the HTML file and the video folder together.

I think this is the best fit for us. It handles both maths and video well, the source is plain text that works with git, and every group member can edit it without a paid program.

## Option 2: Manim Slides

Manim Slides is a Python package that extends Manim. A scene inherits from `Slide` in place of `Scene`, and each call to `self.next_slide()` marks a pause where the presenter must press a key to continue. A call to `self.next_slide(loop=True)` makes a section repeat until the presenter moves on. This gives us control over the animation during the talk, which a plain MP4 cannot offer.

The command `manim-slides present` shows the deck in its own player, and its documentation describes this as the option with the most features. The command `manim-slides convert` can also produce an HTML deck that runs in any modern browser. With the `--one-file` flag it packs the animations into a single HTML file, and adding `--offline` packs the scripts and styles as well. There is a PowerPoint export through `--to=pptx`, which embeds every video in the file, but the project marks it as experimental. A PDF export exists too, although it keeps only one still image per slide.

The weakness of this tool is that every slide must be written as a Manim scene in Python. Title slides, bullet points and tables take more effort than they would in any of the other tools. Our existing scenes use `Scene`, so we would also need to change their base class and add the slide breaks.

A good middle path is to keep Quarto for the deck and use Manim Slides only for the one or two scenes where we want to pause and explain. Its HTML output is a reveal.js deck, so it should sit well beside a Quarto deck, although I have not tested that combination.

## Option 3: Beamer with pdfpc

Beamer is the LaTeX class for slides. It gives us exactly the same fonts and maths as the report, it compiles with Tectonic, and it can reuse our `ref.bib` and the macros in the report.

Video is the problem with Beamer. Beamer's own `\movie` command does not store the video inside the PDF. It only tells the PDF viewer to play a file that sits next to the PDF, and most viewers either ignore this or behave badly. The older `movie15` package is marked obsolete by its author, and `media9`, which replaced it, depends on Adobe Reader.

The workable route is the presenter program pdfpc together with the LaTeX package `pdfpc-movie`. The package documentation lists the options `autostart`, `loop` and `noprogress` for each movie. This works, but only on a laptop with pdfpc installed, and only if the video files travel with the PDF. If the room computer opens the PDF in a normal viewer, the videos will not play. We would need to present from our own laptop.

I would only choose this if the group wants the slides to match the report exactly and is happy to present from one of our laptops.

## Option 4: Keynote

Keynote has a built-in equation editor that accepts LaTeX. On a Mac it is under Insert, then Equation. Recent versions can also place equations inline inside a text box. MP4 files can be dragged straight onto a slide, and Keynote can play them automatically and loop them. It can export to PowerPoint and PDF.

Keynote only runs on Apple devices. If anyone in the group uses Windows, they would have to work through iCloud in a browser, which is more limited. The equations are typeset by Keynote, so they will look close to the report with small differences in font and spacing.

## Option 5: PowerPoint with IguanaTex

PowerPoint plays MP4 files without any extra setup. Its own equation editor accepts only part of LaTeX, so for our equations we would want the free add-in IguanaTex, which runs real LaTeX and inserts the result. The current release is 1.61. On a Mac it needs MacTeX, Ghostscript for vector output, and on Apple Silicon it needs PowerPoint to run under Rosetta. We already have MacTeX installed for Manim.

This is the safest choice if the talk must run on a room computer that we do not control, because almost every university machine has PowerPoint. Its weaknesses are that the file is binary and does not merge well in git, and that the IguanaTex setup on a Mac has several steps.

## Option 6: Slidev

Slidev is another Markdown slide tool that runs in a browser. It renders LaTeX with KaTeX and has a `SlidevVideo` component with `autoplay` and `controls` options. It also has a presenter view and exports to PDF. It is built on Vue and Node.js, so it suits a group that already knows web development. For us it offers little over Quarto, and Quarto fits our Python setup better.

## Tools I would avoid

Google Slides has no LaTeX support of its own, so every equation would need to be pasted in as an image. Typst slide packages are fast and handle maths well, but I could not find a supported way to play video in them.

## Recommendation

I suggest we build the deck in Quarto with reveal.js and embed our MP4 clips with the `video` shortcode or as background videos. We should export a PDF backup from the speaker view and bring the deck on a laptop as well as on a USB drive. If we want to pause inside an animation during the talk, we can convert that one scene to Manim Slides and link to it from the main deck.

If the group would rather not use a browser on the day, PowerPoint with IguanaTex is the most reliable fallback. Beamer with pdfpc is the right choice only if exact visual match with the report matters more to the group than ease of playing video.

## Things to test before we decide

We should check whether the presentation room lets us connect our own laptop or forces us to use its computer. We should render one clip at full quality and confirm it plays in each tool we are still considering. For Quarto, we should test whether `embed-resources: true` packs the video files into the HTML. For Manim Slides, we should test whether its HTML output can sit inside a Quarto deck.

## Sources

- Quarto, Videos: https://quarto.org/docs/authoring/videos.html
- Quarto, Presenting Slides: https://quarto.org/docs/presentations/revealjs/presenting.html
- reveal.js, Slide Backgrounds: https://revealjs.com/backgrounds/
- reveal.js, Math: https://revealjs.com/math/
- Manim Slides, Quickstart: https://manim-slides.eertmans.be/latest/quickstart.html
- Manim Slides, Sharing your slides: https://manim-slides.eertmans.be/reference/sharing.html
- Example of Quarto, reveal.js and Manim together: https://cameronrutherford.quarto.pub/presentations/math-magic/presentation.html
- Beamer user guide, animations section: https://mirrors.ircam.fr/pub/CTAN/macros/latex2e/contrib/beamer/doc/beamerug-animations.tex
- movie15 README, which marks the package obsolete: https://tug.ctan.org/macros/latex/contrib/movie15/README
- pdfpc-movie documentation: https://ctan.csail.mit.edu/macros/latex2e/contrib/pdfpc-movie/pdfpc-movie-doc.pdf
- Apple, Add mathematical equations in Keynote on Mac: https://support.apple.com/en-au/guide/keynote/tan72a69d01f/12.1/mac
- IguanaTex on GitHub: https://github.com/Jonathan-LeRoux/IguanaTex
- Slidev built-in components: https://slidev.antfu.me/builtin/components
