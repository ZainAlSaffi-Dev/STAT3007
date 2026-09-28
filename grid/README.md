# STAT3007 Deep Learning

<!-- SPHINX-START -->

Course notes, worked problems and experiment code for STAT3007. The Python
tooling comes from
[scientific-python/cookie](https://github.com/scientific-python/cookie) through
copier. [`.copier-answers.yml`](.copier-answers.yml) records the answers, and
`copier update` brings in later changes.

## Layout

| Path                                                                               | Contents                                                         |
| ---------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| [`main.tex`](main.tex)                                                             | Document root; the one place the parts and chapters are ordered  |
| [`notes/`](notes/)                                                                 | `<N>_<part>/<NN>_<chapter>.tex`, numbered as the PDF prints them |
| [`styles/`](styles/)                                                               | Shared preamble: packages, theorem styles, macros                |
| [`References/`](References/)                                                       | BibTeX database                                                  |
| [`lectures/`](lectures/), [`tutorials/`](tutorials/), [`practicals/`](practicals/) | Course material as supplied                                      |
| [`assignments/`](assignments/)                                                     | Assignment write-ups                                             |
| [`data/`](data/)                                                                   | Inputs                                                           |
| [`src/deeplearning/`](src/deeplearning/), [`tests/`](tests/)                       | Python helper code and its generated tests                       |

The notes build in construction order: each chapter uses only objects defined in
the chapters before it. Chapter 0 fixes notation ahead of Part I, so a file's
prefix is its chapter number and a folder's prefix its part number.

## Build

```bash
latexmk main.tex
```

The macro layer in [`styles/latex_packages.tex`](styles/latex_packages.tex)
supplies the probability, convergence and linear-algebra shorthands used in the
notes (`\E`, `\Prob`, `\cvD`, `\argmin`, `\norm` and `\inner` among them),
theorem environments numbered per chapter, and `Pin`/`Pout` listing environments
for Python input and output.

## Python environment

```bash
uv sync
```

CUDA wheels are pinned for Windows in [`pyproject.toml`](pyproject.toml), and
other platforms resolve to the default PyTorch index.

## Checks

```bash
uvx nox
```

Tests are written by `hypothesis write` rather than by hand.
