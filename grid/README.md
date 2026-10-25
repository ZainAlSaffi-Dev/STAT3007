# STAT3007 Group G2: what sets the grokking time on modular addition

<!-- SPHINX-START -->

The training code, the Snakemake grid and the report source. The Python tooling
comes from
[scientific-python/cookie](https://github.com/scientific-python/cookie) through
copier. [`.copier-answers.yml`](.copier-answers.yml) records the answers, and
`copier update` brings in later changes.

## Layout

| Path                                                         | Contents                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------- |
| [`src/deeplearning/`](src/deeplearning/), [`tests/`](tests/) | Python helper code and its generated tests                    |
| [`workflow/`](workflow/)                                     | Snakemake rules, scripts, environments and Friday profile     |
| [`config/`](config/)                                         | Grid and animation settings                                   |
| [`project/report/`](project/report/)                         | Report source, which reads `results/grid.duckdb` by PythonTeX |
| [`docs/`](docs/)                                             | Sphinx documentation                                          |

## Python environment

```bash
uv sync
```

## Checks

```bash
uvx nox
```

Tests are written by `hypothesis write` rather than by hand.

## Experiments on Friday

The grid in [`workflow/Snakefile`](workflow/Snakefile) runs through the global
profile `friday` and [`workflow/profiles/friday`](workflow/profiles/friday),
whose template sets each job's partition, GPU, time and memory from Slurm's own
records. A change to the training code is checked on one debug H100 before the
grid runs. Stage 1 is timed by the train rule's `benchmark` directive in an
unprofiled run, then compared with a reference run's stage-1 tables, from a
controller on a login node:

```bash
uv run --no-sync snakemake results/equivalence/stage1.duckdb --forcerun train --profile friday --workflow-profile workflow/profiles/friday --config reference=<reference results/stages/stage=1>
```

The `equivalence` rule stops on differing events or phases and keeps the
greatest differences of the measurements in the table `differences` of
`results/equivalence/stage1.duckdb`. GPU time is attributed in a second run
under Nsight Systems, which traces each kernel inside the captured training step
with `--cuda-graph-trace=node` (by default it traces a CUDA graph as one item).
Nsight profiles the process it launches. The controller runs inside one debug
job, with `--executor local` in place of the global profile's Slurm executor and
the memory the template gives training, the RealMemory of a CPU of an H100 node.
The trace is kept as Nsight's SQLite export, `results/profiles/stage1.sqlite`,
which `nsys stats` and DuckDB aggregate, and the report file is deleted:

```bash
sbatch --wait -p debug --gpus=h100:1 -t 60 --mem-per-cpu="$(scontrol show node --json | jq '[.nodes[] | select(.features | index("h100")) | .real_memory / .cpus | floor] | min')" --wrap 'mkdir -p results/profiles && conda run -n nsys nsys profile --cuda-graph-trace=node --export=sqlite -o results/profiles/stage1 uv run --no-sync snakemake results/stages/stage=1/events.parquet --forcerun train --profile friday --executor local --cores "$SLURM_CPUS_ON_NODE" && rm results/profiles/stage1.nsys-rep'
```
