# Code for the G2 report

This folder holds the code behind the G2 report for STAT3007. The report asks what sets the grokking time on modular addition, and it repeats two of its tests on MNIST. The folder has two parts.

- Part A rebuilds every number, figure and table of the report from a small data extract that is included here. It needs a CPU and about 25 minutes.
- Part B regenerates the data. The modular addition sweep took 2.7 hours on 12 CPU workers. The MNIST sweep needs a GPU and took 134 summed GPU hours.

The folders keep the names they have in our repository, so every path in the code works from this folder as it is. No file was edited for the submission. The analysis in Part A was last rerun in full from a standalone copy on 9 November 2026. The sweeps in Part B ran in October 2026.

## Folder map

| Folder | Contents |
|---|---|
| `report_v2/code/` | The analysis. One script per figure, table or set of numbers, and `make_all.py`, which runs them all. `data/` holds the extract. Its own `README.md` says which script makes what. |
| `repoduced-code/` | The modular addition sweep, `dataset_sweep.py`, and the modules it imports: `ntk_lib.py` (models, training and kernels), `RepoducedCode.py`, `ntk_trace.py`, `kernel_snapshots.py` and `dataset_features.py`. |
| `animations/common/` | Two small files that `ntk_trace.py` imports for its kernel helpers. |
| `model_fitting/` | `report_kernel.py` and `report_runs.py`, which turn the sweep into the two tables that the analysis reads. |
| `mnist/` | The MNIST sweep, `mnist_sweep.py`, with `mnist_data.py` and `mnist_features.py`. `Snakefile` records how we launched it on a GPU cluster. `exploration/` holds the small MNIST pilot that the report cites. |
| `pyproject.toml`, `uv.lock` | The environment of the training code. |

`mnist/README.md` also lists notebooks, checks and exploration files of our repository. Those are not part of this submission.

## Part A: rebuild the report from the extract

From `report_v2/code/`, run

```
python make_all.py
```

This needs Python 3.11 or later with numpy, pandas, pyarrow, scipy, matplotlib, scikit-learn, lifelines, xgboost and torch. We tested it with the versions in `report_v2/code/environment.txt`, which are Python 3.11.4, numpy 2.4.6, pandas 2.3.3, pyarrow 23.0.1, scipy 1.17.1, matplotlib 3.10.8, lifelines 0.30.0, scikit-learn 1.9.1, xgboost 3.2.0 and torch 2.10.0. `make_all.py` overwrites `environment.txt` with the versions of your run, so the list is repeated here. `uv.lock` pins newer versions of pandas and matplotlib for the training code, and we did not run the analysis under them. On macOS, xgboost also needs the OpenMP runtime, for example from `conda install llvm-openmp`.

The run took about 25 minutes on an Apple silicon Mac. Most of that time is the bootstrap in `E06_aft.py` and the MNIST fits in `E19_mnist.py`.

`make_all.py` first checks the extract in `data/` against `data/SHA256SUMS.json`. It writes its outputs to `report_v2/code/out/`. It then copies `numbers.tex`, the figures and the tables into `report_v2/overleaf/`, which it creates.

## Part B: regenerate the data

The scripts need the `git` program, since they record the commit of the code with each run. Outside a git repository that record is empty.

### B1. The modular addition sweep

The sweep has 720 runs: α in {0.5, 1, 2}, width in {50, 100, 200, 400, 800, 1600}, ηλ in {0, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3} and 5 seeds, with 200,000 steps each. We ran it on 3 October 2026 on an Apple silicon Mac with Python 3.14.4, torch 2.14.0, numpy 2.5.3 and pandas 3.0.6. It took 2.7 hours of wall time on 12 workers, 4 for the wide networks and 8 for the narrow ones, which is 18.9 hours of summed run time. Its output is about 11 GB, most of it the saved weights.

The chain of commands from the sweep to the report is below.

```
cd repoduced-code
python dataset_sweep.py check        # checks the kernels and the logger; about 2 minutes
python dataset_sweep.py pilot        # times short runs and projects the wall time
python dataset_sweep.py run          # the full sweep; rerun it to resume
python dataset_sweep.py assemble     # the tables and the manifest; it also runs validate
cd ../model_fitting
python report_kernel.py              # S, R, D and A of the report's kernel from the saved weights
python report_runs.py                # one row per run with the report's event times
cd ../report_v2/code
STAT3007_DATA="$PWD/../../model_fitting/data" python extract.py
python make_all.py
```

`dataset_sweep.py` writes to `model_fitting/data/`. The two scripts in `model_fitting/` read and write that folder. `extract.py` reads the sweep from the folder named by `STAT3007_DATA`, checks it, and writes a new extract to `report_v2/code/data/` with a new `SHA256SUMS.json`. Set `--wide-workers` and `--narrow-workers` on `run` to match your processor.

### B2. The MNIST sweep

The sweep has 560 runs: initial scale in {3, 5, 7, 9}, width in {100, 200, 400, 800}, seven decay levels from 0 to 4e-5 and 5 seeds, with 500,000 steps each. We ran it on 19 October 2026 on a SLURM cluster with Python 3.14.7 and torch 2.14.0 with CUDA 13.0. The runs of one width and one scale were trained together as one fused model on one GPU. They took 134 summed GPU hours on H100s with TF32 matrix multiplies.

From `mnist/`, run

```
python mnist_sweep.py check            # downloads MNIST from OpenML on first use
python mnist_sweep.py pilot --fused
python mnist_sweep.py run --fused      # rerun it to resume
python mnist_sweep.py assemble
```

`RUN_SWEEP.md` gives the steps in more detail. `Snakefile` records how we launched one job per width and scale. It names a cluster profile that is not included. `assemble` writes `mnist/sweep/`, and `extract.py` copies three of its tables into `report_v2/code/data/` with the prefix `mnist_`.

`extract.py` needs both sweeps. To rerun only the modular addition sweep, first copy our MNIST tables back: `report_v2/code/data/mnist_runs.parquet` to `mnist/sweep/runs.parquet`, `mnist_report_kernel.parquet` to `report_kernel.parquet`, and `mnist_manifest.json` to `manifest.json`.

### B3. The MNIST pilot

The report cites a pilot in which the centred two-layer network trains on MNIST. Its five runs and their summary are in `mnist/exploration/results/mnist_pilot/`. From `mnist/exploration/`, `python mnist_pilot.py summary kumar_b` rebuilds the summary from the saved runs, and `python mnist_pilot.py run kumar_b` trains them again. Both need MNIST, since the learning rate comes from the data.

## Data not included

- The full modular addition sweep, about 11 GB. This is `model_fitting/data/runs/` with the weights, the kernels and the history of each run, `checkpoints.parquet`, and the logs.
- The per-run files of the MNIST sweep. These are a few GB of weights, `checkpoints.parquet`, and the logs.
- The MNIST images. `mnist_data.py` downloads them into `mnist/data/` on first use.
- Our earlier saved runs of the baseline. `dataset_sweep.py validate` compares new runs with them when they are present and skips the comparison when they are absent.

## Definitions and seeds

The event times and the kernel statistics follow the sections "Event times" and "The second task" of the report. `report_v2/code/README.md` restates them. Every seed is fixed at the top of its script, and every output records the seeds it used.
