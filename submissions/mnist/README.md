# MNIST

This folder holds the MNIST extension of the project. It asks whether the
kernel decomposition used for modular addition also describes grokking on
MNIST, following the setup of Liu, Michaud and Tegmark (2023), Omnigrok,
arXiv:2210.01117. The network is a bias-free three layer ReLU network, so the
tangent kernel has a closed form. The shared training code is
`repoduced-code/ntk_lib.py`.

| Path | Contents |
| --- | --- |
| `mnist_data.py` | Loads MNIST: a random draw of training images and all 10000 test images. The download is cached in `data/`, which git ignores. |
| `mnist_grokking.ipynb` | Figures of grokking on MNIST from the exploration runs: the network, typical runs, the effect of initial scale, decay and width, the weight norm, the learning rate, the kernel, and where grokking happens. |
| `mnist_sweep.py` | The grid sweep over initial scale, width and weight decay with five seeds, in the style of `repoduced-code/dataset_sweep.py`. Its docstring gives the commands. |
| `mnist_features.py` | The columns the sweep records at every checkpoint, with the meaning and source of each. |
| `sweep/` | The output of the sweep, once it has run. |
| `model_fitting/` | MNIST copies of Zain's four model-fitting notebooks from the zain branch. They read `sweep/` through `mnist_settings.py`, which holds the MNIST settings: initial scale s for alpha, grokking at test accuracy 0.80, the 500000 step budget and the landmark step. Their takeaways are left to write once the sweep has run. |
| `checks/` | `verify_closed_form_deep.py` checks the closed-form kernel against autograd. `regression_modadd.py` checks that the changes to `ntk_lib.py` leave a modular addition run unchanged. |
| `exploration/` | The scripts, plan and results of the first MNIST study: the pilot, Option B, and the grid pilot. `exploration/results/mnist_optionB/REPORT.md` reports what was found. |

## Running the sweep

`RUN_SWEEP.md` gives the steps in short. Run from this folder with the environment at the repository root.

```bash
uv run python mnist_sweep.py check      # a few minutes
uv run python mnist_sweep.py pilot      # times every width and projects the wall time
uv run python mnist_sweep.py run --widths 100 200    # a batch; rerunning resumes
uv run python mnist_sweep.py assemble   # tables, data dictionary, manifest and validation
```

The full grid is 4 initial scales by 4 widths by 7 decay levels by 5 seeds,
560 runs of 500000 steps. The default is 8 workers in total. Change this
with `--wide-workers` and `--narrow-workers` to match the number of processor
cores.

## Event definitions

Memorisation is the first checkpoint with training accuracy at least 0.99, as
in the golden report. MNIST does not reach the report's test levels of 0.90
and 0.95 reliably, so the primary grokking event on MNIST is test accuracy
0.80. The sweep also records the levels 0.50 to 0.90.
