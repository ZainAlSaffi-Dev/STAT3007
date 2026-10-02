"""Full-batch training of every seed and setting of a stage in one batched computation.

Task. Modular addition with one-hot codes of ``a`` and ``b`` concatenated
(``2p`` inputs), a one-hot target for ``(a + b) mod p`` and all ``p^2`` pairs,
a fraction of them for training: Kumar et al. (2024), Section 8.3, and Gromov
(2023), Section 2.

Network. Two layers without biases and standard normal weights, with hidden
preactivation ``W1 x / sqrt(D)``: Gromov (2023), Equation 1. The readout is
scaled by ``1/N`` in the mean-field parameterisation (Gromov, Equation 2) or by
``1/sqrt(N)`` in the NTK parameterisation (Jacot et al., 2018, Section 2).

Predictor and step size. ``alpha [f(x; theta) - f(x; theta_0)]`` trained with
learning rate ``eta_0 / alpha^2``: Kumar et al. (2024), Section 8.1. The loss
is mean squared error averaged over every output of every training pair.

Weight decay. Coupled L2 under plain gradient descent, which coincides with
decoupled weight decay: Kim (2026), Remark on coupled and decoupled decay. The
grid sets the product ``eta * lambda`` directly.

Seeds. The seeds of a cell are stacked with ``torch.func.stack_module_state``
and trained together under ``vmap``, the model-ensembling pattern of the
PyTorch tutorials; full-batch gradient descent makes the batched run exact.
Each seed draws its own train-test split as well as its weights, since data
sampling is a leading source of variance (Bouthillier et al., 2021, Section 1),
and every cell uses the same seeds, so repetitions are matched across cells
(Morris et al., 2019, Section 5.4). The cells of a stage, which differ only in
their settings, are fused into the same batch, with each model's output scale,
learning rate and decay broadcast as vectors: the horizontally fused training
of Wang et al. (2021), Section 3, mathematically equivalent to training them
one by one.

Domains. Each field of a cell declares its valid range as annotated-types
metadata, which ``hypothesis`` reads when it generates test inputs and pydantic
enforces when a cell is built. Pydantic also converts NumPy scalars, which
pandas and Snakemake's ``Paramspace`` hand over, to the builtin types that
``torch.compile`` treats as constants. Sizes that grow the computation are
bounded by the largest values in ``config/grid.yaml``, and the output scale and
step sizes, which decide whether gradient descent stays finite, by the values
it states for the cells and the design.

Kernels. Each kernel is the empirical NTK of one scalar output on every pair:
the sum of the logits over the square root of their number, the pseudo-NTK of
Mohamadi, Bae and Sutherland (2023), Equation 2, and the first logit, as in
Mohamadi et al. (2024), footnote 1.

Statistics. For each seed, the scale, shape and movement compare the
uncentred kernel of that seed's test pairs with its value at initialisation:
the ratio of Frobenius norms, ``||K(t)||_F`` of Shan and Bordelon (2022),
over its initial value; one minus the cosine between the two, the kernel
distance of Fort et al. (2020); and the relative Frobenius distance, the total
relative variation of Geiger et al. (2020). The alignment is centred kernel
alignment with the target on the same test pairs, as Baratin et al. (2021)
measure it on held-out data, and its uncentred form is kept beside it.

Spectra. At every evaluated step each centred kernel is diagonalised, and the
power of the centred target along each eigenvector is recorded. Over all
``p^2`` pairs, which the data distribution weights uniformly, the matrix
divided by ``p^2`` is the integral operator of Canatar, Bordelon and Pehlevan
(2021), Equation 3, so eigenvalue over ``p^2`` is their ``eta_rho`` and power
over ``p^2`` is their ``eta_rho w_rho^2``, summed over outputs. Sorted by
eigenvalue, the cumulative share of power is their cumulative power
distribution, Equation 6, which the common factor leaves unchanged. Both the
kernel and the target are centred here.

Events. The first step at which training accuracy reaches the memorisation
level, and the first step at which test accuracy reaches each grokking level,
are recorded for every seed; Khanh et al. (2026) define both times this way.
The kernel is evaluated at every such step and on a log-spaced grid.
"""

from __future__ import annotations

import copy
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
import torch
import yaml
from annotated_types import Ge, Gt, Interval, Le, MaxLen, MinLen
from pydantic.dataclasses import dataclass
from torch import nn
from torch.func import functional_call, grad, stack_module_state, vmap

from deeplearning import kernels

DTYPE = torch.float64
GRID = yaml.safe_load(
    (Path(__file__).resolve().parents[2] / "config" / "grid.yaml").read_text(
        encoding="utf-8"
    )
)
Seed = Annotated[int, Interval(ge=-(2**31), le=2**31 - 1)]
Level = Annotated[float, Interval(ge=0, le=1)]
Modulus = Annotated[int, Ge(1), Le(max(c["p"] for c in GRID["cells"]))]
Fraction = Annotated[float, Interval(ge=0, le=1)]
Budget = Annotated[int, Ge(1), Le(max(c["steps"] for c in GRID["cells"]))]
Width = Annotated[int, Ge(1), Le(max(c["width"] for c in GRID["cells"]))]
Parameterisation = Literal["ntk", "mean_field"]


@dataclass(frozen=True)
class Setting:
    """Setting.

    Attributes:
        alpha: Output scale of the centred predictor.
        eta_0: Base learning rate before division by ``alpha^2``.
        eta_lambda: Product of learning rate and L2 coefficient.
    """

    alpha: Annotated[
        float,
        Interval(
            ge=min(c["alpha"] for c in GRID["cells"]),
            le=max(c["alpha"] for c in GRID["cells"]),
        ),
    ]
    eta_0: Annotated[float, Gt(0), Le(max(c["eta_0"] for c in GRID["cells"]))]
    eta_lambda: Annotated[
        float,
        Interval(
            ge=0,
            le=max(c["eta_0"] for c in GRID["cells"]) * max(GRID["scan"]["lambda"]),
        ),
    ]


@dataclass(frozen=True)
class Cell:
    """One cell of the grid: every seed shares these settings.

    Attributes:
        p: Modulus.
        train_fraction: Fraction of the ``p^2`` pairs used for training.
        width: Hidden width ``N``.
        parameterisation: ``"ntk"`` or ``"mean_field"``.
        settings: The hyper-parameters of each fused model, trained together
            with every seed.
        steps: Step budget; a run that has not grokked by then is censored.
        seeds: Seeds trained together; each draws its own split and weights.
        memorised: Training accuracy that marks memorisation.
        grokked: Test accuracies that mark grokking.
        threshold: Accuracy that the phase of a run is judged against.
        kernels: Whether to measure the kernel statistics and spectra; a run
            that only needs its phase measures accuracies alone.
    """

    p: Modulus
    train_fraction: Fraction
    width: Width
    parameterisation: Parameterisation
    settings: Annotated[tuple[Setting, ...], MinLen(1)]
    steps: Budget
    seeds: Annotated[tuple[Seed, ...], MinLen(1), MaxLen(len(GRID["fixed"]["seeds"]))]
    memorised: Level
    grokked: Annotated[
        tuple[Level, ...], MinLen(1), MaxLen(len(GRID["fixed"]["grokked"]))
    ]
    threshold: Level
    kernels: bool = True


class TwoLayer(nn.Module):
    """Bias-free two-layer network ``c_N W2 relu(W1 x / sqrt(D))``."""

    def __init__(
        self,
        inputs: Annotated[int, Ge(1), Le(2 * max(c["p"] for c in GRID["cells"]))],
        width: Width,
        outputs: Modulus,
        parameterisation: Parameterisation,
        generator: torch.Generator,
    ) -> None:
        """Draw standard normal weights and fix the layer scales.

        Args:
            inputs: Input dimension ``D``.
            width: Hidden width ``N``.
            outputs: Output dimension.
            parameterisation: ``"ntk"`` for ``c_N = N^(-1/2)``, ``"mean_field"``
                for ``c_N = 1/N``.
            generator: Source of the standard normal initial weights.
        """
        super().__init__()
        self.w1 = nn.Parameter(
            torch.randn(width, inputs, generator=generator, dtype=DTYPE)
        )
        self.w2 = nn.Parameter(
            torch.randn(outputs, width, generator=generator, dtype=DTYPE)
        )
        self.hidden = 1 / math.sqrt(inputs)
        self.readout = {"ntk": 1 / math.sqrt(width), "mean_field": 1 / width}[
            parameterisation
        ]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Evaluate the network on one-hot inputs, given by their hot positions.

        For an input with ones at ``tokens``, ``W1 x`` is the sum of the columns
        of ``W1`` at those positions, Gromov (2023), Claim I, where ``W1`` is a
        row of two ``N x p`` matrices. ``embedding`` looks the columns up, so the
        product with the dense one-hot matrix is never formed; ``embedding_bag``
        would sum them in one call but has no ``vmap`` batching rule.

        Args:
            tokens: Positions of the ones of each input, one row per example.

        Returns:
            Logits, one row per example.
        """
        preactivation = nn.functional.embedding(tokens, self.w1.T).sum(-2)
        return self.readout * torch.relu(self.hidden * preactivation) @ self.w2.T


def modular_addition(
    p: Modulus, train_fraction: Fraction, seed: Seed
) -> tuple[torch.Tensor, ...]:
    """Every pair ``(a, b)`` of residues, encoded one-hot, with a random split.

    Args:
        p: Modulus.
        train_fraction: Fraction of pairs assigned to training.
        seed: Seed of the split.

    Returns:
        Inputs ``(p^2, 2p)``, one-hot targets ``(p^2, p)``, the indices of the
        training and test pairs, and the positions ``(a, p + b)`` of the ones of
        each input, ``(p^2, 2)``.
    """
    eye = torch.eye(p, dtype=DTYPE)
    a, b = torch.cartesian_prod(torch.arange(p), torch.arange(p)).T
    x = torch.cat([eye[a], eye[b]], dim=1)
    y = eye[(a + b) % p]
    order = torch.randperm(p * p, generator=torch.Generator().manual_seed(seed))
    cut = int(train_fraction * p * p)
    return x, y, order[:cut], order[cut:], torch.stack([a, p + b], 1)


def checkpoints(
    steps: Annotated[int, Interval(ge=1, le=2**sys.float_info.mant_dig)],
) -> np.ndarray:
    """Log-spaced evaluation steps with step 0.

    Args:
        steps: Step budget, at most the largest integer a float64 holds
            exactly, since ``numpy.geomspace`` spaces the steps in float64.

    Returns:
        Sorted unique integer steps from 0 to ``steps``.
    """
    return np.unique(np.concatenate([[0], np.geomspace(1, steps).round().astype(int)]))


def train(cell: Cell) -> dict[str, pd.DataFrame]:
    """Train every seed of every setting and measure the kernels along the way.

    Args:
        cell: The settings of the fused models and what they share.

    Returns:
        The tables by name, each with the setting and seed of its model:
        ``metrics``, the loss, accuracy and kernel measurements, one row per
        model and evaluated step; ``events``, one row per model, event and
        level, with the duration and event columns of a lifelines fitter:
        ``time`` is the step of the event, or the budget when it did not
        happen, and ``observed`` is false for such a right-censored run;
        ``phases``, the phase of each model, judged at the final checkpoint
        as the ``fixed`` block of ``config/grid.yaml`` sets out; and, when
        ``cell.kernels`` is set, ``spectra``, one row per model, evaluated
        step, kernel and eigenvector.

    Raises:
        ValueError: If the split leaves no training pairs or no test pairs.
    """
    torch.use_deterministic_algorithms(True)
    splits = [modular_addition(cell.p, cell.train_fraction, s) for s in cell.seeds]
    y = splits[0][1]
    tokens = splits[0][4]
    # Every setting trains every seed, settings outermost: the models of the
    # repeated jobs fused into one, Wang et al. (2021), Section 3.
    keys = pd.DataFrame(
        [asdict(s) | {"seed": seed} for s in cell.settings for seed in cell.seeds]
    ).rename_axis("member")
    fused = len(cell.settings)
    train_idx = torch.stack([split[2] for split in splits]).repeat(fused, 1)
    test_idx = torch.stack([split[3] for split in splits]).repeat(fused, 1)
    members = torch.arange(len(keys))[:, None, None]
    if not (train_idx.shape[1] and test_idx.shape[1]):
        msg = "the split must leave both training and test pairs"
        raise ValueError(msg)
    labels = y.argmax(1)
    models = [
        TwoLayer(
            2 * cell.p,
            cell.width,
            cell.p,
            cell.parameterisation,
            torch.Generator().manual_seed(s),
        )
        for s in keys.seed.tolist()
    ]
    # The weights are drawn on the CPU, so every device starts from the same
    # ones, and training runs on the accelerator where one is present, as the
    # torch.accelerator.current_accelerator documentation chooses it.
    device = torch.accelerator.current_accelerator(check_available=True) or (
        torch.device("cpu")
    )
    alpha, eta_0, eta_lambda = torch.tensor(
        keys[["alpha", "eta_0", "eta_lambda"]].to_numpy().T, dtype=DTYPE, device=device
    )
    params = {
        k: v.detach().to(device) for k, v in stack_module_state(models)[0].items()
    }
    y, tokens, train_idx, test_idx, labels, members = (
        t.to(device) for t in (y, tokens, train_idx, test_idx, labels, members)
    )
    base = copy.deepcopy(models[0]).to("meta")

    def fnet(prm: kernels.Params, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the network at parameters ``prm``.

        Args:
            prm: Parameters of one seed.
            inputs: Positions of the ones of each input, one row per example.

        Returns:
            Logits, one row per example.
        """
        logits: torch.Tensor = functional_call(base, (dict(prm),), (inputs,))
        return logits

    f_0 = vmap(fnet, (0, None))(params, tokens)
    theta_0 = torch.cat([v.flatten(1) for v in params.values()], 1)
    # The fused optimiser broadcasts a vector of learning rates eta_0 / alpha^2
    # over the models' gradients, Wang et al. (2021), Section 3, in the coupled
    # L2 step theta - eta g - eta lambda theta.
    lr, decay = (v[:, None, None] for v in (eta_0 / alpha**2, eta_lambda))
    # One row per event and level: memorisation on training accuracy, and
    # grokking at each level on test accuracy.
    levels = [("memorised", cell.memorised)] + [("grokked", g) for g in cell.grokked]
    thresholds = torch.tensor([lvl for _, lvl in levels], dtype=DTYPE, device=device)
    on_test = torch.tensor([e == "grokked" for e, _ in levels], device=device)
    first = torch.full((len(levels), len(keys)), -1, device=device)
    # CUDAGraph Trees copy every input from eager into a static buffer before
    # each replay, but read parameters and buffers in place, in-place writes
    # included (PyTorch, CUDAGraph Trees, Input Mutation Support). Every tensor
    # a step reads or writes is therefore a buffer of one module, so the update,
    # the gradient and the event bookkeeping replay as one graph.
    state = nn.Module()
    for name, value in (
        params
        | {f"grad_{name}": torch.empty_like(value) for name, value in params.items()}
        | {
            "f_0": f_0,
            "tokens": tokens,
            "y": y,
            "labels": labels,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "alpha": alpha,
            "lr": lr,
            "decay": decay,
            "thresholds": thresholds,
            "on_test": on_test,
            "first": first,
            "step": torch.zeros((), dtype=first.dtype, device=device),
        }
    ).items():
        state.register_buffer(name, value)

    def loss(
        prm: kernels.Params, f0: torch.Tensor, pairs: torch.Tensor, a: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training loss of the centred, rescaled predictor.

        Args:
            prm: Parameters of one model.
            f0: That model's initial logits on every pair.
            pairs: That model's training pairs.
            a: That model's output scale ``alpha``.

        Returns:
            The mean squared error over the training pairs, and the predictor's
            outputs on every pair.
        """
        out = a * (fnet(prm, state.get_buffer("tokens")) - f0)
        return (
            nn.functional.mse_loss(out[pairs], state.get_buffer("y")[pairs]),
            out.detach(),
        )

    def evaluate() -> tuple[torch.Tensor, torch.Tensor]:
        """Gradients, outputs and events at the current step's parameters.

        Returns:
            The predictor's outputs on every pair, one block per model, and
            whether any model reached a level for the first time.
        """
        grads, out = vmap(grad(loss, has_aux=True))(
            {name: state.get_buffer(name) for name in params},
            state.get_buffer("f_0"),
            state.get_buffer("train_idx"),
            state.get_buffer("alpha"),
        )
        for name, value in grads.items():
            state.get_buffer(f"grad_{name}").copy_(value)
        predicted = out.argmax(-1)
        labels, test_idx, train_idx = (
            state.get_buffer(name) for name in ("labels", "test_idx", "train_idx")
        )
        accuracy = torch.where(
            state.get_buffer("on_test")[:, None],
            (predicted.gather(1, test_idx) == labels[test_idx]).to(DTYPE).mean(1),
            (predicted.gather(1, train_idx) == labels[train_idx]).to(DTYPE).mean(1),
        )
        first = state.get_buffer("first")
        hit = (accuracy >= state.get_buffer("thresholds")[:, None]) & (first < 0)
        first.copy_(torch.where(hit, state.get_buffer("step"), first))
        return out, hit.any()

    def advance() -> tuple[torch.Tensor, torch.Tensor]:
        """Take one gradient step, then evaluate at the new parameters.

        Returns:
            What ``evaluate`` returns at the next step.
        """
        lr, decay = state.get_buffer("lr"), state.get_buffer("decay")
        for name in params:
            value = state.get_buffer(name)
            value.sub_(lr * state.get_buffer(f"grad_{name}") + decay * value)
        state.get_buffer("step").add_(1)
        return evaluate()

    # torch.compile's default mode fuses the step without CUDA graphs; on a CUDA
    # device the fused step is then captured as a CUDA graph in the loop below.
    advance = torch.compile(advance)
    target = y @ y.T
    test_target = target[test_idx[..., None], test_idx[:, None, :]]
    centred_target = y - y.mean(0)

    # The scalar outputs u^T f: the sum of the logits over the square root of
    # their number, and the first logit.
    outputs = (
        {
            "sum": torch.ones(cell.p, dtype=DTYPE, device=device) / math.sqrt(cell.p),
            "first": torch.eye(cell.p, dtype=DTYPE, device=device)[0],
        }
        if cell.kernels
        else {}
    )

    def kernel_matrices() -> dict[str, torch.Tensor]:
        """Kernel of each scalar output on every pair, for every seed.

        Returns:
            One ``seeds x n x n`` kernel per output.
        """
        return {
            name: kernels.two_layer_entk(
                params["w1"], params["w2"], tokens, u, base.hidden, base.readout
            )
            for name, u in outputs.items()
        }

    def measure(
        step: int, out: torch.Tensor
    ) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
        """Measure losses, accuracies, norms, kernel statistics and spectra.

        Args:
            step: The step being measured.
            out: The predictor's outputs on every pair, one block per model.

        Returns:
            One row per model, and one spectrum per kernel, with one row per
            model and eigenvector.
        """
        variants = kernel_matrices()
        spectra = []
        theta = torch.cat([v.flatten(1) for v in params.values()], 1)
        out_train = torch.take_along_dim(out, train_idx[..., None], dim=1)
        out_test = torch.take_along_dim(out, test_idx[..., None], dim=1)
        row = {
            "train_loss": ((out_train - y[train_idx]) ** 2).mean((1, 2)),
            "test_loss": ((out_test - y[test_idx]) ** 2).mean((1, 2)),
            "train_acc": (out_train.argmax(-1) == labels[train_idx]).to(DTYPE).mean(1),
            "test_acc": (out_test.argmax(-1) == labels[test_idx]).to(DTYPE).mean(1),
            "weight_norm": torch.linalg.vector_norm(theta, dim=1),
            "parameter_movement": torch.linalg.vector_norm(theta - theta_0, dim=1)
            / torch.linalg.vector_norm(theta_0, dim=1),
        }
        for name, k in variants.items():
            # Each seed's test block, now and at initialisation.
            k_test, k0_test = torch.stack([k, initial[name]])[
                :, members, test_idx[..., None], test_idx[:, None, :]
            ]
            row |= {
                f"S_{name}": kernels.norm(k_test) / kernels.norm(k0_test),
                f"R_{name}": kernels.shape(k_test, k0_test),
                f"D_{name}": kernels.movement(k_test, k0_test),
                f"A_{name}": kernels.alignment(k_test, test_target),
                f"A_uncentred_test_{name}": kernels.frobenius(
                    kernels.unit(k_test), kernels.unit(test_target)
                ),
            }
            eigenvalues, eigenvectors = torch.linalg.eigh(kernels.centre(k))
            spectra.append(
                pd.DataFrame(
                    {
                        "step": step,
                        "kernel": name,
                        "eigenvalue": eigenvalues.flatten().numpy(force=True),
                        "power": (eigenvectors.mT @ centred_target)
                        .square()
                        .sum(-1)
                        .flatten()
                        .numpy(force=True),
                    },
                    index=pd.MultiIndex.from_product(
                        [keys.index, range(k.shape[-1])], names=["member", "order"]
                    ),
                )
            )
        return pd.DataFrame(
            {"step": step} | {k: v.numpy(force=True) for k, v in row.items()},
            index=keys.index,
        ), spectra

    initial = kernel_matrices()
    grid = set(checkpoints(cell.steps).tolist())

    rows: list[pd.DataFrame] = []
    spectra: list[pd.DataFrame] = []
    out, reached = evaluate()
    graph = torch.cuda.CUDAGraph() if device.type == "cuda" else None
    captured = None
    for step in range(cell.steps + 1):
        if step in grid or bool(reached):
            measured, spectrum = measure(step, out)
            rows.append(measured)
            spectra += spectrum
        if step == cell.steps:
            break
        if graph is None:
            out, reached = advance()
        elif captured is None:
            # A replay runs the whole step with one cudaGraphLaunch, skipping the
            # Python, C++ and driver dispatch of each kernel (PyTorch, CUDA
            # semantics, CUDA Graphs). The captured workload is first warmed up
            # on a side stream, here by the real step, once, as CUDAGraph Trees
            # warm up each graph before recording it.
            side = torch.Stream(device)
            side.wait_stream(torch.accelerator.current_stream())
            with side:
                out, reached = advance()
            torch.accelerator.current_stream().wait_stream(side)
            with torch.cuda.graph(graph):
                captured = advance()
        else:
            graph.replay()
            out, reached = captured

    events = pd.concat(
        pd.DataFrame(
            {
                "event": event,
                "level": level,
                "time": times.where(times >= 0, cell.steps).numpy(force=True),
                "observed": (times >= 0).numpy(force=True),
            },
            index=keys.index,
        )
        for (event, level), times in zip(levels, first, strict=True)
    )
    metrics = pd.concat(rows)
    final = metrics[metrics.step == cell.steps]
    fitted = final.train_acc > cell.threshold
    crossed = metrics.train_acc.groupby("member").max() > cell.threshold
    phases = pd.DataFrame(
        {
            "phase": np.select(
                [fitted & (final.test_acc > cell.threshold), fitted, crossed],
                ["grokking", "memorization", "forgetting"],
                "no fitting",
            )
        },
        index=final.index,
    )
    tables = {"metrics": metrics, "events": events, "phases": phases}
    if cell.kernels:
        tables["spectra"] = pd.concat(spectra)
    return {
        name: table.reset_index().join(keys, on="member").drop(columns="member")
        for name, table in tables.items()
    }
