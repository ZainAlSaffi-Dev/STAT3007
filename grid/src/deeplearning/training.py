"""Full-batch training of every seed of one grid cell in a single batched computation.

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
(Morris et al., 2019, Section 5.4).

Domains. Each field of a cell declares its valid range as annotated-types
metadata, which ``hypothesis`` reads when it generates test inputs and pydantic
enforces when a cell is built. Pydantic also converts NumPy scalars, which
pandas and Snakemake's ``Paramspace`` hand over, to the builtin types that
``torch.compile`` treats as constants. Sizes that grow the computation are
bounded by the largest values in ``config/grid.yaml``.

Kernels. Each kernel is the empirical NTK of one scalar output on every pair:
the sum of the logits over the square root of their number, the pseudo-NTK of
Mohamadi, Bae and Sutherland (2023), Equation 2, and the first logit, as in
Mohamadi et al. (2024), footnote 1.

Spectra. At every evaluated step each centred kernel is diagonalised, and the
power of the centred target along each eigenvector is recorded. Over all
``p^2`` pairs, which the data distribution weights uniformly, the matrix
divided by ``p^2`` is the integral operator of Canatar, Bordelon and Pehlevan
(2021), Equation 3, so eigenvalue over ``p^2`` is their ``eta_rho`` and power
over ``p^2`` is their ``eta_rho w_rho^2``, summed over outputs. Sorted by
eigenvalue, the cumulative share of power is their cumulative power
distribution, Equation 6, which the common factor leaves unchanged. Both the
kernel and the target are centred here, as for the statistics above.

Events. The first step at which training accuracy reaches the memorisation
level, and the first step at which test accuracy reaches each grokking level,
are recorded for every seed; Khanh et al. (2026) define both times this way.
The kernel is evaluated at every such step and on a log-spaced grid.
"""

from __future__ import annotations

import copy
import math
import sys
from functools import partial
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


@dataclass(frozen=True)
class Cell:
    """One cell of the grid: every seed shares these settings.

    Attributes:
        p: Modulus.
        train_fraction: Fraction of the ``p^2`` pairs used for training.
        width: Hidden width ``N``.
        parameterisation: ``"ntk"`` or ``"mean_field"``.
        alpha: Output scale of the centred predictor.
        eta_0: Base learning rate before division by ``alpha^2``.
        eta_lambda: Product of learning rate and L2 coefficient.
        steps: Step budget; a run that has not grokked by then is censored.
        seeds: Seeds trained together; each draws its own split and weights.
        memorised: Training accuracy that marks memorisation.
        grokked: Test accuracies that mark grokking.
        chunk_size: Seeds per batch when forming kernels; ``None`` for all.
    """

    p: Modulus
    train_fraction: Fraction
    width: Annotated[int, Ge(1), Le(max(c["width"] for c in GRID["cells"]))]
    parameterisation: Literal["ntk", "mean_field"]
    alpha: Annotated[
        float,
        Interval(
            ge=math.sqrt(sys.float_info.min * sys.float_info.epsilon),
            le=math.sqrt(sys.float_info.max),
        ),
    ]
    eta_0: Annotated[float, Gt(0)]
    eta_lambda: Annotated[float, Ge(0)]
    steps: Budget
    seeds: Annotated[tuple[Seed, ...], MinLen(1), MaxLen(len(GRID["fixed"]["seeds"]))]
    memorised: Level
    grokked: Annotated[
        tuple[Level, ...], MinLen(1), MaxLen(len(GRID["fixed"]["grokked"]))
    ]
    chunk_size: Annotated[int, Ge(1)] | None = None


class TwoLayer(nn.Module):
    """Bias-free two-layer network ``c_N W2 relu(W1 x / sqrt(D))``."""

    def __init__(
        self,
        inputs: int,
        width: int,
        outputs: int,
        parameterisation: str,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the network.

        Args:
            x: Inputs, one example per row.

        Returns:
            Logits, one row per example.
        """
        return self.readout * torch.relu(self.hidden * x @ self.w1.T) @ self.w2.T


def modular_addition(
    p: Modulus, train_fraction: Fraction, seed: Seed
) -> tuple[torch.Tensor, ...]:
    """Every pair ``(a, b)`` of residues, encoded one-hot, with a random split.

    Args:
        p: Modulus.
        train_fraction: Fraction of pairs assigned to training.
        seed: Seed of the split.

    Returns:
        Inputs ``(p^2, 2p)``, one-hot targets ``(p^2, p)``, and the indices of
        the training and test pairs.
    """
    eye = torch.eye(p, dtype=DTYPE)
    a, b = torch.cartesian_prod(torch.arange(p), torch.arange(p)).T
    x = torch.cat([eye[a], eye[b]], dim=1)
    y = eye[(a + b) % p]
    order = torch.randperm(p * p, generator=torch.Generator().manual_seed(seed))
    cut = int(train_fraction * p * p)
    return x, y, order[:cut], order[cut:]


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


def train(cell: Cell) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train every seed of a cell and measure its kernel along the way.

    Args:
        cell: The settings shared by the seeds.

    Returns:
        Kernel and loss measurements, one row per seed and evaluated step; the
        event steps, one row per seed and event with ``None`` when the event
        did not happen within the budget; and the spectra, one row per seed,
        evaluated step, kernel and eigenvector.

    Raises:
        ValueError: If the split leaves no training pairs or no test pairs.
    """
    torch.use_deterministic_algorithms(True)
    splits = [modular_addition(cell.p, cell.train_fraction, s) for s in cell.seeds]
    x, y = splits[0][:2]
    train_idx = torch.stack([split[2] for split in splits])
    test_idx = torch.stack([split[3] for split in splits])
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
        for s in cell.seeds
    ]
    params = {k: v.detach() for k, v in stack_module_state(models)[0].items()}
    base = copy.deepcopy(models[0]).to("meta")

    def fnet(prm: kernels.Params, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the network at parameters ``prm``.

        Args:
            prm: Parameters of one seed.
            inputs: Inputs, one example per row.

        Returns:
            Logits, one row per example.
        """
        logits: torch.Tensor = functional_call(base, (dict(prm),), (inputs,))
        return logits

    def fnet_single(prm: kernels.Params, xi: torch.Tensor) -> torch.Tensor:
        """Evaluate the network on one example, as the eNTK recipe requires.

        Args:
            prm: Parameters of one seed.
            xi: One input.

        Returns:
            The logits of that input.
        """
        return fnet(prm, xi.unsqueeze(0)).squeeze(0)

    f_0 = vmap(fnet, (0, None))(params, x)
    theta_0 = torch.cat([v.flatten(1) for v in params.values()], 1)

    def loss(
        prm: kernels.Params, f0: torch.Tensor, pairs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training loss of the centred, rescaled predictor.

        Args:
            prm: Parameters of one seed.
            f0: That seed's initial logits on every pair.
            pairs: That seed's training pairs.

        Returns:
            The mean squared error over the training pairs, and the predictor's
            outputs on every pair.
        """
        out = cell.alpha * (fnet(prm, x) - f0)
        return nn.functional.mse_loss(out[pairs], y[pairs]), out.detach()

    step_grad = torch.compile(vmap(grad(loss, has_aux=True)))
    eta = cell.eta_0 / cell.alpha**2
    target = y @ y.T
    test_target = target[test_idx[..., None], test_idx[:, None, :]]
    centred_target = y - y.mean(0)

    def sum_of_logits(prm: kernels.Params, xi: torch.Tensor) -> torch.Tensor:
        """Sum of the logits over the square root of their number.

        Args:
            prm: Parameters of one seed.
            xi: One input.

        Returns:
            The scaled sum as a one-element output.
        """
        return fnet_single(prm, xi).sum(-1, keepdim=True) / math.sqrt(cell.p)

    def first_logit(prm: kernels.Params, xi: torch.Tensor) -> torch.Tensor:
        """First logit of the network.

        Args:
            prm: Parameters of one seed.
            xi: One input.

        Returns:
            The first logit as a one-element output.
        """
        return fnet_single(prm, xi)[:1]

    outputs = {"sum": sum_of_logits, "first": first_logit}

    def kernel_matrices() -> dict[str, torch.Tensor]:
        """Kernel of each scalar output on every pair, for every seed.

        Returns:
            One ``seeds x n x n`` kernel per output.
        """
        return {
            name: vmap(
                partial(kernels.entk_jacobian_contraction, output, compute="trace"),
                in_dims=(0, None, None),
                chunk_size=cell.chunk_size,
            )(params, x, x)
            for name, output in outputs.items()
        }

    def measure(
        step: int, out: torch.Tensor
    ) -> tuple[list[dict[str, float | int]], pd.DataFrame]:
        """Measure losses, accuracies, norms, kernel statistics and spectra.

        Args:
            step: The step being measured.
            out: The predictor's outputs on every pair, one block per seed.

        Returns:
            One row per seed, and one row per seed, kernel and eigenvector.
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
            k_c = kernels.centre(k)
            k_c0 = initial[name]
            test_block = torch.take_along_dim(
                torch.take_along_dim(k, test_idx[..., None], dim=1),
                test_idx[:, None, :],
                dim=2,
            )
            row |= {
                f"S_{name}": kernels.scale(k_c, k_c0),
                f"R_{name}": kernels.shape(k_c, k_c0),
                f"D_{name}": kernels.movement(k_c, k_c0),
                f"A_{name}": kernels.alignment(k, target),
                f"A_test_{name}": kernels.alignment(test_block, test_target),
                f"A_uncentred_test_{name}": kernels.uncentred_alignment(
                    test_block, test_target
                ),
            }
            eigenvalues, eigenvectors = torch.linalg.eigh(k_c)
            spectra.append(
                pd.DataFrame(
                    {
                        "step": step,
                        "kernel": name,
                        "eigenvalue": eigenvalues.flatten().numpy(),
                        "power": (eigenvectors.mT @ centred_target)
                        .square()
                        .sum(-1)
                        .flatten()
                        .numpy(),
                    },
                    index=pd.MultiIndex.from_product(
                        [cell.seeds, range(k.shape[-1])], names=["seed", "order"]
                    ),
                )
            )
        return [
            {"step": step, "seed": seed} | {k: v[i].item() for k, v in row.items()}
            for i, seed in enumerate(cell.seeds)
        ], pd.concat(spectra)

    initial = {name: kernels.centre(k) for name, k in kernel_matrices().items()}
    grid = set(checkpoints(cell.steps).tolist())
    levels = {"memorised": [cell.memorised], "grokked": list(cell.grokked)}
    reached = {
        (e, lvl): torch.full((len(cell.seeds),), -1)
        for e, ls in levels.items()
        for lvl in ls
    }

    rows: list[dict[str, float | int]] = []
    spectra: list[pd.DataFrame] = []
    for step in range(cell.steps + 1):
        grads, out = step_grad(params, f_0, train_idx)
        predicted = out.argmax(-1)
        acc = {
            "memorised": (predicted.gather(1, train_idx) == labels[train_idx])
            .to(DTYPE)
            .mean(1),
            "grokked": (predicted.gather(1, test_idx) == labels[test_idx])
            .to(DTYPE)
            .mean(1),
        }
        new = False
        for (event, level), first in reached.items():
            hit = (acc[event] >= level) & (first < 0)
            first[hit] = step
            new |= bool(hit.any())
        if new or step in grid:
            measured, spectrum = measure(step, out)
            rows += measured
            spectra.append(spectrum)
        params = {
            k: (1 - cell.eta_lambda) * v - eta * grads[k] for k, v in params.items()
        }

    events = pd.DataFrame(
        [
            {
                "seed": s,
                "event": event,
                "level": level,
                "step": int(first[i]) if first[i] >= 0 else None,
                "budget": cell.steps,
            }
            for (event, level), first in reached.items()
            for i, s in enumerate(cell.seeds)
        ]
    )
    return pd.DataFrame(rows), events, pd.concat(spectra).reset_index()
