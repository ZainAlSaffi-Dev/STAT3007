"""Aalen's additive regression for a batch of bootstrap resampling vectors."""

from __future__ import annotations

import torch


def cumulative_regression(
    covariates: torch.Tensor,
    time: torch.Tensor,
    observed: torch.Tensor,
    cluster: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Aalen's cumulative regression functions under each resampling vector.

    Aalen (1989, Eq. 2) estimates the cumulative regression functions by the
    least squares inverse of the design of the subjects at risk, summed over
    the event times. A bootstrap sample of clusters weights each subject by the
    number of times its cluster is drawn, Efron and Tibshirani's (1993, Eq.
    14.20) resampling form, so the design of every resample is one product of
    the counts with the per-cluster sums of outer products. From the first
    event time at which the design loses full rank, by the numerical rank of
    ``torch.linalg.matrix_rank`` at its default tolerance, every increment is
    zero: Aalen (1989, Sec. 4.1) takes that time as a final censoring time.
    Columns that vary are scaled by their standard deviation before the
    Cholesky factorisation, as lifelines' fitter does.

    Args:
        covariates: Design matrix of shape ``(n, d)``, intercept included.
        time: Time of each subject's event or censoring, shape ``(n,)``.
        observed: Whether each subject's event was observed, shape ``(n,)``.
        cluster: Index of each subject's cluster in ``range(s)``, shape
            ``(n,)``.
        counts: Number of draws of each cluster, shape ``(..., s)``, in the
            dtype of ``covariates``.

    Returns:
        The cumulative regression functions at the sorted distinct observed
        times, shape ``(..., k, d)``.
    """
    event_times = torch.unique(time[observed])
    at_risk = (time[:, None] >= event_times).to(covariates.dtype)
    events = (observed[:, None] & (time[:, None] == event_times)).to(covariates.dtype)
    scale = covariates.std(0)
    scale = torch.where(scale > 0, scale, 1)
    x = covariates / scale
    members = torch.nn.functional.one_hot(cluster, counts.shape[-1]).to(x.dtype)
    gram = torch.einsum("ic,it,ip,iq->ctpq", members, at_risk, x, x)
    score = torch.einsum("ic,it,ip->ctp", members, events, x)
    design = torch.einsum("...c,ctpq->...tpq", counts, gram)
    factor, info = torch.linalg.cholesky_ex(design)
    full = (info == 0) & (
        torch.linalg.matrix_rank(design, hermitian=True) == design.shape[-1]
    )
    increments = torch.cholesky_solve(
        torch.einsum("...c,ctp->...tp", counts, score)[..., None], factor
    )[..., 0]
    stopped = (~full).cumsum(-1) > 0
    return torch.where(stopped[..., None], 0, increments).cumsum(-2) / scale
