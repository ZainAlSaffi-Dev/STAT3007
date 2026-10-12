"""
Verification of the closed-form kernel of a bias-free ReLU network of any depth.

This is section 2.3 of mnist/exploration/PLAN_mnist_closed_form.md. It checks
ntk_lib.entk_closed_form_deep in four ways and prints PASS or FAIL for each.

1. It equals ntk_lib.entk_closed_form for the two layer NTKMLP and
   MeanFieldMLP on a modular addition probe, to a relative Frobenius error
   below 1e-10 in float64.
2. It equals the autograd kernel for depth 2, 3 and 4, both
   parameterisations and init_scale 1 and 9, on a probe of 64 MNIST images,
   to a relative Frobenius error below 1e-6. The autograd kernel is computed
   twice. The first is ntk_lib.entk on a float64 model, which accumulates the
   Gram matrix in float32 on this path. The second accumulates in float64,
   one output at a time, with the same torch.func calls. Both must pass.
3. The same holds after 500 steps of minibatch SGD on a 3 layer network.
4. It times one probe kernel of 256 images for the 3 layer width 200
   network, by the closed form and by autograd.

    uv run python verify_closed_form_deep.py
"""

import sys
import time
from pathlib import Path

import torch
from torch.func import functional_call, vmap, grad

# ntk_lib and RepoducedCode live in repoduced-code at the repository root, and mnist_data in mnist/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "repoduced-code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ntk_lib as L
from mnist_data import load_mnist
from RepoducedCode import make_modular_addition_dataset

torch.set_num_threads(4)
RESULTS = []


def rel(A, B):
    A, B = A.double(), B.double()
    return (torch.linalg.norm(A - B) / torch.linalg.norm(B)).item()


def report(label, err, tol):
    ok = err < tol
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label}: relative error {err:.3e} (tolerance {tol:g})")


def entk_float64(model, x):
    """The sum-over-outputs kernel by autograd, one output at a time, accumulated in float64."""
    params = {k: v.detach() for k, v in model.named_parameters()}
    n_out = functional_call(model, params, x[:1]).shape[1]
    K = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64)
    for c in range(n_out):
        def f(p, xs, c=c):
            return functional_call(model, p, xs.unsqueeze(0)).squeeze(0)[c]
        g = vmap(grad(f), (None, 0))(params, x)
        G = torch.cat([t.flatten(start_dim=1).double() for t in g.values()], dim=1)
        K += G @ G.T
    return K


def check_two_layer():
    print("Check 1: two layer models against entk_closed_form on a modular addition probe")
    (Xtr, ytr), (Xte, yte) = make_modular_addition_dataset(p=23, train_fraction=0.9, seed=42)
    x, _, _ = L.select_probe(Xtr, ytr, Xte, yte, "mixed", 256)
    for par in ("ntk", "mean_field"):
        for scale in (1.0, 3.0):
            torch.manual_seed(0)
            model = L.build_model(par, x.shape[1], 100, 23, "relu", scale)
            K_W1, K_W2 = L.entk_closed_form(model, x, parts=True)
            terms = L.entk_closed_form_deep(model, x, parts=True)
            err = max(rel(sum(terms), K_W1 + K_W2), rel(terms[0], K_W1), rel(terms[1], K_W2))
            report(f"{par}, init_scale {scale:g}", err, 1e-10)


def check_against_autograd(x, models, tag):
    for label, model in models:
        model = model.double()
        K_cf = sum(L.entk_closed_form_deep(model, x.double(), parts=True))
        K_lib = L.entk(model, x.double())
        K_64 = entk_float64(model, x.double())
        report(f"{tag}, {label}, against ntk_lib.entk", rel(K_cf, K_lib), 1e-6)
        report(f"{tag}, {label}, against float64 autograd", rel(K_cf, K_64), 1e-6)


def check_init(x):
    print("Check 2: closed form against autograd at initialisation, 64 MNIST images")
    models = []
    for depth in (2, 3, 4):
        for par in ("deep_standard", "deep_ntk"):
            for scale in (1.0, 9.0):
                torch.manual_seed(0)
                models.append((f"depth {depth}, {par}, init_scale {scale:g}",
                               L.build_model(par, x.shape[1], 200, 10, "relu", scale, depth=depth)))
    check_against_autograd(x, models, "init")


def check_trained(X, y, x):
    print("Check 3: closed form against autograd after 500 SGD steps of a 3 layer network")
    models = []
    for par in ("deep_standard", "deep_ntk"):
        for scale in (1.0, 3.0):
            torch.manual_seed(0)
            model = L.build_model(par, X.shape[1], 200, 10, "relu", scale, depth=3)
            lam = L.gauss_newton_top_eigenvalue(model, X)
            opt = torch.optim.SGD(model.parameters(), lr=1.0 / lam, weight_decay=1e-5 * lam)
            gen = torch.Generator().manual_seed(0)
            for _ in range(500):
                idx = torch.randint(0, len(X), (200,), generator=gen)
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(model(X[idx]), y[idx])
                loss.backward()
                opt.step()
            with torch.no_grad():
                acc = (model(X).argmax(1) == y.argmax(1)).float().mean().item()
            models.append((f"{par}, init_scale {scale:g}, train accuracy {acc:.2f}", model))
    check_against_autograd(x, models, "trained")


def time_kernels(x):
    print("Check 4: cost of one probe kernel of 256 images, 3 layers, width 200, one thread")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = L.build_model("deep_standard", x.shape[1], 200, 10, "relu", 3.0, depth=3)
    for method in ("closed_form", "autograd"):
        L.probe_kernel(model, x, method)
        reps = 10 if method == "closed_form" else 2
        t0 = time.perf_counter()
        for _ in range(reps):
            L.probe_kernel(model, x, method)
        print(f"  {method}: {(time.perf_counter() - t0) / reps:.4f} s per kernel")
    torch.set_num_threads(4)


def main():
    (X, y), (Xte, yte) = load_mnist(n_train=1000, seed=42)
    check_two_layer()
    check_init(X[:64])
    check_trained(X, y, X[:64])
    x256, _, _ = L.select_probe(X, y, Xte, yte, "mixed", 256)
    time_kernels(x256)
    print(f"\n{sum(RESULTS)} of {len(RESULTS)} checks passed. "
          f"{'ALL CHECKS PASS' if all(RESULTS) else 'SOME CHECKS FAIL'}")


if __name__ == "__main__":
    main()
