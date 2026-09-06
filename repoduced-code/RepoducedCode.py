"""
Empirical neural tangent kernel tracking for grokking on modular addition.

This script reproduces the Tier 0 baseline of the G2 proposal and records the
kernel scale, rotation, and centred alignment measures that the proposal
defines.

Every claim below that comes from a paper names the paper, the arXiv
identifier, and where possible the section, theorem, or equation. Each of
these was checked against the arXiv text on 4 September 2026. Version numbers
are given because equation numbers can change between versions.

- Power et al. (2022), arXiv:2201.02177. Grokking was first reported on
  binary operation tables written as token sequences of the form "a o b = c"
  and learned by a decoder-only transformer with causal masking. See their
  Section 2 and Appendix A.1.
- Nanda et al. (2023), arXiv:2301.05217. A one-layer transformer trained on
  modular addition with weight decay is reverse engineered. See their
  Section 2.
- Jacot, Gabriel, and Hongler (2018), arXiv:1806.07572. The neural tangent
  kernel is defined and shown to be constant during training in the infinite
  width limit.
- Kumar et al. (2024), arXiv:2310.06110v3. Appendix 8.3 gives the modular
  addition baseline used here. Appendix 8.1, Equation 7, gives the centred
  and rescaled predictor and the learning rate scaling. Section 5 gives their
  kernel alignment measure. Appendix 13 gives their weight decay argument.
- Chizat, Oyallon, and Bach (2019), arXiv:1812.07956v5. Section 2.1,
  Equation 2, defines the rescaled objective. Theorem 2.2 bounds parameter
  movement by O(1/alpha) when the initial output is zero.
- Lewkowycz and Gur-Ari (2020), arXiv:2006.08643v2. Theorems 1 and 2 give
  the kernel evolution under L2 regularisation at infinite width. Equations
  S5 and S6 give the exact finite width evolution.
- Mohamadi et al. (2024), arXiv:2407.12332v1. Theorem 3.4 is the lower bound
  for permutation-equivariant kernel methods on modular addition.
- Cortes, Mohri, and Rostamizadeh (2012), Journal of Machine Learning
  Research 13, pages 795 to 828. Lemma 1 gives the centred kernel matrix and
  Definition 4 gives centred alignment.
- Atanasov, Bordelon, and Pehlevan (2022), arXiv:2111.00034, ICLR 2022. The
  silent alignment effect. This paper is not in ref.bib.

Where this file departs from a source, the docstring of the relevant
function says so.
"""

import copy
import math
import torch
import torch.nn as nn
from torch.func import functional_call, vmap, jacrev


# =====================================================================
# SECTION 1: DATASET GENERATION
# =====================================================================
def make_modular_addition_dataset(p=23, train_fraction=0.9, seed=42):
    """
    Build the modular addition dataset for (a + b) mod p.

    The encoding follows Kumar et al. (2024), Appendix 8.3
    (arXiv:2310.06110v3). The input is the concatenation of the one-hot codes
    of a and b, so it has dimension 2p. The target is the one-hot code of
    (a + b) mod p, so it has dimension p. The loss is mean squared error on
    the p outputs. Kumar et al. use 90 percent of the p^2 pairs for training
    and the rest for testing.

    Power et al. (2022), Section 2 and Appendix A.1 (arXiv:2201.02177),
    instead wrote each equation as the token sequence "a o b = c" and trained
    a decoder-only transformer with causal masking. The one-hot encoding used
    here removes the token embedding and attention dynamics, so the kernel
    can be tracked on a finite set of p^2 inputs.
    """
    torch.manual_seed(seed)
    pairs = [(a, b) for a in range(p) for b in range(p)]
    num_samples = len(pairs)
    
    # The input dimension is 2p and the output dimension is p, as in
    # Kumar et al. (2024), Appendix 8.3.
    X = torch.zeros((num_samples, 2 * p), dtype=torch.float32)
    y = torch.zeros((num_samples, p), dtype=torch.float32)
    
    for idx, (a, b) in enumerate(pairs):
        X[idx, a] = 1.0
        X[idx, p + b] = 1.0
        target = (a + b) % p
        y[idx, target] = 1.0

    # Random split into training and test pairs. Kumar et al. (2024),
    # Appendix 8.3, use 90 percent of the pairs for training.
    indices = torch.randperm(num_samples)
    train_size = int(num_samples * train_fraction)
    
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])


# =====================================================================
# SECTION 2: MODEL ARCHITECTURE
# =====================================================================
class ModularMLP(nn.Module):
    """
    One-hidden-layer perceptron with ReLU activation.

    The architecture follows Kumar et al. (2024), Appendix 8.3
    (arXiv:2310.06110v3). It has a single hidden layer of width N = 100,
    input dimension 2p, and output dimension p, and it is trained with
    vanilla gradient descent. Kumar et al. do not state whether biases are
    used.

    Nanda et al. (2023), Section 2 (arXiv:2301.05217), used a one-layer
    transformer with attention on the same task. The perceptron here has no
    attention and no softmax, so its kernel is simpler to analyse.

    The output layer has no bias. The hidden layer keeps its bias, so the
    network is not exactly homogeneous in its parameters. The kernel decay
    result of Lewkowycz and Gur-Ari (2020), Theorems 1 and 2
    (arXiv:2006.08643v2), is stated for k-homogeneous networks, and the
    proposal notes that a bias-free ReLU network has k equal to its depth.
    Anyone comparing a run against that theorem should set the hidden bias
    to False as well.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            # No output bias. See the class docstring for what this does to
            # homogeneity.
            nn.Linear(hidden_dim, output_dim, bias=False)
        )

    def forward(self, x):
        return self.net(x)


# =====================================================================
# SECTION 3: EMPIRICAL NTK (eNTK) COMPUTATION
# =====================================================================
def compute_entk(model, params, x_probe, mode="trace"):
    """
    Compute the empirical neural tangent kernel Gram matrix on a probe set.

    The kernel is K(x_i, x_j) = sum over parameters of the product of the
    output gradients at x_i and x_j. This is the neural tangent kernel of
    Jacot, Gabriel, and Hongler (2018) (arXiv:1806.07572) evaluated at the
    current finite-width parameters, rather than in the infinite-width limit
    where they show it stays constant during training. Mohamadi et al.
    (2024), Section 3.1 (arXiv:2407.12332v1), write the same empirical kernel
    as K_theta(x, x') = grad g(theta; x) grad g(theta; x')^T.

    Kumar et al. (2024) do not report the kernel matrix itself. They report
    the alignment of the initial kernel with the labels in their Section 5
    and the relative parameter movement in their Appendix 8.2.

    The Jacobians are computed with torch.func.jacrev inside torch.func.vmap.

    With p outputs the Jacobian has shape (N_probe, p, P). In mode "trace"
    the kernel sums the gradient inner products over all p outputs, which is
    the sum-over-logits kernel that the proposal specifies. In mode
    "first_logit" only output 0 is used, as a robustness check. Neither
    choice is taken from a paper.
    """
    # Define a single-sample forward pass taking parameter dictionary explicitly
    def fnet_single(p_dict, x_single):
        return functional_call(model, p_dict, x_single.unsqueeze(0)).squeeze(0)

    # Compute Jacobians: shape (N_probe, C_out, *param_shape)
    # vmap runs jacrev over the probe inputs along dimension 0.
    jac_dict = vmap(jacrev(fnet_single), (None, 0))(params, x_probe)
    
    # Flatten parameter dimensions: (N_probe, C_out, P_layer)
    flat_jacs = [j.flatten(start_dim=2) for j in jac_dict.values()]
    
    # Concatenate all model parameters along feature dimension: (N_probe, C_out, P_total)
    J = torch.cat(flat_jacs, dim=2)
    
    if mode == "trace":
        # Sum the gradient inner products over all p outputs.
        K = torch.einsum("acp, bcp -> ab", J, J)
    elif mode == "first_logit":
        # Use output 0 only.
        K = torch.einsum("ap, bp -> ab", J[:, 0, :], J[:, 0, :])
    else:
        raise ValueError(f"Unsupported mode: {mode}")
        
    return K


# =====================================================================
# SECTION 4: SCALE (S_t) AND ROTATION (R_t) DECOMPOSITION
# =====================================================================
def compute_scale_and_rotation(K_0, K_t):
    """
    Split the movement of the empirical kernel into a scale term and a rotation term.

    The usual one-number summary of kernel change is the relative movement
    ||K_t - K_0||_F / ||K_0||_F. It cannot tell a kernel that shrinks from
    one that turns. The proposal defines the scale term
    S_t = log(||K_t||_F / ||K_0||_F) and the rotation term
    R_t = 1 - <K_t / ||K_t||_F, K_0 / ||K_0||_F>_F, and expands the squared
    relative movement as exp(2 S_t) + 1 - 2 exp(S_t) (1 - R_t). This
    identity is the group's own algebra, Equation 1 of the proposal. It is
    not a published result.

    The split matters for the following reason. Lewkowycz and Gur-Ari
    (2020), Theorem 1, Equation 2, and Theorem 2 (arXiv:2006.08643v2), show
    that an infinitely wide k-homogeneous network trained by gradient flow on
    mean squared error with L2 coefficient lambda has a kernel that obeys
    d Theta_t / dt = -2 (k - 1) lambda Theta_t. Its eigenvalues decay as
    exp(-2 (k - 1) lambda t) while its eigenvectors stay fixed. Theorem 1
    assumes the correlation function conjecture of Dyer and Gur-Ari, and
    Theorem 2 assumes k is at least 2. In the terms used here that is
    S_t = -2 (k - 1) lambda t and R_t = 0. Kumar et al. (2024), Appendix 13
    and the conclusion (arXiv:2310.06110v3), cite this result to argue that
    weight decay pushes the network out of the lazy regime by "forcing
    changes to the NTK", and they call the argument speculative. Mohamadi
    et al. (2024), Theorem 3.4 (arXiv:2407.12332v1), show that no
    permutation-equivariant kernel method can beat the trivial predictor on
    modular addition unless it trains on a constant fraction of all possible
    data points. So a kernel that only shrinks cannot explain generalisation
    at low training fractions, and the rotation term is the quantity to
    watch.
    """
    norm_0 = torch.linalg.norm(K_0, ord="fro")
    norm_t = torch.linalg.norm(K_t, ord="fro")
    
    # Scale term. A negative value means the kernel shrank and a positive
    # value means it grew.
    S_t = torch.log(norm_t / norm_0)
    
    # Kernels normalised to unit Frobenius norm.
    K_0_hat = K_0 / norm_0
    K_t_hat = K_t / norm_t
    
    # Rotation term. The Frobenius inner product of the normalised kernels
    # is 1 when nothing has rotated, which gives R_t = 0.
    overlap = torch.sum(K_t_hat * K_0_hat)
    R_t = 1.0 - overlap
    
    return S_t.item(), R_t.item()


# =====================================================================
# SECTION 5: CENTERED TASK ALIGNMENT (A_t)
# =====================================================================
def compute_task_alignment(K_t, y_probe):
    """
    Compute the centred alignment between the empirical kernel and the probe labels.

    The measure follows Cortes, Mohri, and Rostamizadeh (2012), Journal of
    Machine Learning Research 13, pages 795 to 828. Their Lemma 1 writes the
    centred kernel matrix as K_c = H K H with H = I - (1/n) 1 1^T, and their
    Definition 4 defines the alignment of two kernel matrices as
    <K_c, K'_c>_F / (||K_c||_F ||K'_c||_F). Here K' is the label kernel
    y y^T built from the one-hot probe labels. Rescaling K_t leaves A_t
    unchanged, so A_t moves only when the kernel rotates.

    Kumar et al. (2024), Section 5 (arXiv:2310.06110v3), use
    y^T K_0 y / (||K_0||_F ||y||^2) on the initial kernel and call it centred
    kernel alignment, citing Cortes et al. among others, but the formula as
    printed has no centring matrix. The version here applies the centring of
    Cortes et al. and is tracked through training rather than at
    initialisation only.

    Atanasov, Bordelon, and Pehlevan (2022), arXiv:2111.00034, describe a
    silent alignment effect in which the tangent kernel changes in
    eigenstructure while it is small and before the loss falls, and grows
    only in scale afterwards. Because A_t ignores scale, it can show that
    early change.
    """
    n = K_t.shape[0]
    device = K_t.device
    
    # Centring matrix H = I - (1/n) 1 1^T from Cortes et al. (2012), Lemma 1.
    I = torch.eye(n, device=device)
    ones = torch.ones((n, n), device=device)
    H = I - (1.0 / n) * ones
    
    # Centred kernel H K H.
    HKH = H @ K_t @ H
    norm_HKH = torch.linalg.norm(HKH, ord="fro")
    
    # Centred label kernel H y y^T H.
    YY = y_probe @ y_probe.T
    HYH = H @ YY @ H
    norm_HYH = torch.linalg.norm(HYH, ord="fro")
    
    # Return zero if either centred matrix vanishes, to avoid dividing by zero.
    if norm_HKH == 0 or norm_HYH == 0:
        return 0.0
        
    # Definition 4 of Cortes et al. (2012).
    alignment = torch.sum(HKH * HYH) / (norm_HKH * norm_HYH)
    return alignment.item()


# =====================================================================
# SECTION 6: TRAINING PIPELINE WITH ALPHA RESCALING & WEIGHT DECAY
# =====================================================================
def train_modular_addition(
    p=23,
    hidden_dim=100,
    alpha=1.0,
    eta_0=100.0,
    eta_kappa=0.01,
    steps=10000,
    probe_size=256,
    eval_interval=100,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Run full-batch gradient descent and record the kernel measures at fixed intervals.

    Laziness control. The predictor is the centred and rescaled function
    f_tilde(x, theta) = alpha (f(x, theta) - f(x, theta_0)) with learning
    rate eta_0 / alpha^2. This is Equation 7 of Kumar et al. (2024),
    Appendix 8.1 (arXiv:2310.06110v3), who attribute the idea to Chizat,
    Oyallon, and Bach (2019). Chizat et al., Section 2.1, Equation 2
    (arXiv:1812.07956v5), scale the objective by 1/alpha^2 and note that
    this gives the proper time parameterisation for large alpha. Their
    Theorem 2.2 shows that when the initial output is zero the parameters
    move by O(1/alpha) over a fixed time horizon, so large alpha is lazy and
    small alpha is rich. Subtracting f(x, theta_0) makes the initial output
    zero, which Chizat et al. also do for their convolutional experiments in
    Section 3.2, and it keeps the initial training loss the same for every
    alpha. Kumar et al., Appendix 8.2, show that the initialisation scale
    sigma and alpha act on the dynamics through the product sigma^2 alpha,
    so only alpha is swept here.

    Weight decay. The update with coupled L2 regularisation is
    w_{t+1} = (1 - eta kappa) w_t - eta grad, which is the form Lewkowycz and
    Gur-Ari (2020) write in Equation S11 for a deep linear network
    (arXiv:2006.08643v2). The sweep is over the product eta kappa. Lewkowycz
    and Gur-Ari note after their Theorem 1 that the function and kernel decay
    on a time scale proportional to 1/lambda in gradient flow time, which is
    about 1/(eta kappa) steps here. The bound eta kappa < 2 is not from a
    paper. It is the condition for the map w to (1 - eta kappa) w to be a
    contraction. PyTorch SGD applies decay as
    w_{t+1} = w_t - lr (grad + wd w_t), so wd = eta kappa / lr reproduces the
    update above.

    Departure from the baseline. Kumar et al. (2024), Appendix 8.3, state
    that they use no weight decay. The default eta_kappa here is 0.01, so a
    Tier 0 reproduction must pass eta_kappa=0.

    The recorded parameter distance ||w_t - w_0|| / ||w_0|| is the relative
    parameter change that Kumar et al. (2024), Appendix 8.2, use to measure
    how far training is from the lazy regime.
    """
    # 1. Build the modular addition dataset.
    (X_train, y_train), (X_test, y_test) = make_modular_addition_dataset(p=p, train_fraction=0.9)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    
    # Fix one probe set for the whole run so that K_0 and K_t are comparable.
    probe_size = min(probe_size, X_train.shape[0])
    x_probe = X_train[:probe_size]
    y_probe = y_train[:probe_size]
    
    # 2. Build the model and a frozen copy at theta_0 for the centred
    # predictor of Kumar et al. (2024), Appendix 8.1, Equation 7.
    model = ModularMLP(input_dim=2 * p, hidden_dim=hidden_dim, output_dim=p).to(device)
    initial_state = copy.deepcopy(model.state_dict())
    
    # Frozen reference model holding theta_0.
    model_0 = ModularMLP(input_dim=2 * p, hidden_dim=hidden_dim, output_dim=p).to(device)
    model_0.load_state_dict(initial_state)
    for p_ref in model_0.parameters():
        p_ref.requires_grad = False
        
    # 3. Learning rate and weight decay. The rate eta_0 / alpha^2 follows
    # Kumar et al. (2024), Appendix 8.1.
    lr = eta_0 / (alpha ** 2)
    # PyTorch applies w <- w - lr * (grad + wd * w) = (1 - lr * wd) * w - lr * grad.
    # Setting wd = (eta * kappa) / lr gives the factor (1 - eta * kappa).
    weight_decay = eta_kappa / lr if lr > 0 else 0.0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    
    # 4. Record the initial kernel K_0 on the probe set.
    curr_params = dict(model.named_parameters())
    K_0 = compute_entk(model, curr_params, x_probe, mode="trace").detach()
    
    # Metric tracking container
    history = {
        "step": [],
        "train_loss": [],
        "test_loss": [],
        "S_t": [],          # Kernel scale term from the proposal.
        "R_t": [],          # Kernel rotation term from the proposal.
        "A_t": [],          # Centred alignment, Cortes et al. (2012), Definition 4.
        "param_dist": []    # Relative parameter change, Kumar et al. (2024), Appendix 8.2.
    }
    
    # Flatten the initial weights for the parameter distance.
    initial_flat_params = torch.cat([p.flatten() for p in model.parameters()]).detach()
    norm_w0 = torch.linalg.norm(initial_flat_params)
    
    print(f"--- Training: p={p}, N={hidden_dim}, alpha={alpha}, eta*kappa={eta_kappa} ---")
    
    # 5. Full-batch gradient descent, as in Kumar et al. (2024), Appendix 8.3.
    for step in range(steps + 1):
        model.train()
        optimizer.zero_grad()
        
        # Centred and rescaled predictor, Kumar et al. (2024), Equation 7.
        train_pred = alpha * (model(X_train) - model_0(X_train))
        loss = loss_fn(train_pred, y_train)
        loss.backward()
        optimizer.step()
        
        # Evaluate the kernel measures every eval_interval steps.
        if step % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                # Test loss under the same centred predictor.
                test_pred = alpha * (model(X_test) - model_0(X_test))
                t_loss = loss_fn(test_pred, y_test).item()
                
                # Relative parameter change, Kumar et al. (2024), Appendix 8.2.
                curr_flat_params = torch.cat([p.flatten() for p in model.parameters()])
                param_dist = (torch.linalg.norm(curr_flat_params - initial_flat_params) / norm_w0).item()
            
            # Current empirical kernel K_t on the fixed probe set.
            p_dict = dict(model.named_parameters())
            K_t = compute_entk(model, p_dict, x_probe, mode="trace").detach()
            
            # Scale and rotation terms from the proposal.
            S_t, R_t = compute_scale_and_rotation(K_0, K_t)
            
            # Centred alignment, Cortes et al. (2012), Definition 4.
            A_t = compute_task_alignment(K_t, y_probe)
            
            # Store recorded metrics
            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["test_loss"].append(t_loss)
            history["S_t"].append(S_t)
            history["R_t"].append(R_t)
            history["A_t"].append(A_t)
            history["param_dist"].append(param_dist)
            
            if step % (eval_interval * 5) == 0:
                print(
                    f"Step {step:5d} | Train Loss: {loss.item():.5f} | Test Loss: {t_loss:.5f} | "
                    f"Scale S_t: {S_t:+.3f} | Rotation R_t: {R_t:.4f} | Alignment A_t: {A_t:.4f} | "
                    f"||dw||/||w0||: {param_dist:.3f}"
                )
                
    return history


# =====================================================================
# VERIFICATION RUN
# =====================================================================
if __name__ == "__main__":
    # Short run in the Tier 0 configuration of Kumar et al. (2024),
    # Appendix 8.3, except that eta_kappa=0.01 adds weight decay, which
    # Kumar et al. do not use. Pass eta_kappa=0 for the exact baseline.
    results = train_modular_addition(
        p=23,
        hidden_dim=100,
        alpha=1.0,
        eta_0=100.0,
        eta_kappa=0.01,
        steps=1000,
        eval_interval=100
    )
