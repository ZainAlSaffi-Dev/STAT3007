"""
================================================================================
NTK GROKKING DYNAMICS & KERNEL DECOMPOSITION PIPELINE
================================================================================
This script implements empirical Neural Tangent Kernel (eNTK) tracking and 
decomposition for modular addition tasks.

LITERATURE CONTEXT & IMPLEMENTATION ANNOTATIONS:
--------------------------------------------------------------------------------
1. DATASET: Power et al. (2022) vs. Kumar et al. (2024)[cite: 1]
   - Prior Work (Power et al., 2022): Discovered grokking using discrete token 
     sequences on standard Transformer architectures with causal masking[cite: 1].
   - This Implementation (Kumar et al., 2024, App. 8.3): Casts (a + b) mod p directly 
     into concatenated one-hot vectors (dim 2p) and targets (dim p) under MSE loss[cite: 1].
   - Why: Removes confounding sequence-order effects, attention-head routing, and 
     softmax temperature dynamics, isolating pure matrix-algebraic feature learning[cite: 1].

2. ARCHITECTURE: Kumar et al. (2024, App. 8.3) Baseline[cite: 1]
   - Prior Work (Nanda et al., 2023): Analyzed Transformer heads forming Fourier 
     circuits via mechanistic interpretability[cite: 1].
   - This Implementation: Uses a single-hidden-layer MLP (width N=100) with ReLU[cite: 1].
   - Why: Serves as a direct reproduction baseline where the exact finite-width 
     kernel evolution equation can be explicitly analyzed[cite: 1].

3. EMPIRICAL NTK: Jacot et al. (2018) -> Kumar et al. (2024)[cite: 1]
   - Prior Work (Jacot et al., 2018): Formulated the infinite-width asymptotic limit[cite: 1].
   - Prior Work (Kumar et al., 2024): Tracked finite-width eNTK qualitatively via heatmaps[cite: 1].
   - This Implementation: Uses PyTorch's modern functional API (torch.func.jacrev 
     + torch.func.vmap) over a fixed probe set (e.g., 256 samples) for exact online evaluation[cite: 1].

4. KERNEL DECOMPOSITION: Lewkowycz & Gur-Ari (2020) vs. Mohamadi et al. (2024)[cite: 1]
   - The Flaw in Prior Work (Kumar et al., 2024): Reported relative kernel movement 
     ||K_t - K_0||_F / ||K_0||_F as evidence of feature learning driven by weight decay[cite: 1].
   - The Theoretical Contradiction:
     * Lewkowycz & Gur-Ari (2020, Thms 1-2): Proved weight decay kappa under gradient flow 
       causes exponential eigenvalue decay (Theta_t = exp(-2(k-1)*kappa*t)*Theta_0) 
       with zero eigenvector rotation (R_t = 0)[cite: 1]. This is pure scale shrinkage[cite: 1].
     * Mohamadi et al. (2024): Proved no permutation-equivariant static kernel model 
       can solve modular addition without seeing almost all pairs; generalization requires rotation[cite: 1].
   - This Implementation: Decomposes relative movement into scale (S_t) and rotation (R_t):
       ||K_t - K_0||_F^2 / ||K_0||_F^2 = exp(2*S_t) + 1 - 2*exp(S_t)*(1 - R_t)[cite: 1]
     where S_t = ln(||K_t||_F / ||K_0||_F) and R_t = 1 - <\hat{K}_t, \hat{K}_0>_F[cite: 1].

5. CENTERED TASK ALIGNMENT: Cortes et al. (2012) & Atanasov et al. (2022)[cite: 1]
   - Prior Work (Kumar et al., 2024): Computed uncentered alignment <K_t, yy^T>, which 
     drifts when kernel norm ||K_t|| changes, giving false-positive alignment signals[cite: 1].
   - This Implementation: Implements Centered Kernel Alignment (CKA) via projection 
     matrix H = I - (1/n)*11^T[cite: 1].
   - Why (Atanasov et al., 2022): Captures "silent alignment"—the phenomenon where 
     eigenbasis rotates toward task labels early while kernel scale is small[cite: 1].

6. LAZY-TO-RICH CONTROL: Chizat et al. (2019)[cite: 1]
   - Prior Work: Scaled initialization variance (sigma), which changes the initial 
     prediction magnitude and loss, confounding early trajectory comparisons[cite: 1].
   - This Implementation: Uses centered predictor f_tilde = alpha * (f(x, theta) - f(x, theta_0)) 
     and scales lr = eta_0 / alpha^2[cite: 1].
   - Why: Guarantees zero initial training loss across all alpha, matching function-space 
     dynamics at step 0 while tuning feature learning (alpha -> inf is lazy; alpha -> 0 is rich)[cite: 1].
================================================================================
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
    Generates the modular arithmetic dataset for (a + b) mod p[cite: 1].

    LITERATURE COMPARISON:
    - Prior Implementation (Power et al., 2022): Used autoregressive token sequences 
      fed into Transformers, introducing token embedding and positional encoding dynamics[cite: 1].
    - This Implementation (Kumar et al., 2024, App. 8.3): Uses static one-hot encoding[cite: 1]:
      * Inputs x = [one_hot(a); one_hot(b)] in R^{2p}[cite: 1].
      * Targets y = one_hot((a + b) mod p) in R^{p}[cite: 1].
      * Loss = Mean Squared Error (MSE) over the p logits[cite: 1].
    - Correction/Benefit: Eliminates autoregressive confounders and allows exact 
      algebraic tracking of kernel changes on a finite sample space of p^2 pairs[cite: 1].
    """
    torch.manual_seed(seed)
    pairs = [(a, b) for a in range(p) for b in range(p)]
    num_samples = len(pairs)
    
    # Input dimension: 2 * p (concatenated one-hot vectors)[cite: 1]
    # Output dimension: p (one-hot vector for the remainder)[cite: 1]
    X = torch.zeros((num_samples, 2 * p), dtype=torch.float32)
    y = torch.zeros((num_samples, p), dtype=torch.float32)
    
    for idx, (a, b) in enumerate(pairs):
        X[idx, a] = 1.0
        X[idx, p + b] = 1.0
        target = (a + b) % p
        y[idx, target] = 1.0

    # Train / Test random split[cite: 1]
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
    One-hidden-layer Multi-Layer Perceptron (MLP) with ReLU activation[cite: 1].

    LITERATURE COMPARISON:
    - Prior Implementation: Nanda et al. (2023) studied grokking using 1-layer 
      attention mechanisms with softmax layers[cite: 1].
    - This Implementation: Directly follows Kumar et al. (2024, App. 8.3)[cite: 1]:
        f(x; W1, W2) = W2 * ReLU(W1 * x + b1)
      where W1 in R^{N x 2p}, b1 in R^{N}, W2 in R^{p x N}, with default width N=100[cite: 1].
    - Correction/Benefit: By keeping the network shallow and without output bias, the 
      empirical NTK corresponds directly to theoretical finite-width tensor derivations 
      (e.g., Lewkowycz & Gur-Ari, 2020)[cite: 1].
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False)  # Output bias omitted to maintain homogeneity[cite: 1]
        )

    def forward(self, x):
        return self.net(x)


# =====================================================================
# SECTION 3: EMPIRICAL NTK (eNTK) COMPUTATION
# =====================================================================
def compute_entk(model, params, x_probe, mode="trace"):
    """
    Computes the empirical Neural Tangent Kernel (eNTK) Gram matrix over a probe set[cite: 1].

    MATHEMATICAL DEFINITION:
    K(x_i, x_j) = sum_{k} (d f(x_i) / d theta_k) * (d f(x_j) / d theta_k)^T[cite: 1]

    LITERATURE COMPARISON:
    - Prior Implementation: Jacot et al. (2018) only calculated the infinite-width 
      asymptotic analytical kernel Theta_inf[cite: 1]. Kumar et al. (2024) computed eNTK 
      empirically but focused on qualitative heatmaps of logit correlations[cite: 1].
    - This Implementation: Uses PyTorch's torch.func API (jacrev + vmap) for batched 
      reverse-mode automatic differentiation[cite: 1].
    - Multi-Logit Handling:
      * mode="trace" (Sum-over-logits): Sums outer products across all output logits 
        c in {0, ..., p-1}[cite: 1]. Yields an (N_probe x N_probe) Gram matrix[cite: 1].
      * mode="first_logit": Extracts gradient inner product for c=0 only, providing 
        a sanity check against logit-correlation coupling[cite: 1].
    """
    # Define a single-sample forward pass taking parameter dictionary explicitly
    def fnet_single(p_dict, x_single):
        return functional_call(model, p_dict, x_single.unsqueeze(0)).squeeze(0)

    # Compute Jacobians: shape (N_probe, C_out, *param_shape)
    # vmap parallelizes jacrev across probe inputs along dimension 0[cite: 1]
    jac_dict = vmap(jacrev(fnet_single), (None, 0))(params, x_probe)
    
    # Flatten parameter dimensions: (N_probe, C_out, P_layer)
    flat_jacs = [j.flatten(start_dim=2) for j in jac_dict.values()]
    
    # Concatenate all model parameters along feature dimension: (N_probe, C_out, P_total)
    J = torch.cat(flat_jacs, dim=2)
    
    if mode == "trace":
        # Sum-over-logits: \sum_c J_{a, c, :} \cdot J_{b, c, :}[cite: 1]
        K = torch.einsum("acp, bcp -> ab", J, J)
    elif mode == "first_logit":
        # First logit: J_{a, 0, :} \cdot J_{b, 0, :}[cite: 1]
        K = torch.einsum("ap, bp -> ab", J[:, 0, :], J[:, 0, :])
    else:
        raise ValueError(f"Unsupported mode: {mode}")
        
    return K


# =====================================================================
# SECTION 4: SCALE (S_t) AND ROTATION (R_t) DECOMPOSITION
# =====================================================================
def compute_scale_and_rotation(K_0, K_t):
    """
    Decouples empirical NTK movement into Scale (S_t) and Rotation (R_t)[cite: 1].

    LITERATURE COMPARISON & THEORETICAL MOTIVATION:
    - Prior Literature Flaw (Kumar et al., 2024): Tracked unnormalized relative movement:
        Relative Movement = ||K_t - K_0||_F / ||K_0||_F[cite: 1]
      Kumar et al. attributed speed-ups from weight decay to "forcing changes in the NTK," 
      citing Lewkowycz & Gur-Ari (2020)[cite: 1].
    - The Confound:
      Under L2 regularization, an infinitely wide network exhibits pure eigenvalue decay:
        Theta_t = exp(-2(k-1)*kappa*t) * Theta_0  =>  Eigenbasis does NOT rotate (R_t = 0)[cite: 1].
      Meanwhile, Mohamadi et al. (2024) proved that solving modular addition requires true 
      eigenbasis rotation; scale decay alone provably cannot generalize[cite: 1].
    - This Implementation:
      Separates the two mechanisms via the exact identity[cite: 1]:
        ||K_t - K_0||_F^2 / ||K_0||_F^2 = exp(2*S_t) + 1 - 2*exp(S_t)*(1 - R_t)[cite: 1]
      where:
        S_t = log(||K_t||_F / ||K_0||_F)               [Pure norm shrinkage/expansion][cite: 1]
        R_t = 1 - <\hat{K}_t, \hat{K}_0>_F             [Pure subspace rotation][cite: 1]
        \hat{K} = K / ||K||_F                          [Normalized Gram matrix][cite: 1]
    """
    norm_0 = torch.linalg.norm(K_0, ord="fro")
    norm_t = torch.linalg.norm(K_t, ord="fro")
    
    # 1. Scale Movement: S_t[cite: 1]
    # S_t < 0 indicates kernel shrinking; S_t > 0 indicates kernel growing[cite: 1].
    S_t = torch.log(norm_t / norm_0)
    
    # 2. Normalized Kernel Gram Matrices[cite: 1]
    K_0_hat = K_0 / norm_0
    K_t_hat = K_t / norm_t
    
    # 3. Subspace Rotation: R_t[cite: 1]
    # Frobenius inner product: <A, B>_F = sum(A * B)[cite: 1]
    # If eigenvectors do not rotate, overlap = 1.0 => R_t = 0.0[cite: 1].
    overlap = torch.sum(K_t_hat * K_0_hat)
    R_t = 1.0 - overlap
    
    return S_t.item(), R_t.item()


# =====================================================================
# SECTION 5: CENTERED TASK ALIGNMENT (A_t)
# =====================================================================
def compute_task_alignment(K_t, y_probe):
    """
    Computes Centered Kernel Alignment (CKA) between eNTK and probe targets[cite: 1].

    LITERATURE COMPARISON:
    - Prior Implementation (Kumar et al., 2024): Used uncentered alignment <K_t, yy^T>[cite: 1].
      Problem: Shifts in overall kernel scale or mean activation shift the alignment 
      metric without any meaningful rotation toward target task representations[cite: 1].
    - This Implementation: Uses Cortes et al. (2012) Centered Kernel Alignment[cite: 1]:
        A_t = < H K_t H, H (y y^T) H >_F / ( ||H K_t H||_F * ||H (y y^T) H||_F )[cite: 1]
      where H = I - (1/n)*11^T is the centering projection operator[cite: 1].
    - Connection to Atanasov et al. (2022) ("Silent Alignment"):
      Atanasov et al. proved that task-relevant representations align during early 
      training phases while loss remains flat[cite: 1]. Properly centered A_t provides a 
      scale-invariant progress measure that tracks this silent feature emergence[cite: 1].
    """
    n = K_t.shape[0]
    device = K_t.device
    
    # Centering projection matrix H = I - (1/n) * 1 * 1^T[cite: 1]
    I = torch.eye(n, device=device)
    ones = torch.ones((n, n), device=device)
    H = I - (1.0 / n) * ones
    
    # Project kernel into centered subspace: HKH[cite: 1]
    HKH = H @ K_t @ H
    norm_HKH = torch.linalg.norm(HKH, ord="fro")
    
    # Project target similarity matrix: H(y y^T)H[cite: 1]
    YY = y_probe @ y_probe.T
    HYH = H @ YY @ H
    norm_HYH = torch.linalg.norm(HYH, ord="fro")
    
    # Guard against division by zero if weights degenerate[cite: 1]
    if norm_HKH == 0 or norm_HYH == 0:
        return 0.0
        
    # Scale-invariant normalized Frobenius inner product[cite: 1]
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
    Executes full-batch gradient descent with lazy/rich control and metric tracking[cite: 1].

    LITERATURE COMPARISON & METHODOLOGICAL CONTROLS:
    1. Centered Rescaled Predictor (Chizat et al., 2019)[cite: 1]:
         f_tilde(x, theta) = alpha * (f(x, theta) - f(x, theta_0))[cite: 1]
       * Subtracting f(x, theta_0) ensures initial train loss is identical for all alpha[cite: 1].
       * Setting learning rate lr = eta_0 / alpha^2 matches early continuous dynamics[cite: 1].
       * Large alpha enforces the linearized lazy NTK regime (parameter movement ~ O(1/alpha))[cite: 1].
       * Small alpha forces the network into the rich feature-learning regime[cite: 1].

    2. Dimensionless Weight Decay Parameterization (Lewkowycz & Gur-Ari, 2020)[cite: 1]:
       Discrete gradient descent update with weight decay kappa:
         w_{t+1} = (1 - eta * kappa) * w_t - eta * grad[cite: 1]
       * Sweeping the dimensionless product (eta * kappa) ensures discrete numerical 
         stability (requires eta * kappa < 2)[cite: 1].
       * Aligns directly with the theoretical decay timescale t_* ~ 1 / (eta * kappa)[cite: 1].
       * PyTorch optimizer handles weight decay as: w_{t+1} = w_t - lr * grad - lr * wd * w_t.
         Therefore, we set wd = (eta * kappa) / lr[cite: 1].
    """
    # 1. Prepare Modular Addition Dataset[cite: 1]
    (X_train, y_train), (X_test, y_test) = make_modular_addition_dataset(p=p, train_fraction=0.9)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    
    # Fix a persistent probe set for eNTK tracking throughout training[cite: 1]
    probe_size = min(probe_size, X_train.shape[0])
    x_probe = X_train[:probe_size]
    y_probe = y_train[:probe_size]
    
    # 2. Instantiate Model and Freeze Reference Model for Centering (Chizat et al., 2019)[cite: 1]
    model = ModularMLP(input_dim=2 * p, hidden_dim=hidden_dim, output_dim=p).to(device)
    initial_state = copy.deepcopy(model.state_dict())
    
    # Reference snapshot model representing theta_0 (frozen)[cite: 1]
    model_0 = ModularMLP(input_dim=2 * p, hidden_dim=hidden_dim, output_dim=p).to(device)
    model_0.load_state_dict(initial_state)
    for p_ref in model_0.parameters():
        p_ref.requires_grad = False
        
    # 3. Learning Rate and Regularization Setup[cite: 1]
    lr = eta_0 / (alpha ** 2)
    # Match PyTorch weight_decay to dimensionless eta * kappa:
    # PyTorch: w <- w - lr * (grad + wd * w) = (1 - lr * wd) * w - lr * grad
    # To obtain (1 - eta * kappa), we set wd = (eta * kappa) / lr[cite: 1]
    weight_decay = eta_kappa / lr if lr > 0 else 0.0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    
    # 4. Snapshot Initial Kernel K_0 on the Probe Set[cite: 1]
    curr_params = dict(model.named_parameters())
    K_0 = compute_entk(model, curr_params, x_probe, mode="trace").detach()
    
    # Metric tracking container
    history = {
        "step": [],
        "train_loss": [],
        "test_loss": [],
        "S_t": [],          # Kernel Scale[cite: 1]
        "R_t": [],          # Kernel Rotation[cite: 1]
        "A_t": [],          # Task Alignment[cite: 1]
        "param_dist": []    # Relative weight displacement ||w_t - w_0|| / ||w_0||[cite: 1]
    }
    
    # Flatten initial weights for displacement calculation[cite: 1]
    initial_flat_params = torch.cat([p.flatten() for p in model.parameters()]).detach()
    norm_w0 = torch.linalg.norm(initial_flat_params)
    
    print(f"--- Training: p={p}, N={hidden_dim}, alpha={alpha}, eta*kappa={eta_kappa} ---")
    
    # 5. Full-Batch Gradient Descent Training Loop[cite: 1]
    for step in range(steps + 1):
        model.train()
        optimizer.zero_grad()
        
        # Centered predictor: f_tilde(x, theta) = alpha * (f(x, theta) - f(x, theta_0))[cite: 1]
        train_pred = alpha * (model(X_train) - model_0(X_train))
        loss = loss_fn(train_pred, y_train)
        loss.backward()
        optimizer.step()
        
        # Periodic Evaluation of eNTK Properties[cite: 1]
        if step % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                # Generalization test loss[cite: 1]
                test_pred = alpha * (model(X_test) - model_0(X_test))
                t_loss = loss_fn(test_pred, y_test).item()
                
                # Relative parameter movement ||w_t - w_0|| / ||w_0|| (Chizat et al., 2019)[cite: 1]
                curr_flat_params = torch.cat([p.flatten() for p in model.parameters()])
                param_dist = (torch.linalg.norm(curr_flat_params - initial_flat_params) / norm_w0).item()
            
            # Compute current empirical NTK K_t on the fixed probe set[cite: 1]
            p_dict = dict(model.named_parameters())
            K_t = compute_entk(model, p_dict, x_probe, mode="trace").detach()
            
            # Decompose movement: Scale S_t vs. Rotation R_t[cite: 1]
            S_t, R_t = compute_scale_and_rotation(K_0, K_t)
            
            # Compute Centered Kernel Alignment A_t[cite: 1]
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
    # Runs a baseline verification reproducing Tier 0 configuration[cite: 1]
    results = train_modular_addition(
        p=23,
        hidden_dim=100,
        alpha=1.0,
        eta_0=100.0,
        eta_kappa=0.01,
        steps=1000,
        eval_interval=100
    )