import math

import numpy as np
import torch
import warnings
from torch.func import jacrev, vmap

def spectral_radius(M):
    """Compute the spectral radius (largest eigenvalue magnitude) of a matrix.

    Args:
        M: (n, n) torch.Tensor

    Returns:
        rho: float
    """
    eigvals = torch.linalg.eigvals(M)
    return eigvals.abs().max().item()


def transient_constant(M):
    """Compute the transient constant C = ||V|| * ||V^{-1}|| where V is the
    eigenvector matrix of M.

    Args:
        M: (n, n) torch.Tensor

    Returns:
        C: float
    """
    eigenvalues, V = torch.linalg.eig(M)
    cond = torch.linalg.cond(V, p=2).item()
    if cond > 1e10:
        warnings.warn("V is poorly conditioned, M may not be diagonalizable")
    return cond

def compute_encoder_lipschitz(encoder, training_data):
    """Compute lower and upper Lipschitz constants of the encoder.

    Returns:
        m: float, lower Lipschitz (min singular value across all data)
        L: float, upper Lipschitz (max singular value across all data)
    """
    X = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in training_data])
    def encode_single(x):
        return encoder(x.unsqueeze(0)).squeeze(0)
    J_batch = vmap(jacrev(encode_single))(X)
    svdvals = torch.linalg.svdvals(J_batch)  # (N, min(latent, state))
    sigma_mins = svdvals[:, -1]
    sigma_maxs = svdvals[:, 0]
    m = float(sigma_mins.min().detach())
    L = float(sigma_maxs.max().detach())
    # Print distribution for diagnostics
    print(f"  σ_min distribution ({len(sigma_mins)} points): "
          f"min={sigma_mins.min():.6f}  p1={sigma_mins.quantile(0.01):.6f}  "
          f"p5={sigma_mins.quantile(0.05):.6f}  median={sigma_mins.median():.6f}  "
          f"mean={sigma_mins.mean():.6f}")
    return m, L

"""
def compute_lower_lipschitz(encoder, training_data):
    #Compute the lower Lipschitz constant of the encoder via minimum singular
    #value of the Jacobian across training data.

    #Args:
    #    encoder: callable mapping (state_dim,) -> (latent_dim,)
    #    training_data: iterable of state vectors (numpy arrays or tensors)

    #Returns:
    #    m: float, the lower Lipschitz constant
    
    X = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in training_data])
    J_batch = vmap(jacrev(encoder))(X)  # (N, latent_dim, state_dim)
    sigma_mins = torch.linalg.svdvals(J_batch)[:, -1]
    return float(sigma_mins.min().detach())
"""

def max_tolerable_model_error(rho, C, epsilon_max, eta):
    """Compute the maximum tolerable model error for stability.

    Args:
        rho: spectral radius of the Koopman operator
        C: transient constant
        epsilon_max: maximum allowable latent-space error bound
        eta: disturbance bound

    Returns:
        max_error: float
    """
    return (epsilon_max * (1 - rho) / C) - eta


def latent_error_to_state_error(latent_error, m):
    """Convert latent-space error to state-space error.

    Args:
        latent_error: error in latent space
        m: lower Lipschitz constant of the encoder

    Returns:
        state_error: latent_error / m
    """
    return latent_error / m


def state_error_to_latent_error(state_error, m):
    """Convert state-space error to latent-space error.

    Args:
        state_error: error in state space
        m: lower Lipschitz constant of the encoder

    Returns:
        latent_error: state_error * m
    """
    return state_error * m


def optimize_lyapunov_P(A_cl_np, epsilon, eta, rho_grid_size=100):
    """Optimize P via SDP to maximize γ = ε*(1-ρ)/√κ - η.

    For each ρ in a grid, solves the SDP:
        minimize κ  s.t.  A_cl^T P A_cl ≼ ρ²P,  P ≽ I,  P ≼ κI

    Args:
        A_cl_np: (n, n) numpy array, closed-loop matrix A - BF
        epsilon: max tracking error in latent space
        eta: disturbance bound
        rho_grid_size: number of ρ values to sweep

    Returns:
        best_P: (n, n) numpy array, optimal Lyapunov matrix
        best_rho: float, optimal contraction rate
        best_kappa: float, optimal condition number
        best_gamma: float, optimal γ_max
    """
    import cvxpy as cp

    n = A_cl_np.shape[0]
    A_cl = A_cl_np.astype(np.float64)

    best_gamma = -np.inf
    best_P = None
    best_rho = None
    best_kappa = None

    for rho in np.linspace(0.01, 0.99, rho_grid_size):
        rho_sq = rho ** 2

        P = cp.Variable((n, n), symmetric=True)
        kappa = cp.Variable()

        constraints = [
            A_cl.T @ P @ A_cl << rho_sq * P,
            P >> np.eye(n),
            P << kappa * np.eye(n),
        ]
        prob = cp.Problem(cp.Minimize(kappa), constraints)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prob.solve(solver=cp.SCS, verbose=False)
        except cp.SolverError:
            continue

        if prob.status in ("optimal", "optimal_inaccurate") and kappa.value is not None:
            kappa_val = float(kappa.value)
            if kappa_val <= 0:
                continue
            gamma = epsilon * (1.0 - rho) / math.sqrt(kappa_val) - eta

            if gamma > best_gamma:
                best_gamma = gamma
                best_P = P.value.copy()
                best_rho = rho
                best_kappa = kappa_val

    return best_P, best_rho, best_kappa, best_gamma


def optimize_alpha_P(A_cl_np, C, epsilon_x, eta, P_init):
    """Optimize P to maximize γ = ε_x*(1-ρ)/√(α*λ_max(P)) - η.

    Parameterizes P = LᵀL via Cholesky factor and uses scipy.optimize
    to directly maximize γ. Initialized from DARE solution.

    Args:
        A_cl_np: (n, n) closed-loop matrix
        C: (p, n) state extraction matrix [I_p | 0_{p×q}]
        epsilon_x: max tracking error in state space
        eta: disturbance bound (from config)
        P_init: (n, n) initial P matrix (from DARE)

    Returns:
        best_P: (n, n) optimized Lyapunov matrix
        best_rho: float, optimal contraction rate
        best_alpha: float, optimal α = λ_max(P⁻¹ CᵀC)
        best_gamma: float, optimal γ_max
    """
    from scipy.optimize import minimize as scipy_minimize

    n = A_cl_np.shape[0]
    A_cl = A_cl_np.astype(np.float64)
    CtC = (C.T @ C).astype(np.float64)

    def neg_gamma_from_params(L_params):
        L = np.zeros((n, n))
        L[np.tril_indices(n)] = L_params
        P = L @ L.T

        try:
            P_eigvals = np.linalg.eigvalsh(P)
            if P_eigvals.min() < 1e-10:
                return 1e10

            # Contraction rate: ρ² = max|λ(P⁻¹ A_cl^T P A_cl)|
            # M is NOT symmetric, so use eigvals (not eigvalsh)
            M = np.linalg.solve(P, A_cl.T @ P @ A_cl)
            rho_sq = np.max(np.abs(np.linalg.eigvals(M)))

            # Smooth penalty for ρ² ≥ 1 instead of hard barrier
            if rho_sq >= 1.0:
                return 1e4 * rho_sq  # push toward stability
            if rho_sq < 0:
                return 1e10

            rho = np.sqrt(rho_sq)

            # Alpha: λ_max(P⁻¹ CᵀC)
            # P⁻¹ CᵀC is symmetric when P is symmetric, so eigvalsh is fine
            alpha = np.max(np.linalg.eigvalsh(np.linalg.solve(P, CtC)))
            if alpha <= 0:
                return 1e10

            lam_max_P = P_eigvals.max()
            gamma = epsilon_x * (1.0 - rho) / np.sqrt(alpha * lam_max_P) - eta
            return -gamma
        except np.linalg.LinAlgError:
            return 1e10

    # Find a feasible initial P
    # Build candidates: Lyapunov equation solution, DARE P, scaled identities
    from scipy.linalg import solve_discrete_lyapunov

    candidates = []

    # Check A_cl spectral radius
    eigvals_cl = np.linalg.eigvals(A_cl)
    rho_cl = np.max(np.abs(eigvals_cl))
    print(f"  A_cl spectral radius: {rho_cl:.6f}")

    # Discrete Lyapunov: A_cl^T P A_cl - P = -I  =>  P guaranteed feasible if A_cl stable
    try:
        P_lyap = solve_discrete_lyapunov(A_cl.T, np.eye(n))
        lyap_eigvals = np.linalg.eigvalsh(P_lyap)
        print(f"  Lyapunov P: λ_min={lyap_eigvals.min():.6f}, λ_max={lyap_eigvals.max():.6f}, "
              f"cond={lyap_eigvals.max()/lyap_eigvals.min():.4f}")
        candidates.append(("Lyapunov", P_lyap))
    except Exception as e:
        print(f"  Lyapunov solve failed: {e}")

    candidates.append(("DARE", P_init))
    candidates.append(("identity", np.eye(n)))
    for scale in [0.1, 0.5, 2.0, 5.0, 10.0, 100.0]:
        candidates.append((f"{scale}*I", np.eye(n) * scale))

    x0 = None
    for name, P_candidate in candidates:
        try:
            P_c = P_candidate.astype(np.float64)
            L_c = np.linalg.cholesky(P_c)
            params_c = L_c[np.tril_indices(n)]
            val = neg_gamma_from_params(params_c)
            if val < 1e4:  # feasible (ρ < 1)
                x0 = params_c
                print(f"  Initialized from {name} (γ={-val:.6f})")
                break
            else:
                print(f"  {name}: infeasible (obj={val:.4f})")
        except np.linalg.LinAlgError:
            print(f"  {name}: Cholesky failed (not positive definite)")
            continue

    if x0 is None:
        print(f"\033[91m  Could not find feasible initial P (A_cl may be unstable)\033[0m")
        return None, None, None, None, -np.inf

    print(f"  Optimizing alpha bound P ({len(x0)} parameters)...")
    result = scipy_minimize(
        fun=neg_gamma_from_params,
        x0=x0,
        method='L-BFGS-B',
        options={'maxiter': 2000, 'ftol': 1e-14},
    )

    # Extract best P
    L_best = np.zeros((n, n))
    L_best[np.tril_indices(n)] = result.x
    P_best = L_best @ L_best.T
    best_gamma = -result.fun

    # Compute final values
    rho_sq = np.max(np.abs(np.linalg.eigvals(np.linalg.solve(P_best, A_cl.T @ P_best @ A_cl))))
    best_rho = np.sqrt(max(rho_sq, 0.0))
    best_alpha = np.max(np.linalg.eigvalsh(np.linalg.solve(P_best, CtC)))
    lam_max_P = np.max(np.linalg.eigvalsh(P_best))

    print(f"  Optimization {'converged' if result.success else 'did not converge'} "
          f"after {result.nit} iterations")

    return P_best, best_rho, best_alpha, lam_max_P, best_gamma
