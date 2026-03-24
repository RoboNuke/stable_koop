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


def compute_lower_lipschitz(encoder, training_data):
    """Compute the lower Lipschitz constant of the encoder via minimum singular
    value of the Jacobian across training data.

    Args:
        encoder: callable mapping (state_dim,) -> (latent_dim,)
        training_data: iterable of state vectors (numpy arrays or tensors)

    Returns:
        m: float, the lower Lipschitz constant
    """
    X = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in training_data])
    J_batch = vmap(jacrev(encoder))(X)  # (N, latent_dim, state_dim)
    sigma_mins = torch.linalg.svdvals(J_batch)[:, -1]
    return float(sigma_mins.min().detach())


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
