"""A-matrix parameterizations for the Koopman operator.

Each class exposes a ``K`` property (the latent-dynamics matrix) and is callable
as ``z @ K.T``. Some carry an action embedding (``B_from_eigen``) used by the
"normalized" pathway in :class:`KoopmanAutoencoder`.

Single source of truth — no copies elsewhere in the repo.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CayleyK(nn.Module):
    """Cayley parameterization: K is Schur-stable by construction."""

    def __init__(self, latent_dim, rho=0.95):
        super().__init__()
        self.rho = rho
        self.A_upper = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.latent_dim = latent_dim

    @property
    def K(self):
        A = self.A_upper - self.A_upper.T
        I = torch.eye(self.latent_dim, device=A.device)
        return self.rho * torch.linalg.solve(I + A, I - A)

    def forward(self, z):
        return z @ self.K.T


class SchurK(nn.Module):
    """Direct K parameterization with projection onto the Schur-stable set."""

    def __init__(self, latent_dim, rho=0.95):
        super().__init__()
        self.rho = rho
        self.K_param = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.latent_dim = latent_dim

    @property
    def K(self):
        return self.K_param

    @torch.no_grad()
    def project(self):
        U, S, Vh = torch.linalg.svd(self.K_param)
        if S[0].item() > self.rho:
            S = S.clamp(max=self.rho)
            self.K_param.copy_(U @ torch.diag(S) @ Vh)

    def forward(self, z):
        return z @ self.K.T


class NormalK(nn.Module):
    """Normal-matrix parameterization with real eigenvalues."""

    def __init__(self, latent_dim, action_dim, rho=1.2):
        super().__init__()
        self.Q_upper = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)
        self.log_d = nn.Parameter(torch.zeros(latent_dim))
        self.b_eigen = nn.Parameter(torch.randn(latent_dim, action_dim))
        self.latent_dim = latent_dim
        self.rho = rho

    @property
    def Q(self):
        A = self.Q_upper - self.Q_upper.T
        I = torch.eye(self.latent_dim, device=A.device)
        return torch.linalg.solve(I + A, I - A)

    @property
    def K(self):
        d = torch.tanh(self.log_d) * self.rho
        return self.Q @ torch.diag(d) @ self.Q.T

    def forward(self, z):
        return z @ self.K.T

    @property
    def B_from_eigen(self):
        return self.Q @ self.b_eigen  # (latent_dim, action_dim)


class ComplexNormalK(nn.Module):
    """Normal-matrix parameterization with complex conjugate eigenvalue pairs.

    A = Q @ D @ Q.T where Q is orthogonal and D is block-diagonal with 2×2
    rotation blocks, giving complex conjugate pairs r·e^{±iθ}. C=1 for A and
    oscillatory dynamics are supported.
    """

    def __init__(self, latent_dim, action_dim, rho=1.2):
        super().__init__()
        assert latent_dim % 2 == 0, "latent_dim must be even for complex conjugate pairs"
        self.n_pairs = latent_dim // 2
        self.latent_dim = latent_dim
        self.rho = rho

        self.Q_upper = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)

        r_init = torch.linspace(0.5, 0.95, self.n_pairs)
        self.log_r = nn.Parameter(torch.atanh(r_init / rho))

        theta_init = torch.linspace(0.1, torch.pi - 0.1, self.n_pairs)
        self.theta = nn.Parameter(theta_init)

        self.b_eigen = nn.Parameter(torch.randn(latent_dim, action_dim) * 0.1)

    @property
    def Q(self):
        A = self.Q_upper - self.Q_upper.T
        I = torch.eye(self.latent_dim, device=A.device)
        return torch.linalg.solve(I + A, I - A)

    @property
    def K(self):
        r = torch.tanh(self.log_r) * self.rho
        cos_t = torch.cos(self.theta)
        sin_t = torch.sin(self.theta)

        D = torch.zeros(self.latent_dim, self.latent_dim, device=r.device)
        for i in range(self.n_pairs):
            j = 2 * i
            D[j, j] = r[i] * cos_t[i]
            D[j, j + 1] = -r[i] * sin_t[i]
            D[j + 1, j] = r[i] * sin_t[i]
            D[j + 1, j + 1] = r[i] * cos_t[i]

        return self.Q @ D @ self.Q.T

    @property
    def B_from_eigen(self):
        return self.Q @ self.b_eigen

    def forward(self, z):
        return z @ self.K.T


def build_a_parameterization(k_type: str, latent_dim: int, action_dim: int, rho: float):
    """Factory dispatching the YAML ``k_type`` string to its module class.

    The ``unbounded`` option is constructed inline by callers (``nn.Linear``)
    because some call sites apply physics-informed initialization to its
    weight. See :class:`KoopmanAutoencoder` for the reference initialization.
    """
    if k_type == "cayley":
        return CayleyK(latent_dim, rho)
    if k_type == "schur":
        return SchurK(latent_dim, rho)
    if k_type == "complex_normal":
        return ComplexNormalK(latent_dim, action_dim, rho)
    if k_type == "normalized":
        # Historical name in stable_koop: "normalized" → ComplexNormalK.
        return ComplexNormalK(latent_dim, action_dim, rho)
    if k_type == "normal":
        return NormalK(latent_dim, action_dim, rho)
    raise NotImplementedError(f"Unknown k_type {k_type!r}")
