"""Composed Koopman autoencoder: encoder + A + B + decoder.

Single source of truth for the ``KoopmanAutoencoder`` module. Building blocks
live in their own focused files: A parameterizations in
:mod:`models.a_parameterizations`, encoder/decoder MLPs in :mod:`models.encoder`,
hand-designed lifting in :mod:`models.lifting`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.a_parameterizations import (
    CayleyK,
    ComplexNormalK,
    SchurK,
)
from models.encoder import build_decoder, build_encoder
from models.lifting import trig_lift


class KoopmanAutoencoder(nn.Module):
    def __init__(
        self,
        state_dim: int = 2,
        latent_dim: int = 32,
        action_dim: int = 1,
        k_type: str = "cayley",
        encoder_type: str = "linear",
        rho: float = 0.95,
        encoder_spec_norm: bool = False,
        encoder_latent: int = 64,
        prepend_state: bool = False,
        prepend_control: bool = False,
        real_state_dim: int | None = None,
        obs_type: str = "theta",
    ):
        super().__init__()

        self.prepend_state = prepend_state
        self.prepend_control = prepend_control
        self.obs_type = obs_type
        self.raw_state_dim = state_dim
        self.action_dim = action_dim
        self.real_state_dim = real_state_dim if real_state_dim is not None else state_dim

        if prepend_state:
            self.prepend_dim = self.real_state_dim
            if prepend_control:
                self.prepend_dim += action_dim
        else:
            self.prepend_dim = 0
        self.encoder_latent_dim = latent_dim

        self.encoder_type = encoder_type
        self._trig_encoder = encoder_type == "trig"
        self._identity_encoder = encoder_type == "identity"

        if self._trig_encoder:
            assert latent_dim % 2 == 0, (
                f"trig encoder requires even latent_dim, got {latent_dim}"
            )
            self.encoder = None
        elif self._identity_encoder:
            self.encoder = None
        else:
            self.encoder = build_encoder(
                encoder_type=encoder_type,
                state_dim=state_dim,
                latent_dim=latent_dim,
                encoder_latent=encoder_latent,
                spec_norm=encoder_spec_norm and encoder_type == "linear",
            )
            # Cayley family applies spectral_norm directly here to match the
            # historical placement (after sequential construction).
            if encoder_spec_norm and encoder_type == "cayley":
                for layer in self.encoder:
                    if hasattr(layer, "weight"):
                        nn.utils.parametrizations.spectral_norm(layer)

        if self._trig_encoder or self._identity_encoder:
            self.decoder = None
        else:
            self.decoder = build_decoder(latent_dim, encoder_latent, state_dim)

        actual_latent = self.prepend_dim + latent_dim if prepend_state else latent_dim

        self.b_from_k_mod = False
        if k_type == "cayley":
            self.K_module = CayleyK(actual_latent, rho)
        elif k_type == "schur":
            self.K_module = SchurK(actual_latent, rho)
        elif k_type == "unbounded":
            self.K_module = nn.Linear(actual_latent, actual_latent, bias=False)
            # Physics-informed init when actual_latent==5 (preserved historical behavior).
            if actual_latent == 5:
                dt = 0.01
                with torch.no_grad():
                    A_init = torch.rand(5, 5)
                    A_init[0] = torch.tensor([1.0, 0.0, 0.0, 0.0, -dt / 8.0])
                    A_init[1] = torch.tensor([0.0, 1.0, 0.0, dt / 8.0, 0.0])
                    A_init[2] = torch.tensor([0.0, 0.75 / 8.0, 1.0, 0.0, 0.0])
                    self.K_module.weight.copy_(A_init)
        elif k_type == "normalized":
            self.b_from_k_mod = False  # historically False (not True) despite ComplexNormalK
            self.K_module = ComplexNormalK(actual_latent, action_dim, rho)
        else:
            raise NotImplementedError(f"Unknown k_type {k_type!r}")

        self.B = nn.Linear(action_dim, actual_latent, bias=False)

    def encode(self, x):
        if self._trig_encoder:
            g = trig_lift(x, self.encoder_latent_dim, self.obs_type)
        elif self._identity_encoder:
            g = x
        else:
            g = self.encoder(x)
        if self.prepend_state:
            return torch.cat([x[..., : self.prepend_dim], g], dim=-1)
        return g

    def decode(self, z):
        if self._identity_encoder:
            return z
        if self._trig_encoder:
            return z[..., : self.raw_state_dim]
        if self.prepend_state:
            g = z[..., self.prepend_dim :]
            return self.decoder(g)
        return self.decoder(z)

    def predict(self, z, u):
        if self.b_from_k_mod:
            return self.K_module(z) + u @ self.K_module.B_from_eigen.T
        return self.K_module(z) + self.B(u)

    @property
    def A(self):
        return self.K_module.K if hasattr(self.K_module, "K") else self.K_module.weight

    @property
    def B_matrix(self):
        if self.b_from_k_mod:
            return self.K_module.B_from_eigen
        return self.B.weight

    def prediction_error(self, z_t, u_t, z_next):
        """Online one-step latent prediction error norm (single sample)."""
        z_pred = self.predict(z_t, u_t)
        return torch.linalg.norm(z_next - z_pred).item()

    def verify_koopman(self, held_out_data, delta_max):
        """Worst-case prediction error across ``held_out_data`` plus a validity flag."""
        errors = [
            self.prediction_error(z_t, u_t, z_next) for z_t, u_t, z_next in held_out_data
        ]
        max_err = max(errors)
        return max_err, max_err <= delta_max

    def initialize_B_in_eigenbasis(self):
        with torch.no_grad():
            A = self.A.detach()
            _, V = torch.linalg.eig(A)
            V_real = V.real
            B = self.B_matrix
            B_proj = V_real @ (V_real.T @ B)
            self.B.weight.copy_(B_proj)
