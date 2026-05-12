"""Encoder and decoder MLPs used by :class:`KoopmanAutoencoder`.

These are the learned lifting/un-lifting networks. The hand-designed (``trig``)
and pass-through (``identity``) lifting modes are handled separately by the
:class:`KoopmanAutoencoder` orchestrator — see :mod:`models.lifting`.
"""

from __future__ import annotations

import torch.nn as nn

from models.normalized_layers import GroupSort, OrthogonalLinear, SemiOrthogonalLinear


def build_encoder(
    encoder_type: str,
    state_dim: int,
    latent_dim: int,
    encoder_latent: int,
    spec_norm: bool = False,
) -> nn.Module:
    """Build the encoder network for the named family.

    ``encoder_type`` is one of ``"cayley"`` or ``"linear"``. The ``trig`` /
    ``identity`` paths return ``None`` and are handled by the
    :class:`KoopmanAutoencoder` orchestrator. ``spec_norm`` wraps every layer
    with ``nn.utils.parametrizations.spectral_norm`` if true (linear family only).
    """
    if encoder_type == "cayley":
        return nn.Sequential(
            SemiOrthogonalLinear(state_dim, encoder_latent),
            GroupSort(2),
            OrthogonalLinear(encoder_latent, encoder_latent),
            GroupSort(2),
            SemiOrthogonalLinear(encoder_latent, latent_dim),
        )
    if encoder_type == "linear":
        bias = True
        enc = nn.Sequential(
            nn.Linear(state_dim, encoder_latent, bias=bias),
            nn.ReLU(),
            nn.Linear(encoder_latent, encoder_latent, bias=bias),
            nn.ReLU(),
            nn.Linear(encoder_latent, encoder_latent, bias=bias),
            nn.ReLU(),
            nn.Linear(encoder_latent, latent_dim, bias=bias),
        )
        if spec_norm:
            for layer in enc:
                if hasattr(layer, "weight"):
                    nn.utils.parametrizations.spectral_norm(layer)
        return enc
    raise NotImplementedError(f"Unknown encoder_type {encoder_type!r}")


def build_decoder(latent_dim: int, encoder_latent: int, state_dim: int) -> nn.Module:
    """Build the decoder MLP that mirrors the linear/cayley encoder family."""
    decoder_size = encoder_latent // 2
    return nn.Sequential(
        nn.Linear(latent_dim, decoder_size),
        nn.ReLU(),
        nn.Linear(decoder_size, decoder_size),
        nn.ReLU(),
        nn.Linear(decoder_size, decoder_size),
        nn.ReLU(),
        nn.Linear(decoder_size, decoder_size),
        nn.ReLU(),
        nn.Linear(decoder_size, state_dim),
    )
