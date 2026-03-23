"""
Unit tests for Koopman-LQR stability framework.
All tests use analytically known values so correctness can be verified by hand.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, "/mnt/user-data/uploads")

from model.utils import (
    spectral_radius,
    transient_constant,
    compute_lower_lipschitz,
    max_tolerable_model_error,
    latent_error_to_state_error,
    state_error_to_latent_error,
)
from controllers.lqr import LQR
from model.autoencoder import KoopmanAutoencoder, CayleyK, SchurK


# ─────────────────────────────────────────────
# Tolerances
# ─────────────────────────────────────────────
ATOL = 1e-5


# ═════════════════════════════════════════════
# utils.py
# ═════════════════════════════════════════════

class TestSpectralRadius:
    def test_diagonal_matrix(self):
        """Spectral radius of diag(0.5, 0.3, 0.8) is 0.8."""
        M = torch.diag(torch.tensor([0.5, 0.3, 0.8]))
        assert abs(spectral_radius(M) - 0.8) < ATOL

    def test_negative_eigenvalue(self):
        """Spectral radius uses magnitude, so diag(-0.9, 0.5) -> 0.9."""
        M = torch.diag(torch.tensor([-0.9, 0.5]))
        assert abs(spectral_radius(M) - 0.9) < ATOL

    def test_complex_eigenvalues(self):
        """2x2 rotation matrix has eigenvalues e^{±iθ}, magnitude 1."""
        theta = torch.tensor(0.3)
        M = torch.tensor([
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta),  torch.cos(theta)]
        ])
        assert abs(spectral_radius(M) - 1.0) < ATOL

    def test_stable_matrix(self):
        """All eigenvalues inside unit circle -> rho < 1."""
        M = torch.tensor([[0.5, 0.1], [0.0, 0.3]])
        assert spectral_radius(M) < 1.0

    def test_scalar(self):
        """1x1 matrix [[v]] has spectral radius |v|."""
        M = torch.tensor([[-0.7]])
        assert abs(spectral_radius(M) - 0.7) < ATOL


class TestTransientConstant:
    def test_normal_matrix(self):
        """
        A symmetric (normal) matrix has orthonormal eigenvectors,
        so V is orthogonal and cond(V) = 1.
        """
        # Symmetric -> eigenvectors are orthogonal -> V is unitary -> cond = 1
        M = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
        C = transient_constant(M)
        assert abs(C - 1.0) < 1e-4

    def test_identity(self):
        """Identity matrix: eigenvectors are standard basis, C = 1."""
        M = torch.eye(4)
        C = transient_constant(M)
        assert abs(C - 1.0) < ATOL

    def test_diagonal(self):
        """Diagonal matrix with distinct entries: eigenvectors are standard basis, C = 1."""
        M = torch.diag(torch.tensor([0.9, 0.5, 0.3]))
        C = transient_constant(M)
        assert abs(C - 1.0) < ATOL

    def test_c_geq_one(self):
        """C is always >= 1 (condition number is always >= 1)."""
        torch.manual_seed(0)
        M = torch.randn(5, 5) * 0.5
        C = transient_constant(M)
        assert C >= 1.0 - ATOL


class TestMaxTolerableModelError:
    def test_known_value(self):
        """
        With rho=0.5, C=2.0, epsilon_max=1.0, eta=0.1:
        delta_max = (1.0 * (1 - 0.5) / 2.0) - 0.1 = 0.25 - 0.1 = 0.15
        """
        result = max_tolerable_model_error(rho=0.5, C=2.0, epsilon_max=1.0, eta=0.1)
        assert abs(result - 0.15) < ATOL

    def test_zero_eta(self):
        """With eta=0: delta_max = epsilon_max * (1 - rho) / C."""
        result = max_tolerable_model_error(rho=0.8, C=1.0, epsilon_max=1.0, eta=0.0)
        assert abs(result - 0.2) < ATOL

    def test_negative_when_model_too_inaccurate(self):
        """
        If model error budget is exhausted by C and rho alone,
        delta_max should be negative (system cannot meet epsilon_max).
        """
        # epsilon_max * (1 - rho) / C = 0.1 * 0.1 / 2.0 = 0.005, minus eta=0.1 -> negative
        result = max_tolerable_model_error(rho=0.9, C=2.0, epsilon_max=0.1, eta=0.1)
        assert result < 0.0

    def test_inverse_of_iss_bound(self):
        """
        Round-trip: compute delta_max, then verify ISS bound recovers epsilon_max.
        epsilon_max = C * (delta_max + eta) / (1 - rho)
        """
        rho, C, epsilon_max, eta = 0.6, 1.5, 2.0, 0.3
        delta_max = max_tolerable_model_error(rho, C, epsilon_max, eta)
        recovered = C * (delta_max + eta) / (1 - rho)
        assert abs(recovered - epsilon_max) < ATOL


class TestErrorConversions:
    def test_latent_to_state(self):
        """x-space error = latent error / m."""
        assert abs(latent_error_to_state_error(1.0, 0.5) - 2.0) < ATOL

    def test_state_to_latent(self):
        """Latent error = x-space error * m."""
        assert abs(state_error_to_latent_error(2.0, 0.5) - 1.0) < ATOL

    def test_round_trip(self):
        """Converting latent -> state -> latent recovers original value."""
        m = 0.73
        latent_err = 0.42
        state_err = latent_error_to_state_error(latent_err, m)
        recovered = state_error_to_latent_error(state_err, m)
        assert abs(recovered - latent_err) < ATOL

    def test_unit_m_is_identity(self):
        """With m=1, both conversions are identity."""
        assert abs(latent_error_to_state_error(0.5, 1.0) - 0.5) < ATOL
        assert abs(state_error_to_latent_error(0.5, 1.0) - 0.5) < ATOL


class TestComputeLowerLipschitz:
    def test_linear_encoder_known_singular_value(self):
        """
        For a linear encoder (single weight matrix W, no activation),
        the Jacobian is W everywhere. The minimum singular value of W
        is the lower Lipschitz constant.
        We construct W with known singular values via SVD.
        """
        torch.manual_seed(42)
        state_dim, latent_dim = 3, 4
        # Build W = U S V^T with known singular values
        U, _ = torch.linalg.qr(torch.randn(latent_dim, latent_dim))
        V, _ = torch.linalg.qr(torch.randn(state_dim, state_dim))
        S = torch.tensor([2.0, 1.5, 0.7])  # singular values, min = 0.7
        W = U[:, :state_dim] @ torch.diag(S) @ V.T  # (latent_dim, state_dim)

        encoder = nn.Linear(state_dim, latent_dim, bias=False)
        with torch.no_grad():
            encoder.weight.copy_(W)

        training_data = [torch.randn(state_dim) for _ in range(20)]
        m = compute_lower_lipschitz(encoder, training_data)
        assert abs(m - 0.7) < 1e-4

    def test_m_positive(self):
        """Lower Lipschitz bound should always be non-negative."""
        encoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 6))
        training_data = [torch.randn(4) for _ in range(10)]
        m = compute_lower_lipschitz(encoder, training_data)
        assert m >= 0.0


# ═════════════════════════════════════════════
# lqr.py
# ═════════════════════════════════════════════

class TestLQR:
    @pytest.fixture
    def simple_system(self):
        """
        Simple 2D marginally unstable system with scalar control.
        A = [[1.1, 0], [0, 0.9]]  (one unstable mode)
        B = [[1], [0]]
        Q = I, R = I
        """
        A = torch.tensor([[1.1, 0.0], [0.0, 0.9]])
        B = torch.tensor([[1.0], [0.0]])
        Q = torch.eye(2)
        R = torch.eye(1)
        return LQR(A, B, Q, R)

    def test_closed_loop_is_stable(self, simple_system):
        """(A - BF) must have spectral radius < 1 after LQR solve."""
        rho = spectral_radius(simple_system.closed_loop)
        assert rho < 1.0

    def test_f_shape(self, simple_system):
        """F should be (action_dim, state_dim) = (1, 2)."""
        assert simple_system.F.shape == (1, 2)

    def test_closed_loop_shape(self, simple_system):
        """Closed loop matrix should be (state_dim, state_dim) = (2, 2)."""
        assert simple_system.closed_loop.shape == (2, 2)

    def test_gain_norm_positive(self, simple_system):
        """Gain norm should be a positive scalar."""
        assert simple_system.gain_norm.item() > 0.0

    def test_control_drives_toward_reference(self, simple_system):
        """
        u = F(z_ref - z) should point in the direction that reduces error.
        If z > z_ref, u should be negative (push back).
        """
        z = torch.tensor([[1.0, 0.0]])
        z_ref = torch.tensor([[0.0, 0.0]])
        u = simple_system(z, z_ref)
        # Error is positive (z > z_ref in first dim), u should be negative
        assert u[0, 0].item() < 0.0

    def test_zero_error_zero_control(self, simple_system):
        """When z == z_ref, control should be zero."""
        z = torch.tensor([[0.5, -0.3]])
        u = simple_system(z, z)
        assert torch.allclose(u, torch.zeros_like(u), atol=ATOL)

    def test_batch_control(self, simple_system):
        """LQR should handle batched inputs correctly."""
        z = torch.randn(8, 2)
        z_ref = torch.randn(8, 2)
        u = simple_system(z, z_ref)
        assert u.shape == (8, 1)

    def test_stabilizes_unstable_system(self, simple_system):
        """
        Simulate closed-loop trajectory from large initial error.
        Error should decay to near zero within reasonable steps.
        """
        z = torch.tensor([[5.0, 2.0]])
        z_ref = torch.tensor([[0.0, 0.0]])
        A = simple_system.A
        B = simple_system.B

        for _ in range(200):
            u = simple_system(z, z_ref)
            z = z @ A.T + u @ B.T

        error = torch.linalg.norm(z - z_ref).item()
        assert error < 0.1


# ═════════════════════════════════════════════
# autoencoder.py
# ═════════════════════════════════════════════

class TestCayleyK:
    def test_spectral_radius_bounded(self):
        """CayleyK should produce K with spectral radius <= rho."""
        rho = 0.95
        K_module = CayleyK(latent_dim=8, rho=rho)
        sr = spectral_radius(K_module.K)
        assert sr <= rho + ATOL

    def test_output_shape(self):
        z = torch.randn(4, 8)
        K_module = CayleyK(latent_dim=8)
        out = K_module(z)
        assert out.shape == z.shape


class TestSchurK:
    def test_spectral_radius_after_project(self):
        """After project(), spectral radius should be <= rho."""
        rho = 0.9
        K_module = SchurK(latent_dim=6, rho=rho)
        # Manually inflate K to be unstable
        with torch.no_grad():
            K_module.K_param.mul_(10.0)
        K_module.project()
        sr = spectral_radius(K_module.K)
        assert sr <= rho + ATOL

    def test_no_project_needed_when_stable(self):
        """If already stable, project() should not change K."""
        rho = 0.95
        K_module = SchurK(latent_dim=4, rho=rho)
        # Shrink to be well within stable region
        with torch.no_grad():
            K_module.K_param.mul_(0.01)
        K_before = K_module.K.clone()
        K_module.project()
        assert torch.allclose(K_before, K_module.K, atol=ATOL)


class TestKoopmanAutoencoder:
    @pytest.fixture
    def model(self):
        return KoopmanAutoencoder(
            state_dim=4, latent_dim=8, action_dim=2,
            k_type="cayley", rho=0.95
        )

    def test_encode_shape(self, model):
        x = torch.randn(10, 4)
        z = model.encode(x)
        assert z.shape == (10, 8)

    def test_decode_shape(self, model):
        z = torch.randn(10, 8)
        x = model.decode(z)
        assert x.shape == (10, 4)

    def test_predict_shape(self, model):
        z = torch.randn(10, 8)
        u = torch.randn(10, 2)
        z_next = model.predict(z, u)
        assert z_next.shape == (10, 8)

    def test_A_shape(self, model):
        """A matrix should be (latent_dim, latent_dim)."""
        assert model.A.shape == (8, 8)

    def test_B_matrix_shape(self, model):
        """B matrix should be (latent_dim, action_dim)."""
        assert model.B_matrix.shape == (8, 2)

    def test_B_matrix_correct(self, model):
        """
        B_matrix should match the weight of B layer directly (not transposed).
        Verify: B(u) = u @ B.weight.T = B_matrix @ u (as column vec).
        """
        u = torch.randn(1, 2)
        # Direct layer output
        direct = model.B(u)  # (1, latent_dim)
        # Via B_matrix: u @ B_matrix.T
        via_property = u @ model.B_matrix.T
        assert torch.allclose(direct, via_property, atol=ATOL)

    def test_b_init_zeros(self, model):
        """B should be initialized to zeros."""
        assert torch.allclose(model.B_matrix, torch.zeros_like(model.B_matrix))

    def test_prediction_error_zero_for_perfect_model(self):
        """
        Use k_type='unbounded' so K is a plain nn.Linear we can set to identity.
        CayleyK cannot be forced to identity via its A_upper parameterization.
        With K=I and B=0, predict(z, u) = z exactly, so error is zero.
        """
        latent_dim = 8
        model = KoopmanAutoencoder(
            state_dim=4, latent_dim=latent_dim, action_dim=2,
            k_type="unbounded"
        )
        with torch.no_grad():
            model.K_module.weight.copy_(torch.eye(latent_dim))
            model.B.weight.zero_()

        z = torch.randn(latent_dim)
        u = torch.randn(2)
        error = model.prediction_error(z, u, z)
        assert abs(error) < ATOL

    def test_verify_koopman_passes_when_accurate(self, model):
        """verify_koopman should return True when all errors are below delta_max."""
        with torch.no_grad():
            model.B.weight.zero_()

        # Generate data where z_next = predict(z, u) exactly
        data = []
        for _ in range(20):
            z = torch.randn(8)
            u = torch.randn(2)
            z_next = model.predict(z, u).detach()
            data.append((z, u, z_next))

        max_err, is_valid = model.verify_koopman(data, delta_max=1e-4)
        assert is_valid
        assert max_err < 1e-4

    def test_verify_koopman_fails_when_inaccurate(self, model):
        """verify_koopman should return False when errors exceed delta_max."""
        data = []
        for _ in range(20):
            z = torch.randn(8)
            u = torch.randn(2)
            # z_next is random noise, completely wrong
            z_next = torch.randn(8) * 10.0
            data.append((z, u, z_next))

        _, is_valid = model.verify_koopman(data, delta_max=1e-4)
        assert not is_valid

    def test_a_is_stable_cayley(self):
        """CayleyK autoencoder A matrix should have spectral radius <= rho."""
        model = KoopmanAutoencoder(state_dim=4, latent_dim=8, k_type="cayley", rho=0.9)
        sr = spectral_radius(model.A)
        assert sr <= 0.9 + ATOL


class TestLQRWithKoopman:
    def test_full_pipeline(self):
        """
        Integration test: build autoencoder, extract A and B_matrix,
        construct LQR, verify closed loop is stable.
        """
        model = KoopmanAutoencoder(
            state_dim=4, latent_dim=8, action_dim=2,
            k_type="cayley", rho=0.95
        )
        A = model.A.detach()
        B = model.B_matrix.detach()
        Q = torch.eye(8)
        R = torch.eye(2)

        lqr = LQR(A, B, Q, R)
        rho = spectral_radius(lqr.closed_loop)
        assert rho < 1.0

    def test_iss_bound_positive(self):
        """
        Diagonal A has orthonormal eigenvectors so C=1 exactly.
        With B=I, LQR produces a well-conditioned closed loop.
        delta_max is then predictable and guaranteed positive.
        """
        A = torch.diag(torch.tensor([0.8, 0.7, 0.6]))
        B = torch.eye(3)
        Q = torch.eye(3)
        R = torch.eye(3)
        lqr = LQR(A, B, Q, R)

        rho = spectral_radius(lqr.closed_loop)
        C = transient_constant(lqr.closed_loop)
        epsilon_max = 2.0
        eta = 0.1

        delta_max = max_tolerable_model_error(rho, C, epsilon_max, eta)
        assert delta_max > 0.0, (
            f"delta_max={delta_max:.4f}: rho={rho:.4f}, C={C:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
