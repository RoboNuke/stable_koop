import torch
import torch.nn as nn

from model.normalized_layers import SemiOrthogonalLinear, OrthogonalLinear, GroupSort

class CayleyK(nn.Module):
    """Cayley parameterization: K is Schur-stable by construction."""
    def __init__(self, latent_dim, rho=0.95):
        super().__init__()
        self.rho = rho
        self.A_upper = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)
        self.latent_dim = latent_dim

    @property
    def K(self):
        A = self.A_upper - self.A_upper.T
        I = torch.eye(self.latent_dim, device=A.device)
        return self.rho * torch.linalg.solve(I + A, I - A)

    def forward(self, z):
        return z @ self.K.T


class SchurK(nn.Module):
    """Direct K parameterization with projection to Schur-stable set after each step."""
    def __init__(self, latent_dim, rho=0.95):
        super().__init__()
        self.rho = rho
        self.K_param = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)
        self.latent_dim = latent_dim

    @property
    def K(self):
        return self.K_param

    @torch.no_grad()
    def project(self):
        """Project K onto the Schur-stable set: K = K * rho / max(rho, spectral_radius(K))."""
        eigvals = torch.linalg.eigvals(self.K_param)
        spectral_radius = eigvals.abs().max().item()
        if spectral_radius > self.rho:
            self.K_param.mul_(self.rho / spectral_radius)

    def forward(self, z):
        return z @ self.K.T


class NormalK(nn.Module):
    def __init__(self, latent_dim, action_dim, rho=1.2):
        super().__init__()
        self.Q_upper = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)
        self.log_d = nn.Parameter(torch.zeros(latent_dim))
        self.b_eigen = nn.Parameter(torch.randn(latent_dim, action_dim) * 0.1)
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

    @property
    def B_from_eigen(self):
        return self.Q @ self.b_eigen  # (latent_dim, action_dim)

    def forward(self, z):
        return z @ self.K.T
    
    
class KoopmanAutoencoder(nn.Module):
    def __init__(self, state_dim=2, latent_dim=32, action_dim=1, 
                 k_type="cayley", encoder_type="linear", rho=0.95, 
                 encoder_spec_norm=False, encoder_latent=64):
        super().__init__()

        if encoder_type == "cayley":
            self.encoder = nn.Sequential(
                SemiOrthogonalLinear(state_dim, encoder_latent),
                GroupSort(2),
                OrthogonalLinear(encoder_latent, encoder_latent),
                GroupSort(2),
                SemiOrthogonalLinear(encoder_latent, latent_dim)
            )
        elif encoder_type == "linear":
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, encoder_latent), 
                #nn.Tanh(),
                nn.ReLU(),
                nn.Linear(encoder_latent, encoder_latent),        
                #nn.Tanh(),
                nn.ReLU(),
                nn.Linear(encoder_latent, latent_dim)
            )

        if encoder_spec_norm:
            for layer in self.encoder:
                if hasattr(layer, 'weight'):
                    nn.utils.parametrizations.spectral_norm(layer)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),         nn.Tanh(),
            nn.Linear(64, state_dim)
        )
        self.b_from_k_mod = False
        if k_type == "cayley":
            self.K_module = CayleyK(latent_dim, rho)
        elif k_type == "schur":
            self.K_module = SchurK(latent_dim, rho)
        elif k_type == "unbounded":
            self.K_module = nn.Linear(latent_dim, latent_dim, bias=False)
        elif k_type == "normalized":
            self.b_from_k_mod = True
            self.K_module = NormalK(latent_dim, action_dim, rho)
        else:
            raise NotImplementedError
        

        self.B = nn.Linear(action_dim, latent_dim, bias=False)
        nn.init.zeros_(self.B.weight)

    def encode(self, x):    return self.encoder(x)
    def decode(self, z):    return self.decoder(z)
    def predict(self, z, u):
        if self.b_from_k_mod:
            return self.K_module(z) + u @ self.K_module.B_from_eigen.T
        return self.K_module(z) + self.B(u)
    @property
    def A(self):
        return self.K_module.K if hasattr(self.K_module, 'K') else self.K_module.weight

    @property  
    def B_matrix(self):
        if self.b_from_k_mod:
            return self.K_module.B_from_eigen
        else:
            return self.B.weight
    
    def prediction_error(self, z_t, u_t, z_next):
        """Compute the Koopman one-step prediction error in latent space.
            Note this is only for a single sample and met to be used online to verify 
            model validity
        Args:
            z_t: (..., latent_dim) current latent state
            u_t: (..., action_dim) control input
            z_next: (..., latent_dim) actual next latent state

        Returns:
            error: float, ||z_next - (K z_t + B u_t)||
        """
        z_pred = self.predict(z_t, u_t)
        return torch.linalg.norm(z_next - z_pred).item()

    def verify_koopman(self, held_out_data, delta_max):
        """Verify Koopman prediction accuracy on held-out data.

        Args:
            held_out_data: iterable of (z_t, u_t, z_next) tuples
            delta_max: maximum allowable prediction error

        Returns:
            max_error: float, worst-case prediction error
            is_valid: bool, True if max_error <= delta_max
        """
        errors = [self.prediction_error(z_t, u_t, z_next)
                  for z_t, u_t, z_next in held_out_data]
        max_err = max(errors)
        return max_err, max_err <= delta_max
    
    def initialize_B_in_eigenbasis(self):
        with torch.no_grad():
            A = self.A.detach()
            _, V = torch.linalg.eig(A)
            V_real = V.real  # (latent_dim, latent_dim)
            # Project current B onto eigenvector space
            B = self.B_matrix  # (latent_dim, action_dim)
            B_proj = V_real @ (V_real.T @ B)  # project onto eigenbasis
            self.B.weight.copy_(B_proj )
