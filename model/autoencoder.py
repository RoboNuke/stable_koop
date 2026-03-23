import torch
import torch.nn as nn


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


class KoopmanAutoencoder(nn.Module):
    def __init__(self, state_dim=2, latent_dim=32, action_dim=1, 
                 k_type="cayley", rho=0.95, 
                 encoder_spec_norm=False, encoder_latent=64):
        super().__init__()
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
        if k_type == "cayley":
            self.K_module = CayleyK(latent_dim, rho)
        elif k_type == "schur":
            self.K_module = SchurK(latent_dim, rho)
        elif k_type == "unbounded":
            self.K_module = nn.Linear(latent_dim, latent_dim, bias=False)
        else:
            raise NotImplementedError
        

        self.B = nn.Linear(action_dim, latent_dim, bias=False)
        nn.init.zeros_(self.B.weight)

    def encode(self, x):    return self.encoder(x)
    def decode(self, z):    return self.decoder(z)
    def predict(self, z, u): return self.K_module(z) + self.B(u)
    @property
    def A(self):
        return self.K_module.K if hasattr(self.K_module, 'K') else self.K_module.weight

    @property  
    def B_matrix(self):
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
