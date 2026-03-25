import torch
import torch.nn as nn


class OrthogonalLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.A_upper = nn.Parameter(torch.randn(in_features, in_features) * 0.01)
        self.in_features = in_features
        
    @property
    def W(self):
        A = self.A_upper - self.A_upper.T
        I = torch.eye(self.in_features, device=A.device)
        return torch.linalg.solve(I + A, I - A)  # orthogonal matrix
    
    def forward(self, x):
        return x @ self.W.T
import geoopt

class SemiOrthogonalLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Stiefel manifold requires shape[-1] <= shape[-2] (tall matrix).
        # When out < in, store transposed and flip in forward.
        self._transposed = out_features < in_features
        if self._transposed:
            # Store as (in, out) so rows >= cols
            self.weight = geoopt.ManifoldParameter(
                torch.empty(in_features, out_features),
                manifold=geoopt.manifolds.Stiefel()
            )
        else:
            self.weight = geoopt.ManifoldParameter(
                torch.empty(out_features, in_features),
                manifold=geoopt.manifolds.Stiefel()
            )
        nn.init.orthogonal_(self.weight)

    def forward(self, x):
        if self._transposed:
            return x @ self.weight  # (B, in) @ (in, out) -> (B, out)
        return x @ self.weight.T  # (B, in) @ (in, out) -> (B, out)
    
"""
class SemiOrthogonalLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        #assert out_features >= in_features
        # Use PyTorch's built-in orthogonal parameterization
        self.linear = nn.Linear(in_features, out_features, bias=False)
        nn.utils.parametrizations.orthogonal(self.linear)

    def forward(self, x):
        return self.linear(x)
"""


class GroupSort(nn.Module):
    def __init__(self, group_size=2):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.group_size)
        x, _ = x.sort(dim=-1)
        return x.reshape(shape)

