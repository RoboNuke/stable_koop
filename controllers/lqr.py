import torch
from scipy.linalg import solve_discrete_are


class LQR:
    """Discrete-time LQR controller.

    Computes optimal gain F for dynamics x_{t+1} = (A - B F) x_t
    with cost sum of (x - x_ref)^T Q (x - x_ref) + u^T R u.

    Q and R are set at construction. The reference state is passed at call time.

    Args:
        A: (n, n) state transition matrix (torch.Tensor)
        B: (n, m) control input matrix (torch.Tensor)
        Q: (n, n) state cost matrix (torch.Tensor)
        R: (m, m) control cost matrix (torch.Tensor)
    """

    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        P = solve_discrete_are(A.numpy(), B.numpy(), Q.numpy(), R.numpy())
        P = torch.from_numpy(P).to(A.dtype)
        self.P = P
        self.F = torch.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        self.gain_norm = torch.linalg.norm(self.F, ord=2)

    @property
    def closed_loop(self):
        return self.A - self.B @ self.F

    def __call__(self, x, x_ref):
        """Compute the LQR control action.

        Args:
            x: (..., n) current state
            x_ref: (..., n) reference state

        Returns:
            u: (..., m) control action
        """
        return -((x - x_ref) @ self.F.T)
