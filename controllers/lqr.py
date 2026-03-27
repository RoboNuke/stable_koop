import torch
from scipy.linalg import solve_discrete_are
import numpy as np

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

    def __init__(self, A, B, Q, R, q_scale=1.0, controllable_subspace=False,
                 ctrl_threshold=None):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.controllable_subspace = controllable_subspace

        if controllable_subspace:
            self._solve_controllable_subspace(q_scale, ctrl_threshold)
        else:
            self._solve_full(q_scale)

    def _solve_full(self, q_scale):
        """Standard full-dimensional DARE solve."""
        P = solve_discrete_are(
            self.A.numpy(), self.B.numpy(),
            self.Q.numpy(), self.R.numpy()
        )
        P = torch.from_numpy(P).to(self.A.dtype)
        self.P = P
        self.F = torch.linalg.solve(
            self.R + self.B.T @ self.P @ self.B,
            self.B.T @ self.P @ self.A
        )
        self.gain_norm = torch.linalg.norm(self.F, ord=2)
        self.ctrl_dim = self.A.shape[0]
        self.V_ctrl = None

    def _solve_controllable_subspace(self, q_scale, ctrl_threshold=None):
        """Solve LQR only in the controllable subspace, lift F back."""
        n = self.A.shape[0]
        m = self.B.shape[1]
        A_np = self.A.numpy()
        B_np = self.B.numpy()

        # Step 1: controllability matrix and SVD
        C_cols = [B_np]
        Ak = A_np.copy()
        for _ in range(n - 1):
            C_cols.append(Ak @ B_np)
            Ak = Ak @ A_np
        C_mat = np.hstack(C_cols)  # (n, n*m)

        U, S, Vt = np.linalg.svd(C_mat, full_matrices=True)
        if ctrl_threshold is None:
            threshold = S.max() * n * np.finfo(A_np.dtype).eps
        else:
            threshold = ctrl_threshold
        ctrl_mask = S > threshold
        k = ctrl_mask.sum()

        print(f"  Controllability: {k}/{n} dimensions (threshold={threshold:.2e})")
        sv_str = ", ".join(f"{v:.4f}" for v in S)
        print(f"  Controllability singular values: [{sv_str}]")

        if k == n:
            print("  Fully controllable — using standard DARE")
            self._solve_full(q_scale)
            return

        # Step 2: split subspaces via U (left singular vectors)
        V_ctrl = torch.from_numpy(U[:, :k].T).to(self.A.dtype)    # (k, n)
        V_unc = torch.from_numpy(U[:, k:].T).to(self.A.dtype)     # (n-k, n)

        # Step 3: verify uncontrollable modes are stable
        A_unc = V_unc @ self.A @ V_unc.T  # (n-k, n-k)
        eigs_unc = torch.linalg.eigvals(A_unc).abs()
        max_unc_eig = eigs_unc.max().item()
        unc_eig_str = ", ".join(f"{v:.4f}" for v in sorted(eigs_unc.tolist()))
        print(f"  Uncontrollable eigenvalue magnitudes: [{unc_eig_str}]")
        if max_unc_eig >= 1.0:
            print(f"  \033[91mWarning: uncontrollable mode has |λ|={max_unc_eig:.4f} ≥ 1 "
                  f"— proceeding anyway but stability not guaranteed\033[0m")

        # Step 4: project A, B, Q onto controllable subspace
        A_ctrl_t = V_ctrl @ self.A @ V_ctrl.T  # (k, k)
        B_ctrl_t = V_ctrl @ self.B              # (k, m)
        A_ctrl = A_ctrl_t.numpy()
        B_ctrl = B_ctrl_t.numpy()
        R_np = self.R.numpy()
        Q_ctrl = (V_ctrl @ self.Q @ V_ctrl.T).numpy()  # (k, k)

        # Step 5: solve DARE in controllable subspace
        P_ctrl = solve_discrete_are(A_ctrl, B_ctrl, Q_ctrl, R_np)
        P_ctrl = torch.from_numpy(P_ctrl).to(self.A.dtype)
        F_ctrl = torch.linalg.solve(
            self.R + torch.from_numpy(B_ctrl).to(self.A.dtype).T @ P_ctrl @ torch.from_numpy(B_ctrl).to(self.A.dtype),
            torch.from_numpy(B_ctrl).to(self.A.dtype).T @ P_ctrl @ torch.from_numpy(A_ctrl).to(self.A.dtype)
        )  # (m, k)

        # Step 6: store certificate quantities from controllable subspace
        self.P = P_ctrl                                        # (k, k)
        self.Q_ctrl = torch.from_numpy(Q_ctrl).to(self.A.dtype)  # (k, k)
        self.ctrl_dim = k
        self.V_ctrl = V_ctrl                                   # (k, n)
        self.V_unc = V_unc

        # Step 7: lift F back to full dimension
        self.F = F_ctrl @ V_ctrl  # (m, k) @ (k, n) = (m, n)
        self.gain_norm = torch.linalg.norm(self.F, ord=2)

        ctrl_eig_str = ", ".join(f"{v:.4f}" for v in sorted(
            torch.linalg.eigvals(torch.from_numpy(A_ctrl).to(self.A.dtype) -
                                 torch.from_numpy(B_ctrl).to(self.A.dtype) @ F_ctrl).abs().tolist()))
        print(f"  Controllable closed-loop |λ|: [{ctrl_eig_str}]")
        print(f"  Full F shape: {self.F.shape}, gain_norm: {self.gain_norm:.4f}")

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
