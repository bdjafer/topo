"""SignNet: sign-invariant spectral positional encoding (Lim et al., ICML 2022)."""

import torch
import torch.nn as nn
from torch import Tensor


class SignNet(nn.Module):
    """Sign-invariant encoder for spectral positional encodings.

    Eigenvectors have sign ambiguity: if v is an eigenvector, so is -v.
    SignNet handles this by processing both +v and -v through the same MLP
    and summing the outputs, making the result sign-invariant.

    Each eigenvector gets its own MLP that takes [eigvec_component, eigenvalue]
    as input. The eigenvalue provides structural scale context.
    """

    def __init__(self, k: int = 16, hidden: int = 64, out_per_eig: int = 2):
        super().__init__()
        self.k = k
        self.out_dim = k * out_per_eig  # 16 * 2 = 32

        # Per-eigenvector MLPs: input is [eigvec_component, eigenvalue] = 2d
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_per_eig),
            )
            for _ in range(k)
        ])

    def forward(self, spectral_vecs: Tensor, spectral_vals: Tensor) -> Tensor:
        """
        Args:
            spectral_vecs: [N, k] eigenvector components per node
            spectral_vals: [N, k] eigenvalues (repeated per node)

        Returns:
            [N, k * out_per_eig] sign-invariant spectral encoding
        """
        outputs = []
        for i in range(self.k):
            # [N, 2]: eigenvector component + eigenvalue
            x_pos = torch.stack([spectral_vecs[:, i], spectral_vals[:, i]], dim=-1)
            x_neg = torch.stack([-spectral_vecs[:, i], spectral_vals[:, i]], dim=-1)
            # Sign-invariant: f(+v) + f(-v)
            out = self.mlps[i](x_pos) + self.mlps[i](x_neg)
            outputs.append(out)
        return torch.cat(outputs, dim=-1)  # [N, k * out_per_eig]
