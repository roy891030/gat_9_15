# -*- coding: utf-8 -*-
"""Unused experimental DMFM_Lite variant.

This module is intentionally kept outside the main model file because the main
pipeline does not construct DMFM_Lite. It is not covered by the current
training/evaluation path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class DMFM_Lite(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 32,
        heads: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim

        self.batch_norm = nn.BatchNorm1d(num_features)
        self.encoder = nn.Linear(num_features, hidden_dim)

        self.gat_industry = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)
        self.gat_universe = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)

        self.factor_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.factor_attention = nn.Linear(num_features, num_features)

    def forward(self, x, industry_edge_index, universe_edge_index):
        x_norm = self.batch_norm(x)
        C = F.relu(self.encoder(x_norm))

        H_I = F.elu(self.gat_industry(C, industry_edge_index))
        C_I = C - H_I

        H_U = F.elu(self.gat_universe(C_I, universe_edge_index))
        C_U = C_I - H_U

        hierarchical_features = torch.cat([C, C_I, C_U], dim=-1)
        deep_factor = self.factor_decoder(hierarchical_features)

        U = F.leaky_relu(self.factor_attention(x), negative_slope=0.2)
        attn_weights = F.softmax(U, dim=-1)

        contexts = {"C": C, "C_I": C_I, "C_U": C_U, "H_I": H_I, "H_U": H_U}
        return deep_factor, attn_weights, contexts
