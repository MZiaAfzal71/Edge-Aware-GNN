"""Ablatable edge-aware and descriptor-fused molecular regressors."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import AttentionalAggregation, GINConv, GINEConv

from .features import ATOM_VOCAB_SIZES, BOND_FEATURE_DIM


class AtomEncoder(nn.Module):
    """Embed atom categories separately instead of treating atomic number as scalar."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size, hidden_dim) for size in ATOM_VOCAB_SIZES])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([embedding(x[:, i]) for i, embedding in enumerate(self.embeddings)]).sum(dim=0)


class EdgeAwareEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        layers: int = 3,
        dropout: float = 0.15,
        edge_aware: bool = True,
    ):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.convolutions = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            convolution = (
                GINEConv(mlp, edge_dim=BOND_FEATURE_DIM, train_eps=True)
                if edge_aware
                else GINConv(mlp, train_eps=True)
            )
            self.convolutions.append(convolution)
            self.normalizations.append(nn.BatchNorm1d(hidden_dim))
        self.readout = AttentionalAggregation(
            gate_nn=nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        )
        self.dropout = dropout
        self.edge_aware = edge_aware

    def forward(self, data) -> torch.Tensor:
        x = self.atom_encoder(data.x)
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            updated = (
                convolution(x, data.edge_index, data.edge_attr)
                if self.edge_aware
                else convolution(x, data.edge_index)
            )
            x = normalization(x + F.dropout(F.relu(updated), p=self.dropout, training=self.training))
        return self.readout(x, data.batch)


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid())

    def forward(self, graph: torch.Tensor, global_: torch.Tensor) -> torch.Tensor:
        weight = self.gate(torch.cat([graph, global_], dim=-1))
        return weight * graph + (1.0 - weight) * global_


class EdgeAwareRegressor(nn.Module):
    """One implementation supports edge-only, concatenation, and gated ablations."""

    def __init__(
        self,
        *,
        global_dim: int = 0,
        fusion: str = "none",
        hidden_dim: int = 128,
        layers: int = 3,
        dropout: float = 0.15,
        edge_aware: bool = True,
    ):
        super().__init__()
        if fusion not in {"none", "concat", "gated"}:
            raise ValueError("fusion must be one of: none, concat, gated")
        if fusion != "none" and global_dim <= 0:
            raise ValueError("Descriptor fusion requires global_dim > 0")
        self.graph_encoder = EdgeAwareEncoder(hidden_dim, layers, dropout, edge_aware=edge_aware)
        self.fusion_name = fusion
        if global_dim > 0:
            self.global_encoder = nn.Sequential(
                nn.Linear(global_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.global_encoder = None
        self.gated_fusion = GatedFusion(hidden_dim) if fusion == "gated" else None
        head_input = 2 * hidden_dim if fusion == "concat" else hidden_dim
        self.head = nn.Sequential(
            nn.Linear(head_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data) -> torch.Tensor:
        graph = self.graph_encoder(data)
        if self.fusion_name == "none":
            representation = graph
        else:
            global_ = self.global_encoder(data.global_features)
            representation = (
                torch.cat([graph, global_], dim=-1)
                if self.fusion_name == "concat"
                else self.gated_fusion(graph, global_)
            )
        return self.head(representation).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
