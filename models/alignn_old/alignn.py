from __future__ import annotations

import os
from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter

from models.leftnet.leftnet_z import rbf_emb


def _activation_from_str(name: str) -> nn.Module:
    """Return an activation module from a string identifier."""
    lookup: dict[str, nn.Module] = {
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "elu": nn.ELU(),
    }
    try:
        return lookup[name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported activation '{name}'.") from exc


class ALIGNNLayer(MessagePassing):
    """Simple ALIGNN-style message passing layer with residual connection."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        edge_dim: int,
        activation: str = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__(aggr="add")
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            _activation_from_str(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        update = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        update = self.dropout(update)
        return self.norm(x + update)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)


class ALIGNNModel(nn.Module):
    """Lightweight ALIGNN variant operating on CIF-derived BatchData."""

    def __init__(
        self,
        *,
        atom_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_rbf: int = 32,
        cutoff: float = 6.0,
        dropout: float = 0.0,
        readout: Literal["mean", "sum", "max"] = "mean",
        out_dim: int = 1,
        activation: str = "silu",
        rbf_trainable: bool = False,
        max_neighbors: int = 1000,
        encoding: Literal["prop", "z"] = "prop",
        max_num_elements: int = 94,
    ) -> None:
        super().__init__()
        if num_rbf is None:
            raise ValueError("ALIGNNModel requires num_rbf to be specified.")
        self.cutoff = cutoff
        self.readout = readout
        self.max_neighbors = max_neighbors
        self.encoding = encoding
        self.max_num_elements = max_num_elements
        self._activation = _activation_from_str(activation)

        if self.encoding == "prop":
            self.atom_linear = nn.Linear(atom_input_dim, hidden_dim)
            self.atom_embedding = None
        else:
            self.atom_embedding = nn.Embedding(max_num_elements, hidden_dim)
            self.atom_linear = None

        self.encoder_norm = nn.LayerNorm(hidden_dim)

        self.radial_embedding = rbf_emb(num_rbf, cutoff, rbf_trainable=rbf_trainable)
        self.edge_encoder = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            _activation_from_str(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList(
            ALIGNNLayer(hidden_dim=hidden_dim, edge_dim=hidden_dim, activation=activation, dropout=dropout)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            _activation_from_str(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data, edge_index_override: Tensor | None = None) -> Tensor:
        device = next(self.parameters()).device
        pos = data.positions.to(device)
        batch = data.batch_idx.to(device)

        if self.encoding == "prop":
            atom_fea = data.atom_fea.to(device)
            x = self._activation(self.atom_linear(atom_fea))
        else:
            atom_nums = data.atom_num.to(device).long() - 1
            atom_nums = torch.clamp(atom_nums, min=0, max=self.max_num_elements - 1)
            x = self._activation(self.atom_embedding(atom_nums))

        x = self.encoder_norm(x)

        if edge_index_override is None:
            edge_index = radius_graph(
                pos,
                r=self.cutoff,
                batch=batch,
                max_num_neighbors=self.max_neighbors,
                loop=False,
            )
        else:
            edge_index = edge_index_override.to(device)

        if edge_index.numel() == 0:
            pooled = scatter(x, batch, dim=0, reduce=self.readout)
            return self.head(self.final_norm(pooled))

        j, i = edge_index
        distances = (pos[j] - pos[i]).norm(dim=-1)
        edge_attr = self.edge_encoder(self.radial_embedding(distances))

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        x = self.final_norm(x)
        pooled = scatter(x, batch, dim=0, reduce=self.readout)
        return self.head(pooled)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["ALIGNNModel"]
