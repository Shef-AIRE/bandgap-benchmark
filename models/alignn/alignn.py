"""Atomistic Line Graph Neural Network (DGL) adapted for this codebase."""

from typing import Literal, Optional

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling
from torch import nn
from torch.nn import functional as F

from .converter import CrystalGraphConverter, TORCH_DTYPE
from .pyg2dgl import compute_bond_cosines

class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self, in_features: int, out_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


def conditional_grad(context_manager):
    """Decorator mirroring torch.enable_grad usage."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            with context_manager:
                return func(*args, **kwargs)

        return wrapper

    return decorator

class ALIGNN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(
        self,
        atom_input_dim,
        bond_feat_dim,
        num_targets=1,
        alignn_layers=4,
        gcn_layers=4,
        num_gaussians=80,
        triplet_input_features=40,
        embedding_features=64,
        atom_embedding_size=256,
        output_dim=1,
        link="identity",
        regress_forces=False,
        cutoff=6.0,
        readout="mean",
        max_neighbors=50,
        encoding="prop",
        max_num_elements=94,
        ):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.num_targets = num_targets
        self.cutoff = cutoff
        self.regress_forces = regress_forces
        self.max_neighbors = max_neighbors
        self.readout = readout
        self.encoding = encoding
        self.max_num_elements = max_num_elements
        self.graph_converter = CrystalGraphConverter(
            atom_graph_cutoff=cutoff,
            bond_graph_cutoff=cutoff,
            algorithm="legacy",
            on_isolated_atoms="ignore",
        )

        if encoding == "prop":
            self.atom_prop_embed = MLPLayer(atom_input_dim, atom_embedding_size)
            self.atom_num_embed = None
        elif encoding in {"onehot", "z"}:
            self.atom_prop_embed = None
            self.atom_num_embed = nn.Embedding(max_num_elements, atom_embedding_size)
            # normalize alias so downstream checks can look for "z"
            if encoding == "onehot":
                self.encoding = "z"
        else:
            raise ValueError("encoding must be 'prop' or 'z'")

        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=num_gaussians,),
            MLPLayer(bond_feat_dim, embedding_features),
            MLPLayer(embedding_features, atom_embedding_size),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1, vmax=1.0, bins=triplet_input_features,
            ),
            MLPLayer(triplet_input_features, embedding_features),
            MLPLayer(embedding_features, atom_embedding_size),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(atom_embedding_size, atom_embedding_size,)
                for idx in range(alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    atom_embedding_size, atom_embedding_size
                )
                for idx in range(gcn_layers)
            ]
        )

        if output_dim != 0:
            self.num_targets = output_dim

        if self.readout == "mean":
            self.readout = AvgPooling()
        self.fc = nn.Linear(atom_embedding_size, self.num_targets)
        self.link = None
        self.link_name = link
        if link == "identity":
            self.link = lambda x: x
        elif link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif link == "logit":
            self.link = torch.sigmoid

        

    @conditional_grad(torch.enable_grad())
    def _forward(
        self, g
    ):
        """ALIGNN : start with `atom_features`.
        
        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        ##############################
        # CHANGED
        ##############################

        if len(self.alignn_layers) > 0:
            g, lg = g
            lg = lg.local_var()

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))

        g = g.local_var()

        # initial node features: atom feature network...
        if self.encoding == "prop":
            if "atom_features" not in g.ndata:
                raise ValueError("atom_features missing for prop encoding.")
            x = self.atom_prop_embed(g.ndata["atom_features"])
        else:
            if "atom_numbers" not in g.ndata:
                raise ValueError("atom_numbers missing for onehot encoding.")
            atom_nums = g.ndata["atom_numbers"].long() - 1
            atom_nums = torch.clamp(atom_nums, min=0, max=self.max_num_elements - 1)
            x = self.atom_num_embed(atom_nums)

        # initial bond features
        bondlength = g.edata.pop("distances")
        y = self.edge_embedding(bondlength)

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # norm-activation-pool-classify
        h = self.readout(g, x)
        out = self.fc(h)

        if self.link:
            out = self.link(out)

        return torch.squeeze(out)

    def build_dgl_graphs(self, batch):
        structures = getattr(batch, "structures", None)
        if structures is None:
            raise TypeError(
                "ALIGNN expects BatchData to include `structures` for graph conversion."
            )

        device = next(self.parameters()).device
        graphs = []
        line_graphs = []

        for idx, structure in enumerate(structures):
            crystal_graph = self.graph_converter(structure)
            g_local, lg_local = self._graph_from_crystal_graph(
                structure=structure,
                crystal_graph=crystal_graph,
                batch=batch,
                index=idx,
                device=device,
            )
            graphs.append(g_local)
            line_graphs.append(lg_local)

        if not graphs:
            # handle empty batch
            empty = dgl.graph(
                (torch.tensor([], dtype=torch.long, device=device),) * 2,
                num_nodes=0,
                device=device,
            )
            empty_lg = dgl.graph(
                (torch.tensor([], dtype=torch.long, device=device),) * 2,
                num_nodes=0,
                device=device,
            )
            empty_lg.edata["h"] = torch.zeros(0, device=device)
            return empty, empty_lg

        return dgl.batch(graphs).to(device), dgl.batch(line_graphs).to(device)

    def _graph_from_crystal_graph(
        self, structure, crystal_graph, batch, index, device
    ):
        atom_graph = crystal_graph.atom_graph.to(device=device, dtype=torch.long)
        num_atoms = crystal_graph.atomic_number.numel()

        if atom_graph.numel() == 0:
            src = dst = torch.zeros(0, dtype=torch.long, device=device)
        else:
            src = atom_graph[:, 0].long()
            dst = atom_graph[:, 1].long()

        g_local = dgl.graph((src, dst), num_nodes=int(num_atoms), device=device)

        atom_indices = batch.crystal_atom_idx[index]
        if self.encoding == "prop":
            src_tensor = batch.atom_fea
            if src_tensor.device != atom_indices.device:
                atom_indices = atom_indices.to(src_tensor.device)
            atom_features = src_tensor.index_select(0, atom_indices).to(device)
            g_local.ndata["atom_features"] = atom_features
        else:
            src_tensor = batch.atom_num
            if src_tensor.device != atom_indices.device:
                atom_indices = atom_indices.to(src_tensor.device)
            atom_numbers = src_tensor.index_select(0, atom_indices).to(device)
            g_local.ndata["atom_numbers"] = atom_numbers

        coords = torch.as_tensor(
            structure.cart_coords, dtype=TORCH_DTYPE, device=device
        )
        if atom_graph.numel() == 0:
            displacement = torch.zeros((0, 3), dtype=TORCH_DTYPE, device=device)
            distances = torch.zeros(0, dtype=TORCH_DTYPE, device=device)
        else:
            displacement = coords[dst] - coords[src]
            distances = torch.linalg.norm(displacement, dim=-1)

        g_local.edata["r"] = displacement
        g_local.edata["distances"] = distances

        if g_local.num_edges() == 0:
            lg_local = dgl.graph(
                (torch.tensor([], dtype=torch.long, device=device),) * 2,
                num_nodes=0,
                device=device,
            )
            lg_local.edata["h"] = torch.zeros(0, dtype=TORCH_DTYPE, device=device)
            return g_local, lg_local

        lg_local = g_local.line_graph(shared=True)
        if lg_local.num_edges() > 0:
            lg_local.apply_edges(compute_bond_cosines)
        else:
            lg_local.edata["h"] = torch.zeros(0, dtype=TORCH_DTYPE, device=device)

        return g_local, lg_local

    def forward(self, batch):
        g_tuple = self.build_dgl_graphs(batch)
        energy = self._forward(g_tuple)
        return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
