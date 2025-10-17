from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.nn import radius_graph

from .composition_model import AtomRef
from .encoders import AngleEncoder, AtomEmbedding, BondEncoder
from .functions import GatedMLP, MLP, find_activation, find_normalization
from .layers import (
    AngleUpdate,
    AtomConv,
    BondConv,
    GraphAttentionReadOut,
    GraphPooling,
)


class CHGNet(nn.Module):
    """CHGNet rebuilt to operate on PyG BatchData batches."""

    def __init__(
        self,
        *,
        atom_fea_dim: int = 64,
        bond_fea_dim: int = 64,
        angle_fea_dim: int = 64,
        composition_model: str | nn.Module | None = None,
        num_radial: int = 31,
        num_angular: int = 31,
        n_conv: int = 4,
        atom_conv_hidden_dim: Sequence[int] | int = 64,
        update_bond: bool = True,
        bond_conv_hidden_dim: Sequence[int] | int = 64,
        update_angle: bool = True,
        angle_layer_hidden_dim: Sequence[int] | int = 0,
        conv_dropout: float = 0,
        read_out: str = "ave",
        mlp_hidden_dims: Sequence[int] | int = (64, 64, 64),
        mlp_dropout: float = 0,
        mlp_first: bool = True,
        is_intensive: bool = True,
        non_linearity: Literal["silu", "relu", "tanh", "gelu"] = "silu",
        atom_graph_cutoff: float = 6,
        bond_graph_cutoff: float = 3,
        cutoff_coeff: int = 8,
        learnable_rbf: bool = True,
        gMLP_norm: str | None = "layer",  # noqa: N803
        readout_norm: str | None = "layer",
        encoding: Literal["z", "prop"] = "z",
        atom_input_dim: int | None = 92,
        max_num_elements: int = 94,
        version: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_args = {
            key: val
            for key, val in locals().items()
            if key not in {"self", "__class__", "kwargs"}
        }
        self.model_args.update(kwargs)
        if version:
            self.model_args["version"] = version

        self.atom_fea_dim = atom_fea_dim
        self.bond_fea_dim = bond_fea_dim
        self.is_intensive = is_intensive
        self.n_conv = n_conv
        self.atom_graph_cutoff = atom_graph_cutoff
        self.bond_graph_cutoff = bond_graph_cutoff
        self.num_radial = num_radial
        self.num_angular = num_angular
        self.encoding = encoding
        self.atom_input_dim = atom_input_dim
        self.max_num_elements = max_num_elements

        if isinstance(composition_model, nn.Module):
            self.composition_model = composition_model
        elif isinstance(composition_model, str):
            self.composition_model = AtomRef(is_intensive=is_intensive)
            self.composition_model.initialize_from(composition_model)
        else:
            self.composition_model = None

        if self.composition_model is not None:
            for param in self.composition_model.parameters():
                param.requires_grad = False

        # Embedding layers
        if encoding == "prop":
            if atom_input_dim is None:
                raise ValueError("atom_input_dim must be specified when encoding='prop'.")
            self.atom_prop_encoder = nn.Sequential(
                nn.Linear(atom_input_dim, atom_fea_dim),
                find_activation(non_linearity),
            )
            self.atom_embedding = None
        else:
            self.atom_embedding = AtomEmbedding(
                atom_feature_dim=atom_fea_dim,
                max_num_elements=max_num_elements,
            )
            self.atom_prop_encoder = None
        self.bond_basis_expansion = BondEncoder(
            atom_graph_cutoff=atom_graph_cutoff,
            bond_graph_cutoff=bond_graph_cutoff,
            num_radial=num_radial,
            cutoff_coeff=cutoff_coeff,
            learnable=learnable_rbf,
        )
        self.bond_embedding = nn.Linear(
            in_features=num_radial, out_features=bond_fea_dim, bias=False
        )
        self.bond_weights_ag = nn.Linear(
            in_features=num_radial, out_features=atom_fea_dim, bias=False
        )
        self.bond_weights_bg = nn.Linear(
            in_features=num_radial, out_features=bond_fea_dim, bias=False
        )
        self.angle_basis_expansion = AngleEncoder(
            num_angular=num_angular, learnable=learnable_rbf
        )
        self.angle_embedding = nn.Linear(
            in_features=num_angular, out_features=angle_fea_dim, bias=False
        )

        # Interaction blocks
        conv_norm = kwargs.pop("conv_norm", None)
        mlp_out_bias = kwargs.pop("mlp_out_bias", False)

        self.atom_conv_layers = nn.ModuleList(
            [
                AtomConv(
                    atom_fea_dim=atom_fea_dim,
                    bond_fea_dim=bond_fea_dim,
                    hidden_dim=atom_conv_hidden_dim,
                    dropout=conv_dropout,
                    activation=non_linearity,
                    norm=conv_norm,
                    gMLP_norm=gMLP_norm,
                    use_mlp_out=True,
                    mlp_out_bias=mlp_out_bias,
                    resnet=True,
                )
                for _ in range(n_conv)
            ]
        )

        if update_bond:
            self.bond_conv_layers = nn.ModuleList(
                [
                    BondConv(
                        atom_fea_dim=atom_fea_dim,
                        bond_fea_dim=bond_fea_dim,
                        angle_fea_dim=angle_fea_dim,
                        hidden_dim=bond_conv_hidden_dim,
                        dropout=conv_dropout,
                        activation=non_linearity,
                        norm=conv_norm,
                        gMLP_norm=gMLP_norm,
                        use_mlp_out=True,
                        mlp_out_bias=mlp_out_bias,
                        resnet=True,
                    )
                    for _ in range(n_conv - 1)
                ]
            )
        else:
            self.bond_conv_layers = [None for _ in range(n_conv - 1)]

        if update_angle:
            self.angle_layers = nn.ModuleList(
                [
                    AngleUpdate(
                        atom_fea_dim=atom_fea_dim,
                        bond_fea_dim=bond_fea_dim,
                        angle_fea_dim=angle_fea_dim,
                        hidden_dim=angle_layer_hidden_dim,
                        dropout=conv_dropout,
                        activation=non_linearity,
                        norm=conv_norm,
                        gMLP_norm=gMLP_norm,
                        resnet=True,
                    )
                    for _ in range(n_conv - 1)
                ]
            )
        else:
            self.angle_layers = [None for _ in range(n_conv - 1)]

        # Readout
        self.site_wise = nn.Linear(atom_fea_dim, 1)
        self.readout_norm = find_normalization(readout_norm, dim=atom_fea_dim)
        self.mlp_first = mlp_first

        if mlp_first:
            self.read_out_type = "sum"
            input_dim = atom_fea_dim
            self.pooling = GraphPooling(average=False)
        elif read_out in {"attn", "weighted"}:
            self.read_out_type = "attn"
            num_heads = kwargs.pop("num_heads", 3)
            self.pooling = GraphAttentionReadOut(
                atom_fea_dim, num_head=num_heads, average=True
            )
            input_dim = atom_fea_dim * num_heads
        else:
            self.read_out_type = "ave"
            input_dim = atom_fea_dim
            self.pooling = GraphPooling(average=True)

        if kwargs.pop("final_mlp", "MLP") in {"normal", "MLP"}:
            self.mlp = MLP(
                input_dim=input_dim,
                hidden_dim=mlp_hidden_dims,
                output_dim=1,
                dropout=mlp_dropout,
                activation=non_linearity,
            )
        else:
            mlp_hidden_dims = (
                (mlp_hidden_dims,)
                if isinstance(mlp_hidden_dims, int)
                else tuple(mlp_hidden_dims)
            )
            self.mlp = nn.Sequential(
                GatedMLP(
                    input_dim=input_dim,
                    hidden_dim=mlp_hidden_dims,
                    output_dim=mlp_hidden_dims[-1],
                    dropout=mlp_dropout,
                    norm=gMLP_norm,
                    activation=non_linearity,
                ),
                nn.Linear(in_features=mlp_hidden_dims[-1], out_features=1),
            )

        version_str = f" v{version}" if version else ""
        print(f"CHGNet{version_str} initialized with {self.n_params:,} parameters")

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        batch,
        *,
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
    ) -> dict[str, Tensor | list[Tensor]]:
        if not hasattr(batch, "positions") or not hasattr(batch, "batch_idx"):
            raise TypeError("CHGNet expects a BatchData object with positions and batch_idx.")

        dtype = self.bond_embedding.weight.dtype
        device = next(self.parameters()).device
        prop_atom_fea = None
        if self.encoding == "prop":
            if not hasattr(batch, "atom_fea"):
                raise TypeError("BatchData must contain atom_fea when encoding='prop'.")
            prop_atom_fea = batch.atom_fea.to(device=device, dtype=dtype)

        simple_graphs, graph = self._assemble_from_batch(batch)

        comp_energy = (
            0 if self.composition_model is None else self.composition_model(simple_graphs)
        )

        prediction = self._compute(
            graph,
            prop_atom_fea=prop_atom_fea,
            return_site_energies=return_site_energies,
            return_atom_feas=return_atom_feas,
            return_crystal_feas=return_crystal_feas,
        )
        prediction["e"] += comp_energy

        if return_site_energies and self.composition_model is not None:
            site_energy_shifts = self.composition_model.get_site_energies(simple_graphs)
            prediction["site_energies"] = [
                i + j for i, j in zip(prediction["site_energies"], site_energy_shifts)
            ]

        return prediction

    def _assemble_from_batch(self, batch):
        device = next(self.parameters()).device
        dtype = self.bond_embedding.weight.dtype

        positions_all = batch.positions.to(device=device, dtype=dtype)
        atom_numbers_all = batch.atom_num.to(device=device, dtype=torch.long)
        batch_idx = batch.batch_idx.to(device=device, dtype=torch.long)
        batch_size = int(batch.batch_size)

        atomic_numbers = []
        bond_bases_ag = []
        bond_bases_bg = []
        angle_bases = []
        atom_graphs = []
        bond_graphs = []
        directed2undirected = []
        atom_owners = []
        simple_graphs = []

        identity_lattice = torch.eye(3, dtype=dtype, device=device)
        atom_offset = 0
        undirected_offset = 0

        for graph_idx in range(batch_size):
            indices = torch.nonzero(batch_idx == graph_idx, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue

            pos_local = positions_all.index_select(0, indices)
            atom_num_local = atom_numbers_all.index_select(0, indices)

            simple_graphs.append(SimpleNamespace(atomic_number=atom_num_local))
            atomic_numbers.append(atom_num_local)
            atom_owners.append(
                torch.full(
                    (indices.numel(),),
                    graph_idx,
                    dtype=torch.long,
                    device=device,
                )
            )

            edge_index = radius_graph(
                pos_local,
                r=self.atom_graph_cutoff,
                loop=False,
                max_num_neighbors=1000,
            ).to(device=device)

            if edge_index.numel() == 0:
                atom_graph_local = torch.empty((0, 2), dtype=torch.long, device=device)
                directed2undirected_local = torch.empty((0,), dtype=torch.long, device=device)
                undirected2directed_local = torch.empty((0,), dtype=torch.long, device=device)
                bond_basis_ag_local = torch.empty(
                    (0, self.num_radial), dtype=dtype, device=device
                )
                bond_basis_bg_local = torch.empty(
                    (0, self.num_radial), dtype=dtype, device=device
                )
                bond_vectors_local = torch.empty((0, 3), dtype=dtype, device=device)
                line_graph_local = torch.empty((0, 5), dtype=torch.long, device=device)
                angle_basis_local = torch.empty(
                    (0, self.num_angular), dtype=dtype, device=device
                )
            else:
                centers = edge_index[1]
                neighbors = edge_index[0]
                atom_graph_local = torch.stack([centers, neighbors], dim=1)

                pair_to_undirected: dict[tuple[int, int], int] = {}
                directed2undirected_local_list: list[int] = []
                undirected2directed_local_list: list[int] = []

                for dir_idx, (center_i, neighbor_j) in enumerate(atom_graph_local.tolist()):
                    key = (center_i, neighbor_j) if center_i <= neighbor_j else (neighbor_j, center_i)
                    if key not in pair_to_undirected:
                        pair_to_undirected[key] = len(pair_to_undirected)
                        undirected2directed_local_list.append(dir_idx)
                    directed2undirected_local_list.append(pair_to_undirected[key])

                directed2undirected_local = torch.tensor(
                    directed2undirected_local_list, dtype=torch.long, device=device
                )
                undirected2directed_local = torch.tensor(
                    undirected2directed_local_list, dtype=torch.long, device=device
                )

                bond_basis_ag_local, bond_basis_bg_local, bond_vectors_local = (
                    self.bond_basis_expansion(
                        center=pos_local.index_select(0, centers),
                        neighbor=pos_local.index_select(0, neighbors),
                        undirected2directed=undirected2directed_local,
                        image=torch.zeros(
                            (atom_graph_local.shape[0], 3),
                            dtype=dtype,
                            device=device,
                        ),
                        lattice=identity_lattice,
                    )
                )

                bond_lengths = torch.norm(
                    pos_local.index_select(0, centers)
                    - pos_local.index_select(0, neighbors),
                    dim=1,
                )
                center_to_dirs: dict[int, list[int]] = {}
                for dir_idx, center in enumerate(centers.tolist()):
                    center_to_dirs.setdefault(center, []).append(dir_idx)

                line_entries: list[list[int]] = []
                for center, dir_indices in center_to_dirs.items():
                    for dir_idx in dir_indices:
                        if bond_lengths[dir_idx] > self.bond_graph_cutoff:
                            continue
                        undirected_left = directed2undirected_local_list[dir_idx]
                        for other_dir in dir_indices:
                            if other_dir == dir_idx:
                                continue
                            if bond_lengths[other_dir] > self.bond_graph_cutoff:
                                continue
                            undirected_right = directed2undirected_local_list[other_dir]
                            line_entries.append(
                                [center, undirected_left, dir_idx, undirected_right, other_dir]
                            )

                if line_entries:
                    line_graph_local = torch.tensor(
                        line_entries, dtype=torch.long, device=device
                    )
                    angle_basis_local = self.angle_basis_expansion(
                        bond_vectors_local.index_select(0, line_graph_local[:, 2]),
                        bond_vectors_local.index_select(0, line_graph_local[:, 4]),
                    )
                else:
                    line_graph_local = torch.empty((0, 5), dtype=torch.long, device=device)
                    angle_basis_local = torch.empty(
                        (0, self.num_angular), dtype=dtype, device=device
                    )

            directed_count = atom_graph_local.shape[0]
            undirected_count = bond_basis_ag_local.shape[0]

            if directed_count > 0:
                atom_graphs.append(atom_graph_local + atom_offset)
                directed2undirected.append(
                    directed2undirected_local + undirected_offset
                )

            if undirected_count > 0:
                bond_bases_ag.append(bond_basis_ag_local)
                bond_bases_bg.append(bond_basis_bg_local)

            if line_graph_local.shape[0] > 0:
                bond_graph_reduced = torch.empty(
                    (line_graph_local.shape[0], 3),
                    dtype=torch.long,
                    device=device,
                )
                bond_graph_reduced[:, 0] = line_graph_local[:, 0] + atom_offset
                bond_graph_reduced[:, 1] = line_graph_local[:, 1] + undirected_offset
                bond_graph_reduced[:, 2] = line_graph_local[:, 3] + undirected_offset
                bond_graphs.append(bond_graph_reduced)
                angle_bases.append(angle_basis_local)

            atom_offset += indices.numel()
            undirected_offset += undirected_count

        if atomic_numbers:
            atomic_numbers = torch.cat(atomic_numbers, dim=0)
        else:
            atomic_numbers = torch.empty((0,), dtype=torch.long, device=device)

        if bond_bases_ag:
            bond_bases_ag = torch.cat(bond_bases_ag, dim=0)
            bond_bases_bg = torch.cat(bond_bases_bg, dim=0)
        else:
            bond_bases_ag = torch.empty(
                (0, self.num_radial), dtype=dtype, device=device
            )
            bond_bases_bg = torch.empty(
                (0, self.num_radial), dtype=dtype, device=device
            )

        if angle_bases:
            angle_bases = torch.cat(angle_bases, dim=0)
        else:
            angle_bases = torch.empty((0, self.num_angular), dtype=dtype, device=device)

        if atom_graphs:
            batched_atom_graph = torch.cat(atom_graphs, dim=0)
        else:
            batched_atom_graph = torch.empty((0, 2), dtype=torch.long, device=device)

        if bond_graphs:
            batched_bond_graph = torch.cat(bond_graphs, dim=0)
        else:
            batched_bond_graph = torch.empty((0, 3), dtype=torch.long, device=device)

        if directed2undirected:
            directed2undirected = torch.cat(directed2undirected, dim=0)
        else:
            directed2undirected = torch.empty((0,), dtype=torch.long, device=device)

        atom_owners = torch.cat(atom_owners, dim=0) if atom_owners else torch.empty(
            (0,), dtype=torch.long, device=device
        )

        graph = SimpleNamespace(
            atomic_numbers=atomic_numbers,
            bond_bases_ag=bond_bases_ag,
            bond_bases_bg=bond_bases_bg,
            angle_bases=angle_bases,
            batched_atom_graph=batched_atom_graph,
            batched_bond_graph=batched_bond_graph,
            atom_owners=atom_owners,
            directed2undirected=directed2undirected,
        )
        return simple_graphs, graph

    def _compute(
        self,
        g,
        *,
        prop_atom_fea: Tensor | None = None,
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
    ) -> dict[str, Tensor | list[Tensor]]:
        prediction: dict[str, Tensor | list[Tensor]] = {}
        atoms_per_graph = torch.bincount(g.atom_owners)
        prediction["atoms_per_graph"] = atoms_per_graph

        dtype = self.bond_embedding.weight.dtype
        device = g.atomic_numbers.device
        if self.encoding == "prop":
            if prop_atom_fea is None:
                raise ValueError("prop_atom_fea must be provided when encoding='prop'.")
            atom_inputs = prop_atom_fea.to(device=device, dtype=dtype)
            atom_feas = self.atom_prop_encoder(atom_inputs)
        else:
            atom_indices = torch.clamp(
                g.atomic_numbers.to(device=device, dtype=torch.long) - 1,
                min=0,
                max=self.max_num_elements - 1,
            )
            atom_feas = self.atom_embedding(atom_indices)

        bond_feas = self.bond_embedding(g.bond_bases_ag)
        bond_weights_ag = self.bond_weights_ag(g.bond_bases_ag)
        bond_weights_bg = self.bond_weights_bg(g.bond_bases_bg)

        angle_feas = (
            self.angle_embedding(g.angle_bases)
            if g.angle_bases.shape[0] != 0
            else g.angle_bases
        )

        for idx, (atom_layer, bond_layer, angle_layer) in enumerate(
            zip(self.atom_conv_layers[:-1], self.bond_conv_layers, self.angle_layers)
        ):
            atom_feas = atom_layer(
                atom_feas=atom_feas,
                bond_feas=bond_feas,
                bond_weights=bond_weights_ag,
                atom_graph=g.batched_atom_graph,
                directed2undirected=g.directed2undirected,
            )

            if g.angle_bases.shape[0] != 0 and bond_layer is not None:
                bond_feas = bond_layer(
                    atom_feas=atom_feas,
                    bond_feas=bond_feas,
                    bond_weights=bond_weights_bg,
                    angle_feas=angle_feas,
                    bond_graph=g.batched_bond_graph,
                )

                if angle_layer is not None and g.batched_bond_graph.shape[0] != 0:
                    angle_feas = angle_layer(
                        atom_feas=atom_feas,
                        bond_feas=bond_feas,
                        angle_feas=angle_feas,
                        bond_graph=g.batched_bond_graph,
                    )

            if idx == self.n_conv - 2:
                if return_atom_feas:
                    prediction["atom_fea"] = torch.split(
                        atom_feas, atoms_per_graph.tolist()
                    )

        atom_feas = self.atom_conv_layers[-1](
            atom_feas=atom_feas,
            bond_feas=bond_feas,
            bond_weights=bond_weights_ag,
            atom_graph=g.batched_atom_graph,
            directed2undirected=g.directed2undirected,
        )

        if self.readout_norm is not None and atom_feas.shape[0] != 0:
            atom_feas = self.readout_norm(atom_feas)

        if self.mlp_first:
            energies = self.mlp(atom_feas)
            energy = self.pooling(energies, g.atom_owners).view(-1)
            if return_site_energies:
                prediction["site_energies"] = torch.split(
                    energies.squeeze(1), atoms_per_graph.tolist()
                )
            if return_crystal_feas:
                prediction["crystal_fea"] = self.pooling(atom_feas, g.atom_owners)
        else:
            crystal_feas = self.pooling(atom_feas, g.atom_owners)
            energy = self.mlp(crystal_feas).view(-1) * atoms_per_graph
            if return_crystal_feas:
                prediction["crystal_fea"] = crystal_feas

        if self.is_intensive and atoms_per_graph.numel() != 0:
            energy = energy / atoms_per_graph

        prediction["e"] = energy
        return prediction
