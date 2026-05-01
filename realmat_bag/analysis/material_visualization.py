"""Material saliency visualization helpers.

This module contains the 3D saliency plotting workflow.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch

def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _first_present(payload: Dict, keys):
    for k in keys:
        if k in payload and payload[k] is not None:
            return payload[k]
    return None

def edge_index_from_nbr_fea_idx(nbr_fea_idx: np.ndarray) -> np.ndarray:
    nbr_fea_idx = np.asarray(nbr_fea_idx)
    N, M = nbr_fea_idx.shape
    src = np.repeat(np.arange(N), M)
    dst = nbr_fea_idx.reshape(-1)
    return np.stack([src, dst], axis=0)

def reduce_node_mask(node_mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Accepts node_mask of shape:
      (N,), (N,1), (N,F) -> returns (N,) importance
    """
    if node_mask is None:
        return None
    nm = np.asarray(node_mask)
    if nm.ndim == 1:
        return nm
    if nm.ndim == 2:
        # common: (N, F) -> aggregate feature dimension
        return np.abs(nm).sum(axis=1)
    raise ValueError(f"Unsupported node_mask shape: {nm.shape}")

def reduce_edge_mask(edge_mask: np.ndarray, edge_index: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Accepts edge_mask of shape:
      (E,), (E,1), (E,F) -> returns (E,)
    If edge_index is None, returns a reduced vector anyway.
    """
    if edge_mask is None:
        return None
    em = np.asarray(edge_mask)
    if em.ndim == 1:
        out = em
    elif em.ndim == 2:
        out = np.abs(em).sum(axis=1)
    else:
        raise ValueError(f"Unsupported edge_mask shape: {em.shape}")

    # If we know E from edge_index, trim to match
    if edge_index is not None:
        E = int(np.asarray(edge_index).shape[1])
        if out.shape[0] != E:
            out = out[: min(E, out.shape[0])]
    return out

def load_payload(pt_path: Path) -> Dict:
    return torch.load(pt_path, map_location="cpu")

def extract_saliency_fields(payload: Dict):
    # positions
    pos = _first_present(payload, ["pos", "positions"])
    pos = to_numpy(pos)
    if pos is None:
        raise KeyError("Missing pos/positions")

    # node scores
    node_mask = _first_present(payload, ["node_mask_mean", "node_mask"])
    node_scores = reduce_node_mask(to_numpy(node_mask)) if node_mask is not None else None

    # edge index
    edge_index = _first_present(payload, ["edge_index"])
    nbr_fea_idx = _first_present(payload, ["nbr_fea_idx"])
    edge_index = to_numpy(edge_index) if edge_index is not None else None
    if edge_index is None and nbr_fea_idx is not None:
        edge_index = edge_index_from_nbr_fea_idx(to_numpy(nbr_fea_idx))

    if edge_index is not None:
        edge_index = np.asarray(edge_index)
        if edge_index.ndim != 2:
            raise ValueError(f"edge_index must be 2D, got {edge_index.shape}")
        if edge_index.shape[0] == 2:
            pass
        elif edge_index.shape[1] == 2:
            edge_index = edge_index.T
        else:
            raise ValueError(f"edge_index must be [2, E] or [E, 2], got {edge_index.shape}")

    # edge scores
    edge_mask = _first_present(payload, ["edge_mask_mean", "edge_mask"])
    edge_scores = reduce_edge_mask(to_numpy(edge_mask), edge_index=edge_index) if edge_mask is not None else None

    # Align edge_scores length with edge_index if they mismatch
    if edge_scores is not None and edge_index is not None:
        e_len = edge_index.shape[1]
        if edge_scores.shape[0] != e_len:
            m = min(e_len, edge_scores.shape[0])
            edge_scores = edge_scores[:m]
            edge_index = edge_index[:, :m]

    # pred/gt
    pred = _first_present(payload, ["prediction_mean", "prediction"])
    target = _first_present(payload, ["target"])

    pred = float(pred) if pred is not None else None
    target = float(target) if target is not None else None
    return pos, node_scores, edge_index, edge_scores, target, pred


def extract_graph_fields(payload: Dict):
    """Infer structure positions, graph edges, and optional node scores."""
    pos = to_numpy(_first_present(payload, ["pos", "positions"]))
    if pos is None:
        raise KeyError("Missing pos/positions in payload")

    edge_index = _first_present(payload, ["edge_index"])
    nbr_fea_idx = _first_present(payload, ["nbr_fea_idx"])
    edge_index = to_numpy(edge_index) if edge_index is not None else None
    if edge_index is None and nbr_fea_idx is not None:
        edge_index = edge_index_from_nbr_fea_idx(to_numpy(nbr_fea_idx))

    if edge_index is not None:
        edge_index = np.asarray(edge_index)
        if edge_index.ndim != 2:
            raise ValueError(f"edge_index must be 2D, got {edge_index.shape}")
        if edge_index.shape[0] == 2:
            pass
        elif edge_index.shape[1] == 2:
            edge_index = edge_index.T
        else:
            raise ValueError(f"edge_index must be [2,E] or [E,2], got {edge_index.shape}")

    node_mask = _first_present(payload, ["node_mask_mean", "node_mask"])
    node_scores = reduce_node_mask(to_numpy(node_mask)) if node_mask is not None else None
    return pos, edge_index, node_scores


def extract_elements_from_payload(payload: Dict, N: int) -> Optional[List[str]]:
    """Infer element symbols from payload fields when available."""
    elems = _first_present(payload, ["elements", "species", "elem", "symbols"])
    if elems is not None:
        elems = to_numpy(elems)
        elems_list = [str(x) for x in list(elems)]
        return elems_list if len(elems_list) == N else None

    z = _first_present(payload, ["atomic_numbers", "z", "atom_types", "atom_numbers"])
    if z is None:
        return None
    z = to_numpy(z).astype(int).reshape(-1)
    if z.shape[0] != N:
        return None

    if not _HAVE_PYMATGEN:
        periodic = {
            1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 12: "Mg",
            13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 19: "K",
            20: "Ca", 22: "Ti", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co",
            28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As",
            34: "Se", 35: "Br", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
            41: "Nb", 42: "Mo", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag",
            48: "Cd", 49: "In", 50: "Sn", 51: "Sb", 52: "Te", 53: "I",
            55: "Cs", 56: "Ba", 57: "La", 72: "Hf", 73: "Ta", 74: "W",
            78: "Pt", 79: "Au", 80: "Hg", 82: "Pb", 83: "Bi",
        }
        return [periodic.get(int(zi), f"Z{int(zi)}") for zi in z]

    from pymatgen.core.periodic_table import Element

    return [Element.from_Z(int(zi)).symbol for zi in z]


def load_structure_from_cif(cif_path: Path) -> Tuple["Structure", np.ndarray, List[str]]:
    """Load a CIF and return the structure, cartesian positions, and element labels."""
    if not _HAVE_PYMATGEN:
        raise RuntimeError("pymatgen not available; cannot parse CIF.")
    structure = Structure.from_file(str(cif_path))
    pos = np.asarray(structure.cart_coords, dtype=float)
    elems = [site.specie.symbol for site in structure.sites]
    return structure, pos, elems


def build_element_color_map(elements: List[str]) -> Dict[str, Tuple[float, float, float, float]]:
    """Create a deterministic element-to-color map."""
    uniq = sorted(set(elements))
    cmap = mpl.cm.get_cmap("tab20")
    return {el: cmap(i % cmap.N) for i, el in enumerate(uniq)}


def extract_crystalnn_bonds(
    s: "Structure" | None = None,
    cif_path: str | Path | None = None,
    max_bonds_per_site: Optional[int] = None,
    max_dist: float = 3.20,
    **_kwargs,
) -> np.ndarray:
    """Return undirected CrystalNN bonds as an edge_index array with shape [2, E]."""
    if not _HAVE_PYMATGEN:
        raise RuntimeError("pymatgen not available; cannot run CrystalNN.")
    if s is None:
        if cif_path is None:
            raise ValueError("Either s or cif_path must be provided.")
        s = Structure.from_file(str(cif_path))

    from pymatgen.analysis.local_env import CrystalNN

    cnn = CrystalNN()
    pairs = {}
    for i in range(len(s)):
        neighs = cnn.get_nn_info(s, i)
        if max_bonds_per_site is not None and len(neighs) > max_bonds_per_site:
            neighs = sorted(
                neighs,
                key=lambda x: s.get_distance(i, int(x["site_index"])),
            )[:max_bonds_per_site]

        for info in neighs:
            j = int(info["site_index"])
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            dist = float(s.get_distance(a, b))
            if dist > max_dist:
                continue
            if (a, b) not in pairs or dist < pairs[(a, b)]:
                pairs[(a, b)] = dist

    if not pairs:
        return np.zeros((2, 0), dtype=int)
    return np.array(list(pairs.keys()), dtype=int).T

# optional CIF parser helper used by structure panels
try:
    from pymatgen.core import Structure
    _HAVE_PYMATGEN = True
except Exception:
    _HAVE_PYMATGEN = False

def extract_elements_from_cif(cif_path: Path) -> Tuple[np.ndarray, List[str]]:
    if not _HAVE_PYMATGEN:
        raise RuntimeError("pymatgen is required for extract_elements_from_cif; install pymatgen first.")
    s = Structure.from_file(str(cif_path))
    pos = np.asarray(s.cart_coords, dtype=float)
    elems = [str(site.specie.symbol) for site in s.sites]
    return pos, elems


def style_light_3d_axis(ax):
    ax.set_facecolor("white")

    pane_rgba = (0.96, 0.96, 0.96, 0.18)
    edge_rgba = (0.20, 0.20, 0.20, 0.20)

    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor(pane_rgba)
        a.pane.set_edgecolor(edge_rgba)

    ax.grid(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])



def _as_edge_cross_mask(edge_cross_lattice, E):
    if edge_cross_lattice is None:
        return None
    ec = edge_cross_lattice.detach().cpu().numpy() if torch.is_tensor(edge_cross_lattice) else np.asarray(edge_cross_lattice)
    ec = np.asarray(ec).reshape(-1).astype(bool)
    if ec.shape[0] < E:
        return None
    return ec[:E]


def draw_saliency_panel_3d(
    ax,
    pos,
    node_scores,
    edge_index,
    edge_scores,
    edge_cross_lattice=None,
    show_cross_lattice=False,
    node_cmap="viridis",
    edge_cmap="plasma",
    node_norm=None,
    edge_norm=None,
    title="",
    subtitle="",
    max_edges=2000,
    topk_edges=80,
    bond_alpha=0.85,
    bond_lw=2.0,
    node_size=30,
):
    """
    Draw a 3D saliency panel with nodes and edges colored by importance.
    """
    xyz = np.asarray(pos)[:, :3]

    # ---- edges ----
    if edge_index is not None and edge_scores is not None:
        E = min(edge_index.shape[1], len(edge_scores))
        ei = np.asarray(edge_index)[:, :E]
        es = np.asarray(edge_scores)[:E]
        ec = _as_edge_cross_mask(edge_cross_lattice, E)

        # default behavior: only show intra-cell edges
        if ec is not None and not show_cross_lattice:
            keep = ~ec
            ei = ei[:, keep]
            es = es[keep]
            ec = ec[keep]

        if topk_edges is not None and es.shape[0] > topk_edges:
            sel = np.argsort(np.abs(es))[-topk_edges:]
            ei = ei[:, sel]
            es = es[sel]
            if ec is not None:
                ec = ec[sel]

        order = np.argsort(np.abs(es))
        ei = ei[:, order]
        es = es[order]
        if ec is not None:
            ec = ec[order]

        if ei.shape[1] > max_edges:
            idx = np.linspace(0, ei.shape[1] - 1, max_edges).astype(int)
            ei = ei[:, idx]
            es = es[idx]
            if ec is not None:
                ec = ec[idx]

        if edge_norm is None:
            evmin, evmax = (float(es.min()), float(es.max())) if es.size else (0.0, 1.0)
            edge_norm_local = mpl.colors.Normalize(vmin=evmin, vmax=evmax, clip=True)
        else:
            edge_norm_local = edge_norm

        colors = mpl.cm.get_cmap(edge_cmap)(edge_norm_local(es))

        for k in range(ei.shape[1]):
            si, di = int(ei[0, k]), int(ei[1, k])
            is_cross = bool(ec[k]) if ec is not None else False
            if is_cross and show_cross_lattice:
                color = (0.35, 0.35, 0.35, 0.65)
                ls = "--"
                lw = max(0.8, bond_lw * 0.9)
                alpha = 0.9
            else:
                color = colors[k]
                ls = "-"
                lw = bond_lw
                alpha = bond_alpha

            ax.plot(
                [xyz[si, 0], xyz[di, 0]],
                [xyz[si, 1], xyz[di, 1]],
                [xyz[si, 2], xyz[di, 2]],
                color=color,
                linestyle=ls,
                linewidth=lw,
                alpha=alpha,
                zorder=1,
            )

    # ---- nodes ----
    if node_scores is None:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=node_size, c="black", zorder=2)
    else:
        ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            s=node_size,
            c=np.asarray(node_scores),
            cmap=node_cmap,
            norm=node_norm,
            edgecolors="k",
            linewidths=0.25,
            zorder=2,
        )

    ax.set_title(title, fontsize=9)
    if subtitle:
        ax.text2D(0.02, 0.98, subtitle, transform=ax.transAxes,
                  ha="left", va="top", fontsize=8)

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=18, azim=35)
    style_light_3d_axis(ax)


def render_method_encoding_3d(
    fig,
    subspec,
    pt_root: str | Path,
    mpid: str,
    arch: str,
    pattern: str = "{model}_{mpid}_saliency_ens10.pt",
    topk_edges: int = 25,
    node_norm=None,
    edge_norm=None,
    node_cmap="viridis",
    edge_cmap="plasma",
    show_cross_lattice: bool = False,
):
    pt_root = Path(pt_root)
    variants = ["z", "prop"]

    def _resolve_pt(arch, v):
        model = f"{arch}_{v}"
        pt_path = pt_root / pattern.format(model=model, mpid=mpid)
        if not pt_path.exists():
            alt = pt_root / f"{model}_{mpid}_gnnexplainer_ens10.pt"
            if alt.exists():
                pt_path = alt
            else:
                raise FileNotFoundError(f"Missing: {pt_path} (and fallback not found)")
        return pt_path

    parsed = {}
    for v in variants:
        payload = load_payload(_resolve_pt(arch, v))
        parsed[v] = (extract_saliency_fields(payload), payload)

    gs_inner = mpl.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=subspec, wspace=0.10)
    ax_z = fig.add_subplot(gs_inner[0, 0], projection="3d")
    ax_p = fig.add_subplot(gs_inner[0, 1], projection="3d")

    # z
    (pos, node_scores, edge_index, edge_scores, gt, pred), payload_z = parsed["z"]
    subtitle = ""
    if gt is not None:
        subtitle += f"GT={gt:.2f}\n"
    if pred is not None:
        subtitle += f"Pred={pred:.2f}"
    draw_saliency_panel_3d(
        ax_z, pos, node_scores, edge_index, edge_scores,
        edge_cross_lattice=payload_z.get("edge_is_cross_lattice", None),
        show_cross_lattice=show_cross_lattice,
        node_cmap=node_cmap,
        edge_cmap=edge_cmap,
        node_norm=node_norm, edge_norm=edge_norm,
        title=f"{arch.title()} (z)", subtitle=subtitle, topk_edges=topk_edges
    )

    # prop
    (pos, node_scores, edge_index, edge_scores, gt, pred), payload_p = parsed["prop"]
    subtitle = ""
    if gt is not None:
        subtitle += f"GT={gt:.2f}\n"
    if pred is not None:
        subtitle += f"Pred={pred:.2f}"
    draw_saliency_panel_3d(
        ax_p, pos, node_scores, edge_index, edge_scores,
        edge_cross_lattice=payload_p.get("edge_is_cross_lattice", None),
        show_cross_lattice=show_cross_lattice,
        node_cmap=node_cmap,
        edge_cmap=edge_cmap,
        node_norm=node_norm, edge_norm=edge_norm,
        title=f"{arch.title()} (prop)", subtitle=subtitle, topk_edges=topk_edges
    )

    return ax_z, ax_p


def draw_structure_3d_on_axis(
    ax,
    pt_path: str | Path,
    cif_path: str | Path,
    label_mode: str = "topk",
    label_topk: int = 40,
    max_bonds_per_site: int | None = 6,
    max_dist: float = 3.0,
):
    """
    3D structure panel:
    - Atom positions come from payload pos
    - Elements from payload (or CIF fallback)
    - Bonds come from CrystalNN and are drawn on payload positions.
    """
    payload = torch.load(Path(pt_path), map_location="cpu")
    pos, _, _ = extract_graph_fields(payload)
    N = pos.shape[0]

    elements = extract_elements_from_payload(payload, N)
    if elements is None:
        _, elements = extract_elements_from_cif(Path(cif_path))
        if len(elements) != N:
            raise RuntimeError(f"CIF atom count ({len(elements)}) != payload atom count ({N}).")

    color_map = build_element_color_map(elements)
    core_colors = np.array([color_map[e] for e in elements])

    s_cif = Structure.from_file(str(cif_path))

    ei_cn = extract_crystalnn_bonds(
        s=s_cif,
        max_bonds_per_site=max_bonds_per_site,
        max_dist=max_dist,
    )
    xyz = np.asarray(pos)[:, :3]

    # bonds (gray)
    if ei_cn is not None and ei_cn.shape[1] > 0:
        ei = np.asarray(ei_cn)
        maxE = 3000
        if ei.shape[1] > maxE:
            idx = np.linspace(0, ei.shape[1] - 1, maxE).astype(int)
            ei = ei[:, idx]
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            ax.plot(
                [xyz[i, 0], xyz[j, 0]],
                [xyz[i, 1], xyz[j, 1]],
                [xyz[i, 2], xyz[j, 2]],
                color=(0, 0, 0, 0.25),
                linewidth=0.7,
                zorder=1,
            )

    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               s=32, c=core_colors, edgecolors="k", linewidths=0.25, zorder=2)

    # labels
    if label_mode != "none":
        if label_mode == "topk":
            # no saliency here; label a fixed subset
            idx = list(range(min(label_topk, N)))
        else:
            idx = list(range(N))
        for i in idx:
            ax.text(xyz[i, 0], xyz[i, 1], xyz[i, 2], f"{elements[i]}", fontsize=7, zorder=3)

    ax.set_title("3D structure", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    style_light_3d_axis(ax)
    ax.view_init(elev=18, azim=35)


def compose_encoding_saliency_figure(
    out_path: str | Path,
    pt_root: str | Path,
    mpid: str,
    structure_pt: str | Path,
    structure_cif: str | Path,
    architectures: list[str],              # len=3
    pattern: str = "{model}_{mpid}_saliency_ens10.pt",
    topk_edges: int = 25,
    quantile: float = 0.05,
    dpi: int = 200,
):
    if len(architectures) != 3:
        raise ValueError("architectures must have length 3 (method1/method2/method3).")

    pt_root = Path(pt_root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- global norms across all method panels ----
    node_vals_all = []
    edge_vals_all = []

    def _resolve_pt(arch, v):
        model = f"{arch}_{v}"
        pt_path = pt_root / pattern.format(model=model, mpid=mpid)
        if not pt_path.exists():
            alt = pt_root / f"{model}_{mpid}_gnnexplainer_ens10.pt"
            if alt.exists():
                pt_path = alt
            else:
                raise FileNotFoundError(f"Missing: {pt_path} (and fallback not found)")
        return pt_path

    for arch in architectures:
        for v in ["z", "prop"]:
            payload = load_payload(_resolve_pt(arch, v))
            _, node_scores, _, edge_scores, _, _ = extract_saliency_fields(payload)
            if node_scores is not None:
                node_vals_all.append(np.asarray(node_scores))
            if edge_scores is not None:
                edge_vals_all.append(np.asarray(edge_scores))

    node_vals_all = np.concatenate(node_vals_all) if node_vals_all else np.array([0.0])
    edge_vals_all = np.concatenate(edge_vals_all) if edge_vals_all else np.array([0.0])

    if node_vals_all.size > 1:
        nvmin, nvmax = np.quantile(node_vals_all, quantile), np.quantile(node_vals_all, 1 - quantile)
    else:
        nvmin = nvmax = float(node_vals_all[0])

    if edge_vals_all.size > 1:
        evmin, evmax = np.quantile(edge_vals_all, quantile), np.quantile(edge_vals_all, 1 - quantile)
    else:
        evmin = evmax = float(edge_vals_all[0])

    node_norm = mpl.colors.Normalize(vmin=float(nvmin), vmax=float(nvmax), clip=True)
    edge_norm = mpl.colors.Normalize(vmin=float(evmin), vmax=float(evmax), clip=True)

    node_sm = mpl.cm.ScalarMappable(norm=node_norm, cmap="viridis")
    edge_sm = mpl.cm.ScalarMappable(norm=edge_norm, cmap="plasma")

    # ---- layout (same geometry, but 3D axes) ----
    fig = plt.figure(figsize=(13.5, 8.8), dpi=dpi)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.05], wspace=0.22, hspace=0.25)


    ax00 = fig.add_subplot(gs[0, 0], projection="3d")

    # (0,0) structure 3D
    draw_structure_3d_on_axis(
        ax00,
        pt_path=structure_pt,
        cif_path=structure_cif,
        label_mode="topk",
        label_topk=40,
        max_bonds_per_site=6,
        max_dist=3.0,
    )

    # three method cells (each is 1x2 in 3D)
    render_method_encoding_3d(
        fig, gs[0, 1],
        pt_root=pt_root, mpid=mpid, arch=architectures[0],
        pattern=pattern, topk_edges=topk_edges,
        node_norm=node_norm, edge_norm=edge_norm
    )
    render_method_encoding_3d(
        fig, gs[1, 0],
        pt_root=pt_root, mpid=mpid, arch=architectures[1],
        pattern=pattern, topk_edges=topk_edges,
        node_norm=node_norm, edge_norm=edge_norm
    )
    render_method_encoding_3d(
        fig, gs[1, 1],
        pt_root=pt_root, mpid=mpid, arch=architectures[2],
        pattern=pattern, topk_edges=topk_edges,
        node_norm=node_norm, edge_norm=edge_norm
    )

    # colorbars
    # ---- compact colorbars: same height as one row of subplots ----
    # Put colorbars only in the bottom-row right cell (gs[1,2]) so they don't span both rows.
    gs_cb = mpl.gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=gs[1, 2],   # <-- key change: NOT gs[:,2]
        wspace=0.8
    )
    cax_node = fig.add_subplot(gs_cb[0, 0])
    cax_edge = fig.add_subplot(gs_cb[0, 1])

    cb1 = fig.colorbar(node_sm, cax=cax_node, shrink=0.95)
    cb1.ax.tick_params(labelsize=7)
    cb1.ax.yaxis.set_ticks_position("left")
    cb1.ax.yaxis.set_label_position("left")
    cb1.set_label("Node", fontsize=8)

    cb2 = fig.colorbar(edge_sm, cax=cax_edge, shrink=0.95)
    cb2.ax.tick_params(labelsize=7)
    cb2.set_label("Edge", fontsize=8)



    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.show()
    print(f"Saved -> {out_path} and {out_path.with_suffix('.pdf')}")


__all__ = [
    "compose_encoding_saliency_figure",
    "draw_saliency_panel_3d",
    "draw_structure_3d_on_axis",
    "extract_graph_fields",
    "extract_saliency_fields",
    "load_payload",
    "render_method_encoding_3d",
]
