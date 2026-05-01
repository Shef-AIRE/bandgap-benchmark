#!/usr/bin/env python3
"""
Grad×Input saliency across fold checkpoints.
- Uses model(return_features=True) to expose node/edge reps (except CGCNN which uses input tensors).
- Averages saliency over top-K checkpoints per model.

Example:
    python -m realmat_bag.analysis.gradient_saliency \
        --models all \
        --mpids mp-5045 mp-570887 \
        --checkpoint-root saved_models/saved_models/finetune \
        --output-dir analysis_outputs/saliency
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

DEFAULT_CFG_MAP = {
    "alignn_prop": "configs/finetune/alignn/alignn_prop.yaml",
    "alignn_z": "configs/finetune/alignn/alignn_z.yaml",
    "cartnet_z": "configs/finetune/cartnet/cartnet_z.yaml",
    "cartnet_prop": "configs/finetune/cartnet/cartnet_prop.yaml",
    "cgcnn": "configs/finetune/cgcnn/cgcnn.yaml",
    "chgnet_z": "configs/finetune/chgnet/chgnet_z.yaml",
    "chgnet_prop": "configs/finetune/chgnet/chgnet_prop.yaml",
    "leftnet_prop": "configs/finetune/leftnet/leftnet_prop.yaml",
    "leftnet_z": "configs/finetune/leftnet/leftnet_z.yaml",
}

DEFAULT_MODEL_ORDER = list(DEFAULT_CFG_MAP)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Grad×Input saliency across fold checkpoints.")
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Separate model names by space, or 'all' for every default GNN model.",
    )
    ap.add_argument("--mpids", nargs="+", required=True, help="Space-separated MPIDs to process.")
    ap.add_argument("--cfg-map", nargs="*", default=[], help="Override map entries like name=path.yaml.")
    ap.add_argument("--checkpoint-root", default="saved_models/saved_models/finetune")
    ap.add_argument("--data-file", default="data/fine_tune/test_data.json")
    ap.add_argument("--output-dir", default="analysis_outputs/saliency")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-folds", type=int, default=10, help="Top-K checkpoints (by val_mre).")
    ap.add_argument("--fold-pattern", default="*-best-mre-epoch=*-val_mre=*.ckpt")
    ap.add_argument("--save-per-fold", action="store_true")
    return ap.parse_args()


def expand_cfg_map(overrides: List[str]) -> Dict[str, str]:
    mapping = DEFAULT_CFG_MAP.copy()
    for item in overrides:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        mapping[k.strip()] = v.strip()
    return mapping


def expand_models(models: List[str], cfg_map: Dict[str, str]) -> List[str]:
    if len(models) == 1 and models[0].lower() == "all":
        return [name for name in DEFAULT_MODEL_ORDER if name in cfg_map]
    return models


def list_best_mre_checkpoints(folder: Path, pattern: str, k: int) -> List[Path]:
    cands = sorted(folder.glob(pattern))
    if not cands:
        raise FileNotFoundError(f"No checkpoints matching {pattern} under {folder}")

    def parse_mre(p: Path) -> float:
        return float(p.stem.split("val_mre=")[-1])

    return sorted(cands, key=parse_mre)[:k]


def build_trainer(cfg_path: Path, checkpoint: Path, device: torch.device, cfg_override=None):
    from config import get_cfg_defaults
    from main import get_model

    cfg = cfg_override if cfg_override is not None else get_cfg_defaults()
    if cfg_override is None:
        cfg.merge_from_file(str(cfg_path))
        cfg.freeze()
    trainer = get_model(cfg)
    ckpt_obj = torch.load(checkpoint, map_location="cpu")
    state_dict = ckpt_obj.get("state_dict", ckpt_obj)
    if any(k.startswith("model.") for k in state_dict):
        state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}
    trainer.model.load_state_dict(state_dict, strict=False)
    trainer.model.to(device)
    trainer.model.eval()
    return trainer.model, cfg


def subset_dataset(mpids: List[str], cfg, data_file: Path):
    from main import load_json_as_dataframe
    from realmat_bag.loaddata.cifdata import CIFData

    df = load_json_as_dataframe(str(data_file))
    df = df[df["mpids"].isin(mpids)]
    if df.empty:
        raise ValueError(f"No matching MPIDs found in {data_file}.")
    return CIFData(
        df[["mpids", "bg"]],
        cfg.MODEL.CIF_FOLDER,
        cfg.MODEL.INIT_FILE,
        cfg.MODEL.MAX_NBRS,
        cfg.MODEL.RADIUS,
        randomize=False,
    )


def uses_input_feature_saliency(model_name: str) -> bool:
    """Return True for models that need saliency on raw input tensors."""
    return model_name.split("_", 1)[0].lower() == "cgcnn"


def input_feature_grad_x_input(model, batch, device):
    """Compute Grad×Input saliency from atom and neighbor input features."""
    model.zero_grad(set_to_none=True)

    required = ["atom_fea", "nbr_fea", "nbr_fea_idx", "crystal_atom_idx"]
    missing = [k for k in required if not hasattr(batch, k)]
    if missing:
        raise AttributeError(
            f"Input-feature saliency batch missing fields {missing}. "
            f"Available: {list(vars(batch).keys())}"
        )

    atom_fea = batch.atom_fea.detach().clone().to(device).requires_grad_(True)
    nbr_fea = batch.nbr_fea.detach().clone().to(device).requires_grad_(True)
    nbr_fea_idx = batch.nbr_fea_idx.to(device)
    crystal_atom_idx = batch.crystal_atom_idx
    if isinstance(crystal_atom_idx, torch.Tensor):
        crystal_atom_idx = [crystal_atom_idx]
    if not isinstance(crystal_atom_idx, list):
        raise TypeError(
            f"Expected crystal_atom_idx to be list[Tensor]; got {type(crystal_atom_idx)}"
        )
    crystal_atom_idx = [c.to(device) for c in crystal_atom_idx]

    class SimpleBatch:
        pass

    sb = SimpleBatch()
    sb.atom_fea = atom_fea
    sb.nbr_fea = nbr_fea
    sb.nbr_fea_idx = nbr_fea_idx
    sb.crystal_atom_idx = crystal_atom_idx

    pred = model(sb)
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    pred.sum().backward()
    node_imp = (atom_fea.grad * atom_fea).abs().sum(dim=-1).detach().cpu()
    edge_imp = (nbr_fea.grad * nbr_fea).abs().sum(dim=-1).detach().cpu()
    return pred.detach().cpu(), node_imp, edge_imp, nbr_fea_idx.detach().cpu()


def _edge_periodic_info_from_structure(structure, edge_index):
    """
    Compute periodic image info for each edge in edge_index.
    Returns:
      cross_mask: BoolTensor [E], True if edge crosses lattice image
      jimage: LongTensor [E, 3], periodic image offset for destination atom
    """
    if edge_index is None:
        return None, None

    ei = edge_index.detach().cpu() if torch.is_tensor(edge_index) else torch.as_tensor(edge_index)
    if ei.ndim != 2:
        return None, None
    if ei.shape[0] != 2 and ei.shape[1] == 2:
        ei = ei.t()
    if ei.shape[0] != 2:
        return None, None

    frac = structure.frac_coords
    lattice = structure.lattice
    E = int(ei.shape[1])
    cross = torch.zeros(E, dtype=torch.bool)
    jimg = torch.zeros((E, 3), dtype=torch.long)

    for k in range(E):
        src = int(ei[0, k].item())
        dst = int(ei[1, k].item())
        try:
            _dist, image = lattice.get_distance_and_image(frac[src], frac[dst])
            image = tuple(int(round(v)) for v in image)
        except Exception:
            image = (0, 0, 0)
        jimg[k] = torch.tensor(image, dtype=torch.long)
        cross[k] = not (image[0] == 0 and image[1] == 0 and image[2] == 0)

    return cross, jimg


def main():
    args = parse_args()

    from config import get_cfg_defaults
    from main import set_random_seed
    from realmat_bag.loaddata.collate import collate_pool_leftnet

    set_random_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg_map = expand_cfg_map(args.cfg_map)
    model_names = expand_models(args.models, cfg_map)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mpid_limit = args.mpids
    for model_name in model_names:
        cfg_path = Path(cfg_map[model_name])
        ckpt_dir = Path(args.checkpoint_root) / model_name
        checkpoints = list_best_mre_checkpoints(ckpt_dir, args.fold_pattern, args.num_folds)
        # Load cfg; for CGCNN adjust feature dims from a sample to match checkpoint expectations.
        cfg = get_cfg_defaults()
        cfg.merge_from_file(str(cfg_path))
        cfg.freeze()

        dataset = subset_dataset(mpid_limit, cfg, Path(args.data_file))
        use_input_feature_saliency = uses_input_feature_saliency(model_name)

        if use_input_feature_saliency:
            sample = dataset[0]
            cfg = cfg.clone()
            cfg.defrost()
            cfg.CGCNN.ORIG_ATOM_FEA_LEN = sample.atom_fea.shape[-1]
            cfg.CGCNN.NBR_FEA_LEN = sample.nbr_fea.shape[-1]
            cfg.CGCNN.POS_FEA_LEN = sample.positions.shape[-1]
            cfg.freeze()

        model_ref, cfg = build_trainer(cfg_path, checkpoints[0], device, cfg_override=cfg)

        accum = {}
        for fold_idx, ckpt in enumerate(checkpoints):
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            print(f"[{model_name}] Fold {fold_idx + 1}/{len(checkpoints)}: {ckpt.name}")
            model, _ = build_trainer(cfg_path, ckpt, device, cfg_override=cfg)
            loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_pool_leftnet)

            for batch in loader:
                batch_id = batch.cif_ids[0]
                target = float(batch.target.view(-1)[0].detach().cpu().item()) if hasattr(batch, "target") else None
                if use_input_feature_saliency:
                    pred, node_imp, edge_imp, edge_idx = input_feature_grad_x_input(model, batch, device)
                else:
                    model.zero_grad(set_to_none=True)
                    pred, feats = model(batch, return_features=True)
                    if isinstance(pred, (tuple, list)):
                        pred = pred[0]
                    pred_sum = pred.sum()
                    node_imp = edge_imp = None
                    if feats.get("node_repr") is not None:
                        feats["node_repr"].retain_grad()
                    if feats.get("edge_repr") is not None:
                        feats["edge_repr"].retain_grad()
                    pred_sum.backward()
                    if feats.get("node_repr") is not None and feats["node_repr"].grad is not None:
                        node_imp = (feats["node_repr"].grad * feats["node_repr"]).abs().sum(dim=-1).detach().cpu()
                    if feats.get("edge_repr") is not None and feats["edge_repr"].grad is not None:
                        edge_imp = (feats["edge_repr"].grad * feats["edge_repr"]).abs().sum(dim=-1).detach().cpu()
                    edge_idx = feats.get("edge_index")
                    if edge_idx is not None and torch.is_tensor(edge_idx):
                        edge_idx = edge_idx.detach().cpu()
                    pred = pred.detach().cpu()

                if batch_id not in accum:
                    # Batch size is 1 in this script, so edge index is local to this structure.
                    struct0 = batch.structures[0] if hasattr(batch, "structures") and batch.structures else None
                    edge_cross = edge_jimage = None
                    if struct0 is not None and edge_idx is not None and not use_input_feature_saliency:
                        edge_cross, edge_jimage = _edge_periodic_info_from_structure(struct0, edge_idx)

                    accum[batch_id] = {
                        "edge_sum": None,
                        "node_sum": None,
                        "preds": [],
                        "target": target,
                        "n": 0,
                        "edge_index": edge_idx if not use_input_feature_saliency else None,
                        "nbr_fea_idx": edge_idx if use_input_feature_saliency else None,
                        "atom_num": batch.atom_num.detach().cpu() if hasattr(batch, "atom_num") else None,
                        "pos": batch.positions.detach().cpu() if hasattr(batch, "positions") else None,
                        "edge_is_cross_lattice": edge_cross,
                        "edge_jimage": edge_jimage,
                    }
                else:
                    if use_input_feature_saliency:
                        if not torch.equal(accum[batch_id]["nbr_fea_idx"], edge_idx):
                            raise RuntimeError(
                                f"[{model_name}][{batch_id}] nbr_fea_idx mismatch across folds; "
                                "ensure graph construction is fixed."
                            )
                    elif edge_idx is not None and accum[batch_id]["edge_index"] is not None:
                        if not torch.equal(accum[batch_id]["edge_index"], edge_idx):
                            raise RuntimeError(
                                f"[{model_name}][{batch_id}] edge_index mismatch across folds; "
                                "ensure graph construction is fixed."
                            )

                entry = accum[batch_id]
                if edge_imp is not None:
                    entry["edge_sum"] = edge_imp if entry["edge_sum"] is None else (entry["edge_sum"] + edge_imp)
                if node_imp is not None:
                    entry["node_sum"] = node_imp if entry["node_sum"] is None else (entry["node_sum"] + node_imp)
                entry["preds"].append(float(pred.view(-1)[0].item()))
                entry["n"] += 1

                if args.save_per_fold:
                    payload_fold = {
                        "mpid": batch_id,
                        "model": model_name,
                        "fold_idx": fold_idx,
                        "checkpoint": ckpt.name,
                        "variant": "prop" if "prop" in model_name else "z",
                        "edge_index": edge_idx if not use_input_feature_saliency else None,
                        "nbr_fea_idx": edge_idx if use_input_feature_saliency else None,
                        "atom_num": entry["atom_num"],
                        "pos": entry["pos"],
                        "edge_mask": edge_imp,
                        "node_mask": node_imp,
                        "edge_is_cross_lattice": accum[batch_id]["edge_is_cross_lattice"],
                        "edge_jimage": accum[batch_id]["edge_jimage"],
                        "prediction": float(pred.view(-1)[0].item()),
                        "target": target,
                    }
                    out_path_fold = output_dir / f"{model_name}_{batch_id}_fold{fold_idx:02d}_saliency.pt"
                    torch.save(payload_fold, out_path_fold)

        for batch_id, entry in accum.items():
            n = entry["n"]
            payload_avg = {
                "mpid": batch_id,
                "model": model_name,
                "variant": "prop" if "prop" in model_name else "z",
                "num_folds": n,
                "edge_index": entry["edge_index"],
                "nbr_fea_idx": entry["nbr_fea_idx"],
                "atom_num": entry["atom_num"],
                "pos": entry["pos"],
                "edge_mask_mean": (entry["edge_sum"] / n) if entry["edge_sum"] is not None else None,
                "node_mask_mean": (entry["node_sum"] / n) if entry["node_sum"] is not None else None,
                "edge_is_cross_lattice": entry["edge_is_cross_lattice"],
                "edge_jimage": entry["edge_jimage"],
                "prediction_mean": float(sum(entry["preds"]) / len(entry["preds"])) if entry["preds"] else None,
                "prediction_std": float(torch.tensor(entry["preds"]).std(unbiased=False).item()) if len(entry["preds"]) > 1 else 0.0,
                "target": entry["target"],
            }
            out_path = output_dir / f"{model_name}_{batch_id}_saliency_ens{n}.pt"
            torch.save(payload_avg, out_path)
            print(f"[{model_name}] Saved averaged saliency output -> {out_path}")


if __name__ == "__main__":
    main()
