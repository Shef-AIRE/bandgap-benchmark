from __future__ import annotations

import os
from copy import deepcopy

import torch

from realmat_bag.pipeline.trainer import MaterialsTrainer
from .alignn import ALIGNN


def _get_or_default(cfg_section, key, default):
    if hasattr(cfg_section, key):
        value = getattr(cfg_section, key)
        if value is not None:
            return value
    return default


def get_config(cfg):
    """
    Build the configuration blocks needed to instantiate the DGL ALIGNN variant.
    """
    common_cfg = cfg.MODEL_COMMON
    alignn_cfg = cfg.ALIGNN

    num_gaussians = getattr(alignn_cfg, "NUM_GAUSSIANS", None)
    if num_gaussians is None:
        num_gaussians = (
            alignn_cfg.NUM_RBF if alignn_cfg.NUM_RBF is not None else 80
        )
    bond_feat_dim = _get_or_default(alignn_cfg, "BOND_FEAT_DIM", num_gaussians)
    triplet_features = _get_or_default(alignn_cfg, "TRIPLET_INPUT_FEATURES", 40)
    embedding_features = _get_or_default(alignn_cfg, "EMBEDDING_FEATURES", 64)
    atom_embedding_size = _get_or_default(
        alignn_cfg, "ATOM_EMBEDDING_SIZE", alignn_cfg.HIDDEN_DIM
    )
    alignn_layers = _get_or_default(alignn_cfg, "ALIGNN_LAYERS", alignn_cfg.NUM_LAYERS)
    gcn_layers = _get_or_default(alignn_cfg, "GCN_LAYERS", alignn_cfg.NUM_LAYERS)
    link = _get_or_default(alignn_cfg, "LINK", "identity")
    regress_forces = _get_or_default(alignn_cfg, "REGRESS_FORCES", False)

    return {
        "train_params": {
            "init_lr": cfg.SOLVER.LR,
            "lr_milestones": cfg.SOLVER.LR_MILESTONES,
            "max_epochs": cfg.SOLVER.EPOCHS,
            "optimizer": {
                "type": cfg.SOLVER.OPTIM,
                "optim_params": {
                    "momentum": cfg.SOLVER.MOMENTUM,
                    "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
                },
            },
            "layer_freeze": cfg.ALIGNN.LAYER_FREEZE,
        },
        "model_params": {
            "atom_input_dim": alignn_cfg.ATOM_FEA_LEN,
            "bond_feat_dim": bond_feat_dim,
            "num_targets": common_cfg.OUTPUT_DIM,
            "alignn_layers": alignn_layers,
            "gcn_layers": gcn_layers,
            "num_gaussians": num_gaussians,
            "triplet_input_features": triplet_features,
            "embedding_features": embedding_features,
            "atom_embedding_size": atom_embedding_size,
            "output_dim": common_cfg.OUTPUT_DIM,
            "link": link,
            "regress_forces": regress_forces,
            "cutoff": alignn_cfg.CUTOFF,
            "readout": alignn_cfg.READOUT,
            "max_neighbors": alignn_cfg.MAX_NEIGHBORS,
            "encoding": alignn_cfg.ENCODING,
            "max_num_elements": alignn_cfg.MAX_NUM_ELEMENTS,
        },
    }


def get_alignn_model(cfg):
    """
    Instantiate the DGL ALIGNN model wrapped in MaterialsTrainer.
    """
    config_params = get_config(cfg)
    train_params = deepcopy(config_params["train_params"])
    model_params = deepcopy(config_params["model_params"])

    model = ALIGNN(**model_params)
    trainer = MaterialsTrainer(model=model, **train_params)

    return trainer


__all__ = ["get_config", "get_alignn_model"]
