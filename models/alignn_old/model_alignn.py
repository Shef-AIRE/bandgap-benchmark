from __future__ import annotations

import os
from copy import deepcopy

import torch

from trainer import MaterialsTrainer
from .alignn import ALIGNNModel


def get_alignn_old_config(cfg):
    """
    Extract ALIGNN-specific configuration blocks from the global config.

    Mirrors the helper layout used by the other model wrappers so training
    entrypoints can stay uniform across architectures.
    """
    common_cfg = cfg.MODEL_COMMON
    alignn_cfg = cfg.ALIGNN
    num_rbf = alignn_cfg.NUM_RBF if alignn_cfg.NUM_RBF is not None else common_cfg.NUM_RADIAL
    cutoff = alignn_cfg["CUTOFF"] if "CUTOFF" in alignn_cfg else common_cfg.CUTOFF
    out_dim = alignn_cfg["OUTPUT_DIM"] if "OUTPUT_DIM" in alignn_cfg else common_cfg.OUTPUT_DIM

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
            "hidden_dim": alignn_cfg.HIDDEN_DIM,
            "num_layers": alignn_cfg.NUM_LAYERS,
            "num_rbf": num_rbf,
            "cutoff": cutoff,
            "dropout": alignn_cfg.DROPOUT,
            "readout": alignn_cfg.READOUT,
            "out_dim": out_dim,
            "activation": alignn_cfg.ACTIVATION,
            "rbf_trainable": alignn_cfg.RBF_TRAINABLE,
            "max_neighbors": alignn_cfg.MAX_NEIGHBORS,
            "encoding": alignn_cfg.ENCODING,
            "max_num_elements": alignn_cfg.MAX_NUM_ELEMENTS,
        },
    }


def get_alignn_old_model(cfg):
    """
    Build ALIGNN wrapped in MaterialsTrainer, matching the pattern of other models.
    """
    config_params = get_alignn_old_config(cfg)
    train_params = deepcopy(config_params["train_params"])
    model_params = deepcopy(config_params["model_params"])

    model = ALIGNNModel(**model_params)
    trainer = MaterialsTrainer(model=model, **train_params)

    pretrained_path = getattr(cfg.MODEL, "PRETRAINED_MODEL_PATH", "")
    if pretrained_path and os.path.isfile(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        if any(key.startswith("model.") for key in state_dict):
            state_dict = {key[len("model.") :]: value for key, value in state_dict.items()}

        missing, unexpected = trainer.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys when loading ALIGNN checkpoint: {missing}")
        if unexpected:
            print(f"Unexpected keys when loading ALIGNN checkpoint: {unexpected}")
        print(f"Loaded ALIGNN checkpoint from {pretrained_path} (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        print(f"=> no checkpoint found at '{cfg.MODEL.PRETRAINED_MODEL_PATH}'")

    return trainer


__all__ = ["get_alignn_old_config", "get_alignn_old_model"]
