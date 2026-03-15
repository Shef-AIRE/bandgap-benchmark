import os
from copy import deepcopy
from typing import Any, Dict

import torch
from torch import nn

from realmat_bag.pipeline.models.CHGnet.model.CHGNet import CHGNet
from realmat_bag.pipeline.trainer import MaterialsTrainer


def _to_sequence(value):
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


class CHGNetLightningWrapper(nn.Module):
    """
    Thin wrapper that adapts CHGNet to the interface expected by MaterialsTrainer.
    """

    def __init__(self, chgnet: CHGNet):
        super().__init__()
        self.chgnet = chgnet

    def forward(self, batch, return_features: bool = False):
        prediction = self.chgnet(batch, return_features=return_features)
        if return_features:
            energy = prediction["e"]
        else:
            energy = prediction["e"]
        if energy.dim() == 1:
            energy = energy.unsqueeze(-1)
        if return_features:
            return energy, prediction.get("features", {})
        return energy


def get_config(cfg) -> Dict[str, Any]:
    train_params = {
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
        "layer_freeze": "none",
    }

    model_params = {
        "atom_fea_dim": cfg.CHGNET.ATOM_FEA_DIM,
        "bond_fea_dim": cfg.CHGNET.BOND_FEA_DIM,
        "angle_fea_dim": cfg.CHGNET.ANGLE_FEA_DIM,
        "num_radial": cfg.CHGNET.NUM_RADIAL,
        "num_angular": cfg.CHGNET.NUM_ANGULAR,
        "n_conv": cfg.CHGNET.N_CONV,
        "atom_conv_hidden_dim": _to_sequence(cfg.CHGNET.ATOM_CONV_HIDDEN_DIM),
        "bond_conv_hidden_dim": _to_sequence(cfg.CHGNET.BOND_CONV_HIDDEN_DIM),
        "angle_layer_hidden_dim": _to_sequence(cfg.CHGNET.ANGLE_LAYER_HIDDEN_DIM),
        "conv_dropout": cfg.CHGNET.CONV_DROPOUT,
        "read_out": cfg.CHGNET.READ_OUT,
        "mlp_hidden_dims": _to_sequence(cfg.CHGNET.MLP_HIDDEN_DIMS),
        "mlp_dropout": cfg.CHGNET.MLP_DROPOUT,
        "mlp_first": cfg.CHGNET.MLP_FIRST,
        "is_intensive": cfg.CHGNET.IS_INTENSIVE,
        "non_linearity": cfg.CHGNET.NON_LINEARITY,
        "atom_graph_cutoff": cfg.CHGNET.ATOM_GRAPH_CUTOFF,
        "bond_graph_cutoff": cfg.CHGNET.BOND_GRAPH_CUTOFF,
        "cutoff_coeff": cfg.CHGNET.CUTOFF_COEFF,
        "learnable_rbf": cfg.CHGNET.LEARNABLE_RBF,
        "gMLP_norm": cfg.CHGNET.GMLP_NORM,
        "readout_norm": cfg.CHGNET.READOUT_NORM,
        "encoding": cfg.CHGNET.ENCODING,
        "atom_input_dim": cfg.CHGNET.ATOM_INPUT_DIM,
        "max_num_elements": cfg.CHGNET.MAX_NUM_ELEMENTS,
    }

    return {
        "train_params": train_params,
        "model_params": model_params,
    }


def get_chgnet_model(cfg):
    config = get_config(cfg)
    train_params = deepcopy(config["train_params"])
    model_params = deepcopy(config["model_params"])

    chgnet = CHGNet(**model_params)
    wrapped_model = CHGNetLightningWrapper(chgnet=chgnet)
    trainer = MaterialsTrainer(model=wrapped_model, **train_params)

    return trainer
