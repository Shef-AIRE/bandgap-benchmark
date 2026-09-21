"""Factory functions for constructing CartNet benchmark models."""

from realmat_bag.pipeline.models.cartnet.CartNet import CartNet
from realmat_bag.pipeline.trainer import MaterialsTrainer


def get_cartnet_model(cfg):
    """Builds and returns a CartNet model according to the config object passed."""
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
    }
    model_params = {
        "dim_in": cfg.CARTNET.DIM_IN,
        "dim_rbf": cfg.CARTNET.DIM_RBF,
        "num_layers": cfg.CARTNET.NUM_LAYERS,
        "invariant": cfg.CARTNET.INVARIANT,
        "temperature": cfg.CARTNET.TEMPERATURE,
        "use_envelope": cfg.CARTNET.USE_ENVELOPE,
        "atom_types": cfg.CARTNET.ATOM_TYPES,
        "radius": cfg.MODEL.RADIUS,
        "cholesky": False,
        "encoding": cfg.CARTNET.ENCODING,
    }
    model = CartNet(**model_params)

    return MaterialsTrainer(model=model, **train_params)
