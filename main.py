import argparse
import glob
import os
from datetime import datetime
import json
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import KFold
import torch
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
import random
from pathlib import Path

from realmat_bag.loaddata.cifdata import CIFData
from realmat_bag.loaddata.collate import collate_pool_leftnet
from models.cgcnn.model_cgcnn import get_cgcnn_model
from models.leftnet.model_leftnet import get_leftnet_model
from models.cartnet.model_cartnet import get_cartnet_model
from models.alignn.model_alignn import get_alignn_model
from models.CHGnet.model_chgnet import get_chgnet_model
from realmat_bag.pipeline.trainer import MetricsCallback, mean_relative_error
from config import get_cfg_defaults
from traditional_ml import run_traditional_model


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="CGCNN")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument(
        "--devices",
        default=1,
        help="gpu id(s) to use. int(0) for cpu. list[x,y] for xth, yth GPU."
        "str(x) for the first x GPUs. str(-1)/int(-1) for all available GPUs",
    )
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--pretrain", action="store_true", help="Enable pretraining mode using the full dataset")
    args = parser.parse_args()
    return args


def load_json_as_dataframe(path):
    with open(path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'mpids'}, inplace=True)
    return df


def load_data(cfg):
    if not cfg.DATASET.TRAIN:
        raise ValueError("DATASET.TRAIN must be specified in the configuration unless PREDEFINED_SPLIT with SPLIT_GLOB is used.")

    train_data = load_json_as_dataframe(cfg.DATASET.TRAIN)

    val_data = None
    if cfg.DATASET.VAL:
        if not os.path.exists(cfg.DATASET.VAL):
            raise FileNotFoundError(f"Validation file not found: {cfg.DATASET.VAL}")
        val_data = load_json_as_dataframe(cfg.DATASET.VAL)
    return train_data, val_data


def load_predefined_fold_specs(split_glob):
    train_files = sorted(glob.glob(split_glob))
    if not train_files:
        raise ValueError(f"No training files matched DATASET.SPLIT_GLOB pattern: {split_glob}")

    fold_specs = []
    for train_path in train_files:
        candidate_val = train_path.replace(".train.", ".test.")
        if candidate_val == train_path or not os.path.exists(candidate_val):
            candidate_val = train_path.replace(".train.json", ".test.json")
        if not os.path.exists(candidate_val):
            raise ValueError(f"Matching validation file for {train_path} not found. "
                             "Expected the same path with '.test.json'.")

        fold_specs.append(
            {
                "label": Path(train_path).name.replace(".train.json", "").replace(".train.", "."),
                "train_path": train_path,
                "val_path": candidate_val,
                "train_df": load_json_as_dataframe(train_path),
                "val_df": load_json_as_dataframe(candidate_val),
            }
        )
    return fold_specs


def prepare_datasets(cfg, train_fold, val_fold):
    train_fold = shuffle(train_fold[['mpids', 'bg']], random_state=cfg.SOLVER.SEED)
    train_dataset = CIFData(train_fold, cfg.MODEL.CIF_FOLDER, cfg.MODEL.INIT_FILE,
                            cfg.MODEL.MAX_NBRS, cfg.MODEL.RADIUS, cfg.SOLVER.RANDOMIZE)

    val_dataset = CIFData(val_fold[['mpids', 'bg']], cfg.MODEL.CIF_FOLDER, cfg.MODEL.INIT_FILE,
                          cfg.MODEL.MAX_NBRS, cfg.MODEL.RADIUS, cfg.SOLVER.RANDOMIZE)

    collate_fn = collate_pool_leftnet

    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        num_workers=cfg.SOLVER.WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.SOLVER.WORKERS
    )
    return train_loader, val_loader, train_dataset, val_dataset


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model(cfg):
    if cfg.MODEL.NAME == "cgcnn":
        return get_cgcnn_model(cfg)
    elif cfg.MODEL.NAME == "chgnet":
        return get_chgnet_model(cfg)
    elif cfg.MODEL.NAME == "leftnet":
        return get_leftnet_model(cfg)
    elif cfg.MODEL.NAME == "cartnet":
        return get_cartnet_model(cfg)
    elif cfg.MODEL.NAME == "alignn":
        return get_alignn_model(cfg)
    else:
        raise ValueError(f"Unknown model name: {cfg.MODEL.NAME}")


def load_pretrained_model(model, pretrained_model_path):
    if os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from {pretrained_model_path}...")
        checkpoint = torch.load(pretrained_model_path)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("No pretrained model found. Training a new model...")
    return model


def setup_logger(cfg, fold):
    log_dir_name = cfg.LOGGING.LOG_DIR_NAME
    if log_dir_name is None:
        log_dir_name = f"experiment_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{fold}"
    log_dir = f"{cfg.LOGGING.LOG_DIR}/{log_dir_name}"

    wandb_logger = WandbLogger(
        project="bandgap-project",
        name="experiment_{}".format(datetime.now().strftime('%Y%m%d-%H%M%S')),
    )

    wandb_logger.experiment.config.update(cfg, allow_val_change=True)

    return wandb_logger, log_dir


def setup_trainer(cfg, args, wandb_logger, log_dir, fold_label="fold"):
    safe_label = (fold_label or "fold").replace(os.sep, "_")

    # Save best model based on val_mre
    best_mre_checkpoint = ModelCheckpoint(
        dirpath=log_dir,
        monitor="val_mre",
        mode="min",
        save_top_k=1,
        filename=f"{safe_label}-best-mre-{{epoch:02d}}-{{val_mre:.4f}}",
    )

    # Save best model based on val_mae
    best_mae_checkpoint = ModelCheckpoint(
        dirpath=log_dir,
        monitor="val_mae",
        mode="min",
        save_top_k=1,
        filename=f"{safe_label}-best-mae-{{epoch:02d}}-{{val_mae:.4f}}",
    )

    # Always save the last model
    last_checkpoint = ModelCheckpoint(
        dirpath=log_dir,
        save_last=True,
        filename=f"{safe_label}-last-{{epoch:02d}}",
    )

    metrics_callback = MetricsCallback()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    try:
        requested_devices = int(args.devices)
    except (TypeError, ValueError, AttributeError):
        requested_devices = 1

    use_gpu = requested_devices != 0 and torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    devices = requested_devices if use_gpu else 1

    trainer = pl.Trainer(
        max_epochs=cfg.SOLVER.EPOCHS,
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        callbacks=[
            best_mre_checkpoint,
            best_mae_checkpoint,
            last_checkpoint,
            lr_monitor,
            metrics_callback,
        ],
    )
    return trainer, best_mre_checkpoint


def evaluate_model(trainer, model, val_loader, fold):
    val_results = trainer.test(model, dataloaders=val_loader)
    print(f"Validation Results (Fold {str(fold)}): {val_results}")
    return val_results


def test_model(cfg, trainer, model, fold):
    if cfg.DATASET.VAL:
        with open(cfg.DATASET.VAL, 'r') as f:
            test_data = json.load(f)
            test_data = pd.DataFrame.from_dict(test_data, orient='index')
            test_data.reset_index(inplace=True)
            test_data.rename(columns={'index': 'mpids'}, inplace=True)
        test_dataset = CIFData(test_data[['mpids', 'bg']], cfg.MODEL.CIF_FOLDER, cfg.MODEL.INIT_FILE,
                               cfg.MODEL.MAX_NBRS, cfg.MODEL.RADIUS, cfg.SOLVER.RANDOMIZE)
        test_loader = DataLoader(test_dataset, collate_fn=collate_pool_leftnet,
                                 batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, num_workers=cfg.SOLVER.WORKERS)
        test_results = trainer.test(model, dataloaders=test_loader)
        print(f"Test Results (Fold {str(fold)}): {test_results}")
        return test_results


def main():
    """The main for this domain adaptation example, showing the workflow"""
    args = arg_parse()

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    # cfg.freeze()

    # Set random seed for reproducibility
    set_random_seed(cfg.SOLVER.SEED)

    use_predefined_split = cfg.DATASET.PREDEFINED_SPLIT and bool(cfg.DATASET.SPLIT_GLOB)

    fold_specs = None
    if use_predefined_split:
        fold_specs = load_predefined_fold_specs(cfg.DATASET.SPLIT_GLOB)
        print(f"Found {len(fold_specs)} predefined fold(s) from {cfg.DATASET.SPLIT_GLOB}")
    else:
        train_data, val_data = load_data(cfg)

    if args.pretrain:
        if use_predefined_split:
            print("Pretraining mode enabled. Using predefined folds from configuration.")
            for idx, spec in enumerate(fold_specs):
                fold_label = spec["label"]
                print(f"\nPretraining fold {idx + 1}/{len(fold_specs)}: {fold_label}")

                cfg.defrost()
                cfg.DATASET.VAL = spec["val_path"]
                cfg.freeze()

                train_loader, val_loader, train_dataset, val_dataset = prepare_datasets(cfg, spec["train_df"], spec["val_df"])

                structures = train_dataset[0]
                cfg.defrost()
                cfg.CGCNN.ORIG_ATOM_FEA_LEN = structures.atom_fea.shape[-1]
                cfg.CGCNN.NBR_FEA_LEN = structures.nbr_fea.shape[-1]
                cfg.CGCNN.POS_FEA_LEN = structures.positions.shape[-1]
                cfg.freeze()

                model = get_model(cfg)
                wandb_logger, log_dir = setup_logger(cfg, fold_label)
                trainer, checkpoint_callback = setup_trainer(cfg, args, wandb_logger, log_dir, fold_label)

                ckpt_path = cfg.MODEL.PRETRAINED_MODEL_PATH if os.path.exists(cfg.MODEL.PRETRAINED_MODEL_PATH) else None

                trainer.fit(
                    model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                    ckpt_path=ckpt_path
                )

                best_model_path = checkpoint_callback.best_model_path
                model.load_state_dict(torch.load(best_model_path)["state_dict"])
                evaluate_model(trainer, model, val_loader, fold_label)
                test_model(cfg, trainer, model, fold_label)

                if hasattr(wandb_logger, "experiment"):
                    try:
                        wandb_logger.experiment.finish()
                    except Exception:
                        pass
        else:
            print("Pretraining mode enabled. Splitting training data into k folds for validation.")
            
            # Split and prepare data
            kf = KFold(n_splits=cfg.SOLVER.NUM_FOLDS, shuffle=True, random_state=cfg.SOLVER.SEED)
            train_idx, val_idx = next(kf.split(train_data))
            train_fold = train_data.iloc[train_idx]
            val_fold = train_data.iloc[val_idx]
            train_loader, val_loader, train_dataset, val_dataset = prepare_datasets(cfg, train_fold, val_fold)

            # Extract feature dimensions
            structures = train_dataset[0]
            cfg.defrost()
            cfg.CGCNN.ORIG_ATOM_FEA_LEN = structures.atom_fea.shape[-1]
            cfg.CGCNN.NBR_FEA_LEN = structures.nbr_fea.shape[-1]
            cfg.CGCNN.POS_FEA_LEN = structures.positions.shape[-1]
            cfg.freeze()

            # Setup model and trainer
            model = get_model(cfg)
            wandb_logger, log_dir = setup_logger(cfg, "pretrain")
            trainer, checkpoint_callback = setup_trainer(cfg, args, wandb_logger, log_dir, "pretrain")

            # Resume from checkpoint if exists
            ckpt_path = cfg.MODEL.PRETRAINED_MODEL_PATH if os.path.exists(cfg.MODEL.PRETRAINED_MODEL_PATH) else None

            trainer.fit(
                model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=ckpt_path
            )

            # Evaluate best model
            best_model_path = checkpoint_callback.best_model_path
            model.load_state_dict(torch.load(best_model_path)["state_dict"])
            evaluate_model(trainer, model, val_loader, "pretrain")
            test_model(cfg, trainer, model, "pretrain")

            if hasattr(wandb_logger, "experiment"):
                try:
                    wandb_logger.experiment.finish()
                except Exception:
                    pass
    else:
        if use_predefined_split:
            fold_iterable = [
                (idx, spec["label"], spec["train_df"], spec["val_df"], spec["val_path"])
                for idx, spec in enumerate(fold_specs, start=1)
            ]
            total_folds = len(fold_iterable)
        else:
            num_folds = cfg.SOLVER.NUM_FOLDS  # Add this to your configuration (e.g., 5 or 10)
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=cfg.SOLVER.SEED)
            fold_iterable = []
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_data), start=1):
                fold_iterable.append(
                    (
                        fold_idx,
                        str(fold_idx),
                        train_data.iloc[train_idx],
                        train_data.iloc[val_idx],
                        cfg.DATASET.VAL if cfg.DATASET.VAL else None,
                    )
                )
            total_folds = len(fold_iterable)

        for fold_num, fold_label, train_fold, val_fold, fold_val_path in fold_iterable:
            print(f"\nFold {fold_num}/{total_folds}: {fold_label}")

            if use_predefined_split and fold_val_path:
                cfg.defrost()
                cfg.DATASET.VAL = fold_val_path
                cfg.freeze()

            # Prepare datasets and loaders
            train_loader, val_loader, train_dataset, val_dataset = prepare_datasets(cfg, train_fold, val_fold)

            structures = train_dataset[0]  # for just one of the item in cifs
            orig_atom_fea_len = structures.atom_fea.shape[-1]
            nbr_fea_len = structures.nbr_fea.shape[-1]
            pos_fea_len = structures.positions.shape[-1]
            max_neighbours = structures.nbr_fea_idx.shape[-1]
            cfg.defrost()  # Unfreeze the cfg to allow modification
            cfg.CGCNN.ORIG_ATOM_FEA_LEN = orig_atom_fea_len
            cfg.CGCNN.NBR_FEA_LEN = nbr_fea_len
            cfg.CGCNN.POS_FEA_LEN = pos_fea_len
            cfg.freeze()  # Refreeze the cfg to prevent further changes
            print(cfg)

            if cfg.MODEL.NAME in ["random_forest", "linear_regression", "svm"]:
                print(f"Training {cfg.MODEL.NAME} model...")
                test_dataset = None
                if cfg.DATASET.VAL and os.path.exists(cfg.DATASET.VAL):
                    test_df = load_json_as_dataframe(cfg.DATASET.VAL)
                    test_dataset = CIFData(
                        test_df[['mpids', 'bg']],
                        cfg.MODEL.CIF_FOLDER,
                        cfg.MODEL.INIT_FILE,
                        cfg.MODEL.MAX_NBRS,
                        cfg.MODEL.RADIUS,
                        cfg.SOLVER.RANDOMIZE,
                    )

                results = run_traditional_model(
                    cfg,
                    train_dataset,
                    val_dataset,
                    fold_label,
                    test_dataset=test_dataset,
                )

                best_params = results.get("best_params")
                if best_params:
                    print(f"{fold_label} best params: {best_params}")
                    best_params_path = results.get("best_params_path")
                    if best_params_path:
                        print(f"Saved {cfg.MODEL.NAME} best params to {best_params_path}")

                val_metrics = results["val_metrics"]
                print(
                    f"{fold_label} {cfg.MODEL.NAME} Validation Result- "
                    f"MAE: {val_metrics['mae']}, MSE: {val_metrics['mse']}, "
                    f"MRE: {val_metrics['mre']}, R²: {val_metrics['r2']}"
                )

                test_metrics = results.get("test_metrics")
                if test_metrics:
                    print(
                        f"{fold_label} {cfg.MODEL.NAME} Test Result- "
                        f"MAE: {test_metrics['mae']}, MSE: {test_metrics['mse']}, "
                        f"MRE: {test_metrics['mre']}, R²: {test_metrics['r2']}"
                    )
                continue

            model = get_model(cfg)
            model = load_pretrained_model(model, cfg.MODEL.PRETRAINED_MODEL_PATH)

            wandb_logger, log_dir = setup_logger(cfg, fold_label)
            trainer, checkpoint_callback = setup_trainer(cfg, args, wandb_logger, log_dir, fold_label)

            # Train the model
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            # Load the best model for evaluation
            best_model_path = checkpoint_callback.best_model_path
            model.load_state_dict(torch.load(best_model_path)["state_dict"])

            # Evaluate the model
            evaluate_model(trainer, model, val_loader, fold_label)

            # Test on evaluation dataset if applicable
            test_model(cfg, trainer, model, fold_label)

            if hasattr(wandb_logger, "experiment"):
                try:
                    wandb_logger.experiment.finish()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
