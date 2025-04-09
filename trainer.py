import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
# from pytorch_lightning import Trainer
# from sklearn.metrics import accuracy_score

# from cgcnn_train_bg import train


class MaterialsTrainer(pl.LightningModule):
    def __init__(self, model, optimizer, max_epochs, layer_freeze='all', init_lr=0.001, adapt_lr=False, **kwargs):
        super(MaterialsTrainer, self).__init__()
        self.model = model
        self._optimizer_params = optimizer
        self._max_epochs = max_epochs
        self._init_lr = init_lr
        self._adapt_lr = adapt_lr

        self.layers_freeze(mode=layer_freeze)

    def forward(self, batch):

        return self.model(batch)

    def compute_loss(self, batch, split_name="val"):
        # input, target, batch_cif_ids = batch
        
        results = self.forward(batch)
    
        # If forward returns a single object (not tuple/list), wrap it so we can index it.
        if not isinstance(results, (tuple, list)):
            results = (results,)
        
        # Extract only the first item.
        output = results[0]

        target = batch.target.to(self.device)
        l1_loss = nn.L1Loss()
        loss = nn.MSELoss()(output, target)
        mae = l1_loss(output, target)
        mre = torch.mean(torch.abs(output - target) / (target + 1e-8))  # mean relative error
        # Calculate R² (Coefficient of Determination)
        ss_total = torch.sum((target - torch.mean(target)) ** 2)
        ss_residual = torch.sum((target - output) ** 2)
        r2 = 1 - ss_residual / (ss_total + 1e-8)

        log_metrics = {
            f"{split_name}_loss": loss,
            f"{split_name}_mae": mae,
            f"{split_name}_mre": mre,
            f"{split_name}_r2": r2,
        }
        return loss, log_metrics

    def configure_optimizers(self):
        if self._optimizer_params is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=self._init_lr)
            return [optimizer]
        if self._optimizer_params["type"] == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self._init_lr,
                **self._optimizer_params["optim_params"],
            )
            return [optimizer]
        if self._optimizer_params["type"] == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self._init_lr,
                **self._optimizer_params["optim_params"],
            )

            if self._adapt_lr:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self._max_epochs)
                return [optimizer], [scheduler]
            return [optimizer]
        raise NotImplementedError(f"Unknown optimizer type {self._optimizer_params['type']}.")

    def training_step(self, train_batch, batch_idx):
        loss, metrics = self.compute_loss(train_batch, split_name="train")
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        current_lr = optimizer.param_groups[0]['lr']
        metrics['learning_rate'] = current_lr
        self.log_dict(metrics, on_step=True, on_epoch=True, logger=True, batch_size=train_batch.batch_size)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        loss, metrics = self.compute_loss(valid_batch, split_name="val")
        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True, batch_size=valid_batch.batch_size)

    def test_step(self, test_batch, batch_idx):
        loss, metrics = self.compute_loss(test_batch, split_name="test")
        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True, batch_size=test_batch.batch_size)

    def layers_freeze(self, mode='all'):
        """
        Freezes layers of the model based on the provided mode.
        Handles LEFTNet, CrystalGraphConvNet, and skips freezing for CartNet.

        Args:
            mode (str): The freezing mode. Options are:
                - 'all': Freeze all layers.
                - 'embedding': Freeze only embedding-related layers.
                - 'none': Do not freeze any layers.
        """
        model_name = self.model.__class__.__name__.lower()

        # Handle LEFTNet-specific logic
        if "leftnet" in model_name:
            print(f"Model detected as LEFTNet variant: {self.model.__class__.__name__}")
            if mode == 'all':
                if hasattr(self.model, 'z_emb'):
                    self.model.z_emb.weight.requires_grad = False
                if hasattr(self.model, 'radial_emb'):
                    for param in self.model.radial_emb.parameters():
                        param.requires_grad = False
                if hasattr(self.model, 'radial_lin'):
                    for param in self.model.radial_lin.parameters():
                        param.requires_grad = False
                if hasattr(self.model, 'message_layers'):
                    for layer in self.model.message_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                if hasattr(self.model, 'FTEs'):
                    for fte in self.model.FTEs:
                        for param in fte.parameters():
                            param.requires_grad = False

            elif mode == 'embedding':
                if hasattr(self.model, 'z_emb'):
                    self.model.z_emb.weight.requires_grad = False

            elif mode == 'none':
                print("No layers are frozen.")

            else:
                raise ValueError("Invalid mode. Choose from 'all', 'embedding', or 'none'.")

            print(f'LAYERS FREEZED MODE: {mode.upper()} for {self.model.__class__.__name__}')
            return

        # Handle CrystalGraphConvNet-specific logic
        elif "crystalgraphconvnet" in model_name:
            print(f"Model detected as CrystalGraphConvNet: {self.model.__class__.__name__}")

            for param in self.model.parameters():
                param.requires_grad = True

            if mode == 'all' and hasattr(self.model, 'embedding'):
                self.model.embedding.weight.requires_grad = False
                if hasattr(self.model.embedding, 'bias'):
                    self.model.embedding.bias.requires_grad = False
                if hasattr(self.model, 'convs'):
                    for conv in self.model.convs:
                        if hasattr(conv, 'fc_full'):
                            conv.fc_full.weight.requires_grad = False
                            if hasattr(conv.fc_full, 'bias'):
                                conv.fc_full.bias.requires_grad = False
                        if hasattr(conv, 'bn1'):
                            conv.bn1.weight.requires_grad = False
                            conv.bn1.bias.requires_grad = False
                        if hasattr(conv, 'bn2'):
                            conv.bn2.weight.requires_grad = False
                            conv.bn2.bias.requires_grad = False

            elif mode == 'embedding' and hasattr(self.model, 'embedding'):
                self.model.embedding.weight.requires_grad = False
                if hasattr(self.model.embedding, 'bias'):
                    self.model.embedding.bias.requires_grad = False

            elif mode not in ['all', 'embedding', 'none']:
                raise ValueError("Invalid mode. Choose from 'all', 'embedding', or 'none'.")

            print(f'LAYERS FREEZED MODE: {mode.upper()} for {self.model.__class__.__name__}')
            return

        # Skip layer freezing for CartNet
        elif "cartnet" in model_name:
            print(f"Skipping layer freezing for CartNet: {self.model.__class__.__name__}")
            return

        # Raise error for invalid models
        else:
            raise ValueError(f"Invalid model detected: {self.model.__class__.__name__}. "
                            f"Expected models are LEFTNet variants, CrystalGraphConvNet, or CartNet.")


class MetricsCallback(pl.Callback):
    def __init__(self):
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch_metrics = {
            'epoch': trainer.current_epoch,
            'val_loss': float(metrics['val_loss']) if 'val_loss' in metrics else None,
            'val_mae': float(metrics['val_mae']) if 'val_mae' in metrics else None,
            'val_mre': float(metrics['val_mre']) if 'val_mre' in metrics else None,
            'val_r2': float(metrics['val_r2']) if 'val_r2' in metrics else None,
            'train_loss': float(metrics['train_loss']) if 'train_loss' in metrics else None,
            'train_mae': float(metrics['train_mae']) if 'train_mae' in metrics else None,
            'train_mre': float(metrics['train_mre']) if 'train_mre' in metrics else None,
            'train_r2': float(metrics['train_r2']) if 'train_r2' in metrics else None,
            'train_loss_epoch': float(metrics['train_loss_epoch']) if 'train_loss_epoch' in metrics else None,
            'train_mae_epoch': float(metrics['train_mae_epoch']) if 'train_mae_epoch' in metrics else None,
            'train_mre_epoch': float(metrics['train_mre_epoch']) if 'train_mre_epoch' in metrics else None,
            'train_r2_epoch': float(metrics['train_r2_epoch']) if 'train_r2_epoch' in metrics else None
        }
        self.metrics.append(epoch_metrics)

    def get_metrics_dataframe(self):
        return pd.DataFrame(self.metrics)
    


def mean_relative_error(y_true, y_pred):
    # Epsilon to avoid division by zero if y_true can be zero
    eps = 1e-9
    rel_errors = np.abs(y_true - y_pred) / (np.abs(y_true) + eps)
    return np.mean(rel_errors)