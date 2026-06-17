import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
import hydra
import pytorch_lightning as pl

from .AE_module import CSPDiffusion
from .stem_crys_aligner import STEMCrysAligner
from stemcrysnet.common.data_utils import lattice_params_to_matrix_torch

class UnifiedModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.crystal_encoder = hydra.utils.instantiate(self.hparams.crystal_encoder, _recursive_=False)
        self.stem_encoder = hydra.utils.instantiate(self.hparams.stem_encoder, two_stem=self.hparams.two_stem, _recursive_=False)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.decoder = hydra.utils.instantiate(self.hparams.crystal_decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim)
        
        # if getattr(self.hparams, 'freeze_crystal_encoder', False):
        #     for param in self.crystal_encoder.parameters():
        #         param.requires_grad = False
        #     print('crystal_encoder frozen')

        # Instantiate contrastive module
        self.contrastive_module = hydra.utils.instantiate(self.hparams.contrastive_module, stem_encoder=self.stem_encoder, crystal_encoder=self.crystal_encoder)

        # Instantiate Diffusion module
        self.self_module = hydra.utils.instantiate(self.hparams.self_module, encoder=self.crystal_encoder, decoder=self.decoder, beta_scheduler=self.beta_scheduler, sigma_scheduler=self.sigma_scheduler, time_dim = self.hparams.time_dim)
        self.cross_module = hydra.utils.instantiate(self.hparams.cross_module, encoder=self.stem_encoder, decoder=self.decoder, beta_scheduler=self.beta_scheduler, sigma_scheduler=self.sigma_scheduler, time_dim = self.hparams.time_dim)
        # Loss weights
        self.lambda_contrastive = getattr(self.hparams, 'lambda_contrastive', 1.0)
        self.lambda_self = getattr(self.hparams, 'lambda_self', 1.0)
        self.lambda_cross = getattr(self.hparams, 'lambda_cross', 1.0)



    def forward(self, batch: Any):
        self_loss_dict = self.self_module(batch)
        contrastive_loss_dict = self.contrastive_module(batch)
        cross_loss_dict = self.cross_module(batch)

        total_loss = (
            self.lambda_contrastive * contrastive_loss_dict['loss'] +
            self.lambda_self * self_loss_dict['loss'] +
            self.lambda_cross * cross_loss_dict['loss']
        )

        return {
            'loss': total_loss,
            **{f'contra_{k}': v for k, v in contrastive_loss_dict.items()},
            **{f'self_{k}': v for k, v in self_loss_dict.items()},
            **{f'cross_{k}': v for k, v in cross_loss_dict.items()},

        }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output = self(batch)
        loss = output['loss']

        log_dict = {
            'train_loss': loss,
            'train_loss_contra': output['contra_loss'],
            'train_loss_self': output['self_loss'],
            'train_loss_cross': output['cross_loss'],
        }
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output = self(batch)
        loss = output['loss']

        log_dict = {
            'val_loss': loss,
            'val_loss_contra': output['contra_loss'],
            'val_loss_self': output['self_loss'],
            'val_loss_cross': output['cross_loss'],
        }
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)

        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output = self(batch)
        loss = output['loss']

        log_dict = {
            'test_loss': loss,
            'test_loss_contra': output['contra_loss'],
            'test_loss_self': output['self_loss'],
            'test_loss_cross': output['cross_loss'],
        }
        self.log_dict(log_dict, batch_size=1, sync_dist=True)

        return loss

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}
