import math, copy
from argparse import Namespace
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Union, Iterable, List, Dict, Tuple, Optional, Any

import hydra
import omegaconf
import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support
from tqdm import tqdm
from lightning.pytorch.utilities import grad_norm

from stemcrysnet.common.utils import PROJECT_ROOT
from stemcrysnet.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc)

from .resnet import ModifiedResNet
from .utils import RegressionHead
from .cspnet import CSPLayer, SinusoidsEmbedding

MAX_ATOMIC_NUM=100

class CrysEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim = 128,
        num_layers = 4,
        max_atoms = 100,
        act_fn = 'silu',
        dis_emb = 'sin',
        num_freqs = 10,
        edge_style = 'fc',
        cutoff = 6.0,
        max_neighbors = 20,
        ln = False,
        ip = True,
        smooth = False,
        pred_type = False
    ):
        super(CrysEncoder, self).__init__()

        self.ip = ip
        self.node_embedding = nn.Embedding(max_atoms, hidden_dim) #原子类型嵌入层（将离散原子编号映射为连续向量）
        if act_fn == 'silu':
            self.act_fn = nn.SiLU()
        if dis_emb == 'sin':
            self.dis_emb = SinusoidsEmbedding(n_frequencies = num_freqs)
        elif dis_emb == 'none':
            self.dis_emb = None
        for i in range(0, num_layers):
            self.add_module(
                "csp_layer_%d" % i, CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, ip=ip)
            )            
        self.num_layers = num_layers
        self.out_node = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.pred_type = pred_type
        self.ln = ln
        self.edge_style = edge_style
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        if self.pred_type:
            self.type_out = nn.Linear(hidden_dim, max_atoms)

    def gen_edges(self, num_atoms, frac_coords):

        if self.edge_style == 'fc':
            lis = [torch.ones(n,n, device=num_atoms.device) for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]])

    def forward(self, batch):
        atom_types = batch.atom_types #[60] 这个batch的总原子数
        frac_coords = batch.frac_coords #[60, 3]
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles) #[btz, 3, 3]
        num_atoms = batch.num_atoms #[btz]
        node2graph = batch.batch #[60]
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords) #[2, 634] #[634, 3]
        edge2graph = node2graph[edges[0]] #[634]
        node_features = self.node_embedding(atom_types - 1) #[60, 512]

        for i in range(0, self.num_layers):
            node_features = self._modules["csp_layer_%d" % i](node_features, frac_coords, lattices, edges, edge2graph, frac_diff = frac_diff)

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        graph_features = scatter(node_features, node2graph, dim = 0, reduce = 'mean') # [btz, 512]
        final_feat = self.out_node(graph_features) #[btz, 512]
        return final_feat

class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

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
    
    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # norms = grad_norm(self, norm_type=2)
        parameters = self.parameters()
        grads = [p.grad for p in parameters if p.grad is not None]
        first_device = grads[0].device
        norms = []
        foreach = None
        grouped_grads: Dict[Tuple[torch.device, torch.dtype], List[List[torch.Tensor]]] \
        = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])  # type: ignore[assignment]
        # print(grouped_grads.items())
        # with open("/internfs/my/structure/stem2cif-3d-xtalnet/grouped_grads_debug.txt", "w") as f:
        #     f.write(f"grouped_grads: {grouped_grads}\n")
        # for ((device, _), [grads]) in grouped_grads.items():
        for (device_dtype, grads_list) in grouped_grads.items():
            device, dtype = device_dtype
            grads = grads_list[0]
            if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
                norms.extend(torch._foreach_norm(grads, 2.0))
            elif foreach:
                raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
            else:
                norms.extend([torch.norm(g, 2.0) for g in grads])
        total_norm = torch.norm(torch.stack([norm.to(first_device) for norm in norms]), 2.0)
        self.log_dict({'grad_norm_total': total_norm})

class STEMCrysAligner(BaseModule):
    def __init__(self, stem_encoder=None, crystal_encoder=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stem_encoder = stem_encoder
        self.crystal_encoder = crystal_encoder
        # self.stem_encoder = hydra.utils.instantiate(self.hparams.stem_encoder, _recursive_=False) #ModifiedResNet(Namespace())
        # self.crystal_encoder = crystal_encoder if crystal_encoder is not None else hydra.utils.instantiate(self.hparams.crystal_encoder, _recursive_=False)
        self.logit_scale = nn.Parameter(torch.ones([1]))
        # self.stack_view = self.hparams.stack_view
        # self.multi_view = self.hparams.multi_view
        # if self.multi_view:
        #     self.fusion_method = self.hparams.fusion_method
        #     if self.fusion_method == 'concat':
        #         self.mlp = nn.Sequential(
        #             nn.Linear(2*self.hparams.latent_dim, self.hparams.latent_dim),
        #             nn.ReLU(),
        #             nn.Linear(self.hparams.latent_dim, self.hparams.latent_dim)
        #         )

        # if 'crystal_encoder_pretrained' in self.hparams and self.hparams.crystal_encoder_pretrained:
        #     self.load_state_dict(torch.load(self.hparams.crystal_encoder_pretrained)['state_dict'], strict=False)
        #     print('succeffully load crystal encoder pretrained model')
        # if 'freeze_crystal_encoder' in self.hparams and self.hparams.freeze_crystal_encoder:
        #     for param in self.crystal_encoder.parameters():
        #         param.requires_grad = False
        #     print('freeze crystal encoder params')
        
        # if 'stem_pretrained' in self.hparams and self.hparams.stem_pretrained:
        #     self.load_state_dict(torch.load(self.hparams.stem_pretrained)['state_dict'], strict=False)
        #     print('succeffully load stem pretrained model')
        # if 'freeze_stem_encoder' in self.hparams and self.hparams.freeze_stem_encoder:
        #     for param in self.stem_encoder.parameters():
        #         param.requires_grad = False
        #     if self.multi_view and self.fusion_method == 'concat':
        #         for param in self.mlp.parameters():
        #             param.requires_grad = False
        #     print('freeze stem encoder params')

    def inference(self, batch):
        stem_feat = self.stem_encoder(batch)
        # elif self.multi_view:
        #     stem_feat1 = self.stem_encoder(batch.stem_img)
        #     stem_feat2 = self.stem_encoder(batch.stem_img_yz)
        #     if self.fusion_method == 'max':
        #         stem_feat = torch.max(stem_feat1, stem_feat2)
        #     elif self.fusion_method == 'concat':
        #         concat_feat = torch.cat([stem_feat1, stem_feat2], dim=-1)
        #         stem_feat = self.mlp(concat_feat)
        #         # stem_feat = concat_feat
        #     else:
        #         raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        # else:
        #     stem_feat = self.stem_encoder(batch.stem_img)

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        # atom_feat = self.crystal_encoder(batch.atom_types, batch.frac_coords, lattices, batch.num_atoms, batch.batch)
        atom_feat = self.crystal_encoder(batch)
        results = {
            'ids': batch.id,
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : batch.frac_coords,
            'lattices' : lattices
        }
        stem_feat = F.normalize(stem_feat, dim=-1).float()
        atom_feat = F.normalize(atom_feat, dim=-1).float()
        logit_scale = self.logit_scale.exp().float()
        
        results['stem_feat'] = stem_feat
        results['atom_feat'] = atom_feat
        results['logit_scale'] = logit_scale
        return results
        
    def forward(self, batch):
        stem_feat = self.stem_encoder(batch)
        # elif self.multi_view:
        #     stem_feat1 = self.stem_encoder(batch.stem_img, 1) # (bsz*256, 256)
        #     stem_feat2 = self.stem_encoder(batch.stem_img_yz, 1)
        #     # 使用特征融合层融合特征
        #     if self.fusion_method == 'max':
        #         stem_feat = torch.max(stem_feat1, stem_feat2)
        #     elif self.fusion_method == 'concat':
        #         concat_feat = torch.cat([stem_feat1, stem_feat2], dim=-1)
        #         stem_feat = self.mlp(concat_feat)
        #         # stem_feat = concat_feat
        #     else:
        #         raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        # else:
        #     stem_feat = self.stem_encoder(batch.stem_img, 1)
        
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        # atom_feat = self.crystal_encoder(batch.atom_types, batch.frac_coords, lattices, batch.num_atoms, batch.batch) # 使用 torch_geometric.loader.DataLoader 创建数据加载器时，它会自动将每个 batch 中的多个 Data 对象合并为一个 Batch 对象，并自动添加 .batch 属性，是一个 shape 为 [num_nodes_in_batch] 的向量，表示每个节点属于哪个图
        atom_feat = self.crystal_encoder(batch) # 使用 torch_geometric.loader.DataLoader 创建数据加载器时，它会自动将每个 batch 中的多个 Data 对象合并为一个 Batch 对象，并自动添加 .batch 属性，是一个 shape 为 [num_nodes_in_batch] 的向量，表示每个节点属于哪个图
        
        stem_feat = F.normalize(stem_feat, dim=-1).float() # (bsz, latent_dim)
        atom_feat = F.normalize(atom_feat, dim=-1).float() # (bsz, latent_dim)
        
        logit_scale = self.logit_scale.exp().float()
        logits_per_stem = logit_scale * stem_feat @ atom_feat.T # `@` 等价于 `torch.matmul` 函数
        logits_per_atom = logit_scale * atom_feat @ stem_feat.T # (bsz, bsz)
        labels = torch.arange(stem_feat.shape[0], device=stem_feat.device, dtype=torch.long)
        # print(f"***labels: {labels.shape}") # (bsz)
        stem_loss = F.cross_entropy(logits_per_stem, labels)
        atom_loss = F.cross_entropy(logits_per_atom, labels)
        total_loss = (stem_loss + atom_loss) / 2

        loss_dict = {
            'loss' : total_loss,
            'loss_CPCP_stem' : stem_loss,
            'loss_CPCP_atom' : atom_loss
        }
        return loss_dict
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss = output_dict['loss']
        loss_CPCP_stem = output_dict['loss_CPCP_stem']
        loss_CPCP_atom = output_dict['loss_CPCP_atom']

        self.log_dict(
            {'train_loss': loss,
            'train_loss_CPCP_stem': loss_CPCP_stem,
            'train_loss_CPCP_atom': loss_CPCP_atom},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
            sync_dist=True
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
            sync_dist=True
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
            batch_size=1,
            sync_dist=True
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_CPCP_atom = output_dict['loss_CPCP_atom']
        loss_CPCP_stem = output_dict['loss_CPCP_stem']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_loss_CPCP_stem': loss_CPCP_stem,
            f'{prefix}_loss_CPCP_atom': loss_CPCP_atom
        }

        return log_dict, loss