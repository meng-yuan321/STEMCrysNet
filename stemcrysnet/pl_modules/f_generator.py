import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
import hydra
import lightning as L
from torch_scatter import scatter
from tqdm import tqdm
from stemcrysnet.common.data_utils import lattice_params_to_matrix_torch
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment


class BaseModule(L.LightningModule):
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


### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings



class CSPFlow(BaseModule):
    def __init__(self, encoder=None, decoder=None, time_dim=256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.time_dim = time_dim
        # self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.time_dim * 2, _recursive_=False)
        self.timesteps = self.hparams.timesteps
        # self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.align = self.hparams.align
        self.keep_lattice = self.hparams.cost_lattice < 1e-5

        # self.xrd_encoder = hydra.utils.instantiate(self.hparams.encoder_xrd)
        # ckpt_fix = self.hparams.encoder_xrd_fix
        # if ckpt_fix != 'None':
        #     encoder_xrd_ckpt = self.hparams.ckpt_path + self.hparams.encoder_xrd_ckpt
        #     self.xrd_encoder.load_state_dict(torch.load(encoder_xrd_ckpt))
        #     print('Encoder_xrd loads ckpt: %s and fix %s' %(encoder_xrd_ckpt, ckpt_fix))
        #     if ckpt_fix:
        #         self.xrd_encoder.eval()
        #         for para in self.xrd_encoder.parameters():
        #             para.requires_grad = False
        # else:
        #     print('Encoder_xrd loads nothing')


    def clip_loss(self, loss):
        
        if torch.isinf(loss) or torch.isnan(loss):
            return torch.zeros_like(loss)
        return loss

    def get_decoder_state_dict(self, ori_state_dict):

        new_dict = OrderedDict()
        for k,v in ori_state_dict.items():
            if k.startswith('decoder'):
                new_dict[k[8:]] = v

        return new_dict

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps+1), batch_size)
        return torch.from_numpy(ts).to(device)

    def de_translation(self, coord_shift, batch_idx):

        graph_shift = scatter(coord_shift, batch_idx, reduce='mean', dim=0) # 计算每个图的均位移 graph_shift，形状为 (num_graphs, 3)
        return coord_shift - graph_shift[batch_idx]


    @torch.no_grad()
    def lap(self, f1, f2, types, num_atoms, batch):

        optimal_delta_f = self.find_opt_f1_minus_f2(f1, f2)
        optimal_delta_f = self.de_translation(optimal_delta_f, batch)
        optimal_f1 = (f2 + optimal_delta_f) % 1.
        if not self.align:
            return optimal_f1


        atoms_end = torch.cumsum(num_atoms, dim=0)
        atoms_begin = torch.zeros_like(num_atoms)
        atoms_begin[1:] = atoms_end[:-1]
        final_f1 = []

        for st, ed in zip(atoms_begin, atoms_end):
            types_crys = types[st:ed]
            f1_crys = optimal_f1[st:ed].unsqueeze(0)
            f2_crys = f2[st:ed].unsqueeze(1)

            mask = (types_crys.unsqueeze(1) != types_crys.unsqueeze(0)).float()
            dist = self.find_opt_f1_minus_f2(f1_crys, f2_crys).norm(dim=-1)
            dist = dist * (1 - mask) + 100. * mask
            dist = dist.detach().cpu().numpy()
            assignment = linear_sum_assignment(dist)[1].astype(np.int32)
            f1_crys_new = f1_crys[0][assignment]

            final_f1.append(f1_crys_new)
        return torch.cat(final_f1, dim=0)

    def find_opt_f1_minus_f2(self, f1, f2):
        '''
        这个函数计算两个分数坐标 f1 和 f2 之间的“周期最短差值”（考虑周期为 1 的环面），
        返回值在 (-0.5, 0.5] 区间内，适用于处理晶格的周期性坐标对齐问题。
        '''
        p = 2 * math.pi
        return torch.atan2(torch.sin((f1 - f2)*p), torch.cos((f1 - f2)*p)) / p


    def forward(self, batch):
        device = next(self.parameters()).device

        batch_size = batch.num_graphs
        times = self.uniform_sample_t(batch_size, device)
        time_emb = self.time_embedding(times)

        c1 = times / self.timesteps
        c0 = 1 - c1


        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords # x1

        rand_l, rand_x = torch.randn_like(lattices), torch.rand_like(frac_coords) # x0

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        if self.keep_lattice:
            input_lattice = lattices

        optimal_rand_x = self.lap(rand_x, frac_coords, batch.atom_types, batch.num_atoms, batch.batch)

        
        optimal_delta_f = self.find_opt_f1_minus_f2(optimal_rand_x, frac_coords)

        optimal_delta_f = self.de_translation(optimal_delta_f, batch.batch)

        c1_per_atom = c1.repeat_interleave(batch.num_atoms)[:, None]

        input_frac_coords = (frac_coords + c1_per_atom * optimal_delta_f) % 1.  # xt = exp_x1 (t log_x1 (x0))

        condition = self.encoder(batch)
        condition = condition/condition.norm(dim=-1, keepdim=True)
        pred_l, pred_x = self.decoder(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch, condition)

        pred_x = self.de_translation(pred_x, batch.batch)

        tar_x = -optimal_delta_f    # d_xt


        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord = F.mse_loss(pred_x, tar_x)

        loss_lattice = self.clip_loss(loss_lattice)
        loss_coord = self.clip_loss(loss_coord)


        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord)

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord
        }


    @torch.no_grad()
    def sample(self, batch, diff_ratio = 1.0, step_lr = 5, infer_timesteps=200):


        batch_size = batch.num_graphs
        condition = self.encoder(batch)
        condition = condition/condition.norm(dim=-1, keepdim=True)

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)
        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        assert self.timesteps % infer_timesteps == 0

        mult = self.timesteps // infer_timesteps

        time_start = self.timesteps - mult

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}


        for t in tqdm(range(time_start, 0, -mult)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']



            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch, condition)


            pred_x = self.de_translation(pred_x, batch.batch)

            step_size = 1. / infer_timesteps

            x_t_minus_1 = x_t + step_size * pred_x * (1 + step_lr * (1 - t / self.timesteps))

            l_t_minus_1 = l_t - step_size * (pred_l - l_t) / (1. - t / self.timesteps) if not self.keep_lattice else l_t




            traj[t - mult] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types, 
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        # traj_stack = {
        #     'num_atoms' : batch.num_atoms,
        #     'atom_types' : batch.atom_types,
        #     'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -mult)]),
        #     'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -mult)])
        # }

        tar = traj[0]

        return tar
        # return tar, traj_stack


    @torch.no_grad()
    def sample_given_inital_cell(self, batch, initial_cell, step_lr = 5, infer_timesteps=200):


        batch_size = batch.num_graphs
        condition = self.encoder(batch)
        condition = condition/condition.norm(dim=-1, keepdim=True)

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)
        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(initial_cell[:,:3], initial_cell[:,3:6])

        assert self.timesteps % infer_timesteps == 0

        mult = self.timesteps // infer_timesteps

        time_start = self.timesteps - mult

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}


        for t in tqdm(range(time_start, 0, -mult)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']



            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch, condition)


            pred_x = self.de_translation(pred_x, batch.batch)

            step_size = 1. / infer_timesteps

            x_t_minus_1 = x_t + step_size * pred_x * (1 + step_lr * (1 - t / self.timesteps))

            l_t_minus_1 = l_t - step_size * (pred_l - l_t) / (1. - t / self.timesteps) if not self.keep_lattice else l_t




            traj[t - mult] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        # traj_stack = {
        #     'num_atoms' : batch.num_atoms,
        #     'atom_types' : batch.atom_types,
        #     'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -mult)]),
        #     'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -mult)])
        # }

        tar = traj[0]

        return tar
        # return tar, traj_stack

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']


        self.log_dict(
            {'train_loss': loss,
            'lattice_loss': loss_lattice,
            'coord_loss': loss_coord},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
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
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord
        }

        return log_dict, loss

    @torch.no_grad()
    def sample_trajectory(self, batch, initial_cell, step_lr = 5, infer_timesteps=200):


        batch_size = batch.num_graphs
        condition = self.encoder(batch)
        condition = condition/condition.norm(dim=-1, keepdim=True)

        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)
        
        l_T = lattice_params_to_matrix_torch(initial_cell[:,:3], initial_cell[:,3:6])

        assert self.timesteps % infer_timesteps == 0

        mult = self.timesteps // infer_timesteps

        time_start = self.timesteps - mult

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}


        for t in tqdm(range(time_start, 0, -mult)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']



            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch, condition)


            pred_x = self.de_translation(pred_x, batch.batch)

            step_size = 1. / infer_timesteps

            x_t_minus_1 = x_t + step_size * pred_x * (1 + step_lr * (1 - t / self.timesteps))

            l_t_minus_1 = l_t




            traj[t - mult] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in [995, 700, 400, 0]]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in [995, 700, 400, 0]])
        }


        return  traj_stack