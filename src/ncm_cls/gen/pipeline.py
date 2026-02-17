import numpy as np
import os
import torch as torch
import torch.nn as nn
import pytorch_lightning as pl
import random
from torch.utils.data import DataLoader, Dataset
from src.ds.cmnistbar_data_loader import ColoredBarMNIST
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

from src.ncm_cls.gen.scm.ncm.gan_ncm import GAN_NCM

from torch.autograd import Variable, grad

import pandas as pd
from src.ds.obs_dat_eval import get_obs_data_table

def log(x):
    return torch.log(x + 1e-8)

class Pipeline(pl.LightningModule):
    def __init__(self, cg=None, hyperparams=None, **kwargs):
        super().__init__()
        if hyperparams is None:
            hyperparams = dict()

        self.lr = hyperparams.get("lr-gen", 1e-4)

        self.batch_size = hyperparams.get("batch-size", 64)
        self.grad_clamp = hyperparams.get('grad-clamp', 0.01)
        self.gp_weight = hyperparams.get('gp-weight', 10.0)
        self.mc_sample_size = hyperparams.get('mc-sample-size', 10000)

        self.cg = cg
        self.dat_path = hyperparams.get('dat-path', 'dat/img')
        self.gan_mode = hyperparams.get('gan-mode', 'wgan')
        self.graph = hyperparams.get('graph', 'full-ncm')

        self.v_size, self.v_list = self._get_v_size_and_list(self.graph, self.cg)

        self.gt_table = get_obs_data_table(self.graph, self.dat_path)
        print("Obs distribution trying to match:")
        print(self.gt_table)

        self.ncm = GAN_NCM(cg, v_size=self.v_size, default_u_size=hyperparams.get('u-size', 1), hyperparams=hyperparams,
                        gen_use_sigmoid=True,
                        disc_use_sigmoid=(hyperparams.get("gan-mode", "wgan") != "wgan"))

        self.automatic_optimization = False
        self.stored_kl = 1e10

    def training_step(self, batch, batch_idx):

        batch = self._data_to_dict(batch, graph=self.graph)

        # ncm_opt_list, ncm_scheduler_list = self.optimizers()
        G_opt, D_opt, PU_opt = self.optimizers()

        real_batch_v = {k: batch[k] for k in self.v_list}
        ncm_batch_v = self.ncm.sample(n=self.batch_size)
        ncm_disc_real_out = self.ncm.get_disc_outputs(real_batch_v)
        ncm_disc_fake_out = self.ncm.get_disc_outputs(ncm_batch_v)
        loss_v_disc = self._get_D_loss(ncm_disc_real_out, ncm_disc_fake_out)

        if self.gan_mode == "wgangp":
            grad_penalty = self._get_gradient_penalty(real_batch_v, ncm_batch_v)
            self.log('grad_penalty', grad_penalty, prog_bar=True)
            loss_v_disc += grad_penalty


        self.manual_backward(loss_v_disc)

        D_opt.step()

        if self.gan_mode == "wgan":
            for p in self.ncm.f_disc.parameters():
                p.data.clamp_(-self.grad_clamp, self.grad_clamp)

        self.ncm.f.zero_grad()
        self.ncm.f_disc.zero_grad()
        self.ncm.pu.zero_grad()


        ncm_batch_v = self.ncm.sample(n=self.batch_size)
        ncm_disc_fake_out = self.ncm.get_disc_outputs(ncm_batch_v)
        G_loss = self._get_G_loss(ncm_disc_fake_out)

        self.manual_backward(G_loss)
        G_opt.step()
        PU_opt.step()

        self.ncm.f.zero_grad()
        self.ncm.f_disc.zero_grad()
        self.ncm.pu.zero_grad()

        if (self.current_epoch + 1) % 5 == 0 and batch_idx == 0:
            print('G_loss')
            print(G_loss)
            print('D_loss')
            print(loss_v_disc)

            ncm_batch_v = self.ncm.sample(n=100000)
            metrics = self._probability_table(dat=ncm_batch_v)

            if self.graph == "full-ncm":
                ordered_cols = ['D0', 'C0', 'BC0', 'BW0', 'P(V)']
                metrics = metrics[ordered_cols]
                metrics = metrics.sort_values(['D0', 'C0', 'BC0', 'BW0']).reset_index(drop=True)
                print(metrics[metrics['D0'] == 0])

            elif self.graph == "cls-digit":
                ordered_cols = ['X0', 'B0', 'Z0', 'P(V)']
                metrics = metrics[ordered_cols]
                metrics = metrics.sort_values(['X0', 'B0', 'Z0']).reset_index(drop=True)
                print(metrics[metrics['X0'] == 0])

            elif self.graph == "cls-color":
                ordered_cols = ['X0', 'B0', 'P(V)']
                metrics = metrics[ordered_cols]
                metrics = metrics.sort_values(['X0', 'B0']).reset_index(drop=True)
                print(metrics[metrics['X0'] == 0])
            else:
                raise ValueError(f"Graph {self.graph} not supported")

            cols = list(self.gt_table.columns[:-1])
            joined_table = self.gt_table.merge(metrics, how='left', on=cols, suffixes=['_t', '_m']).fillna(0.0000001)
            p_t = joined_table['P(V)_t']
            p_m = joined_table['P(V)_m']
            kl_div = (p_t * (np.log(p_t) - np.log(p_m))).sum()
            print("KL divergence between observed and generated:")
            print(kl_div)

            self.stored_kl = kl_div

        # self.log('train_loss', G_loss + loss_v_disc, prog_bar=True)
        self.log('train_loss', self.stored_kl, prog_bar=True)
        self.log('G_loss', G_loss, prog_bar=True)
        self.log('D_loss', loss_v_disc, prog_bar=True)


    def configure_optimizers(self):
        if self.gan_mode == "wgan":
            opt_gen = torch.optim.RMSprop(self.ncm.f.parameters(), lr=self.lr)
            opt_disc = torch.optim.RMSprop(self.ncm.f_disc.parameters(), lr=self.lr)
            opt_pu = torch.optim.RMSprop(self.ncm.pu.parameters(), lr=self.lr)
        else:
            opt_gen = torch.optim.Adam(self.ncm.f.parameters(), lr=self.lr)
            opt_disc = torch.optim.Adam(self.ncm.f_disc.parameters(), lr=self.lr)
            opt_pu = torch.optim.Adam(self.ncm.pu.parameters(), lr=self.lr)
        return opt_gen, opt_disc, opt_pu


    def _get_D_loss(self, real_out, fake_out):
        if self.gan_mode == "wgan" or self.gan_mode == "wgangp":
            return -(torch.mean(real_out) - torch.mean(fake_out))
        else:
            return -torch.mean(log(real_out) + log(1 - fake_out))

    def _get_G_loss(self, fake_out):
        if self.gan_mode == "bgan":
            return 0.5 * torch.mean((log(fake_out) - log(1 - fake_out)) ** 2)
        elif self.gan_mode == "wgan" or self.gan_mode == "wgangp":
            return -torch.mean(fake_out)
        else:
            return -torch.mean(log(fake_out))

    def _get_gradient_penalty(self, real_data, fake_data, disc_index):
        interpolated_data = dict()
        alpha = torch.rand(self.ncm_batch_size, 1, device=self.device, requires_grad=True)
        for V in real_data:
            v_alpha = alpha.expand_as(real_data[V])
            interpolated_data[V] = v_alpha * real_data[V].detach() + (1 - v_alpha) * fake_data[V].detach()

        interpolated_out, inp = self.ncm.get_disc_outputs(interpolated_data, disc_index, include_inp=True)
        gradients = grad(outputs=interpolated_out, inputs=inp,
                         grad_outputs=torch.ones(interpolated_out.size(), device=self.device),
                         create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(self.ncm_batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return self.gp_weight * (torch.relu(gradients_norm - self.grad_clamp) ** 2).mean()


    def _probability_table(self, m=None, n=1000000, do={}, dat=None):
        assert m is not None or dat is not None

        if dat is None:
            dat = m(n, do=do, evaluating=True)
        
        dat = {k: torch.argmax(v.detach(), dim=1, keepdim=True) for k, v in dat.items()}

        cols = dict()
        for v in sorted(dat):
            result = dat[v].detach().cpu().numpy()
            for i in range(result.shape[1]):
                cols["{}{}".format(v, i)] = np.squeeze(result[:, i])

        df = pd.DataFrame(cols)
        grouped = (df.groupby(list(df.columns))
                    .apply(lambda x: len(x) / len(df))
                    .rename('P(V)').reset_index()
                    [[*df.columns, 'P(V)']])
        
        return grouped

    def train_dataloader(self):
        dat_set = ColoredBarMNIST(cg='full-ncm', root=self.dat_path, env='train', v_only=True)
        return DataLoader(dat_set, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=16)

    def _get_v_size_and_list(self, graph, cg):
        v_size = {k: 2 for k in cg}
        if "cls" not in graph:
            v_size['D'] = 10
        elif "digit" in graph:
            v_size['X'] = 10
        elif "color" in graph:
            v_size['B'] = 10
        else:
            raise ValueError(f"CLS type {graph} not supported")
        
        v_list = [k for k in v_size]
        return v_size, v_list

    def _data_to_dict(self, batch, graph="cls"):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            _, V = batch
        else:
            V = batch

        if "cls" not in graph:
            batch = dict()
            batch['D'] = V[:, :10].float().to(self.device)
            batch['C'] = V[:, 10:12].float().to(self.device)
            batch['BC'] = V[:, 12:14].float().to(self.device)
            batch['BW'] = V[:, 14:16].float().to(self.device)
            return batch
        elif "digit" in graph:
            batch = dict()
            batch['X'] = V[:, :10].float().to(self.device)
            batch['B'] = V[:, 10:12].float().to(self.device)
            batch['Z'] = V[:, 12:14].float().to(self.device)
            return batch
        elif "color" in graph:
            batch = dict()
            batch['X'] = V[:, 10:12].float().to(self.device)
            batch['B'] = V[:, :10].float().to(self.device)
            return batch
        else:
            raise ValueError(f"CLS type {graph} not supported")





