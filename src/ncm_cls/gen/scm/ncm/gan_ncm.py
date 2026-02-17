# =============================================================================
# Portions of this file are adapted from
#   NCMCounterfactuals by Kevin Xia & Yushu Pan (MIT License)
#   https://github.com/CausalAILab/NCMCounterfactuals
#
# Copyright (c) 2024 Kevin Xia and Yushu Pan
# Licensed under the MIT License.
# =============================================================================


import numpy as np
import torch as T
import torch.nn as nn

from src.ncm_cls.gen.scm.distribution.continuous_distribution import UniformDistribution, NeuralDistribution
from src.ncm_cls.gen.scm.ncm.mlp import MLP
from src.ncm_cls.gen.scm.scm import SCM, expand_do


class GAN_NCM(SCM):
    def __init__(self, cg, v_size={}, default_v_size=1, u_size={},
                 default_u_size=1, f={}, hyperparams=None,
                 default_gen_module=MLP, disc_module=MLP, gen_use_sigmoid=True, disc_use_sigmoid=True):
        if hyperparams is None:
            hyperparams = dict()

        self.cg = cg
        self.u_size = {k: u_size.get(k, default_u_size) for k in self.cg.c2}
        self.v_size = {k: v_size.get(k, default_v_size) for k in self.cg}

        v_total_size = sum(self.v_size.values())

        self.gen_use_sigmoid = gen_use_sigmoid

        gens = nn.ModuleDict({
                v: f[v] if v in f else default_gen_module(
                    {k: self.v_size[k] for k in self.cg.pa[v]},
                    {k: self.u_size[k] for k in self.cg.v2c2[v]},
                    self.v_size[v],
                    h_layers=hyperparams.get('h-layers', 2),
                    h_size=hyperparams.get('h-size', 128),
                    use_layer_norm=hyperparams.get('layer-norm', True),
                    use_sigmoid=gen_use_sigmoid
                )
                for v in cg})

        neural_pu = hyperparams.get('neural-pu', False)

        if neural_pu:
            pu_dist = NeuralDistribution(self.cg.c2, self.u_size, hyperparams)
        else:
            pu_dist = UniformDistribution(self.cg.c2, self.u_size)


        super().__init__(
            v=list(cg),
            f=gens,
            pu=pu_dist
        )

        # self.do_set_count = 1 # Only obs data is available

        self.f_disc = disc_module(
                self.v_size,
                {},
                1,
                h_layers=hyperparams.get('h-layers', 2),
                h_size=len(self.v_size) * hyperparams.get('h-size', 128),
                use_sigmoid=disc_use_sigmoid,
                use_layer_norm=hyperparams.get('layer-norm', False)
            )


    def convert_evaluation(self, samples):
        return {k: T.gt(samples[k], 0.5).float() for k in samples}

    def query_loss(self, input, val):
        if self.gen_use_sigmoid:
            return super().query_loss(input, val)
        else:
            if T.is_tensor(val):
                raise NotImplementedError()
            else:
                return T.sum(T.square(input - val))

    def get_disc_outputs(self, samples, include_inp=False):
        return self.f_disc(samples, {}, include_inp=include_inp)
