# =============================================================================
# Portions of this file are adapted from
#   NCMCounterfactuals by Kevin Xia & Yushu Pan (MIT License)
#   https://github.com/CausalAILab/NCMCounterfactuals
#
# Copyright (c) 2024 Kevin Xia and Yushu Pan
# Licensed under the MIT License.
# =============================================================================


import torch as T
import torch.nn as nn


class Distribution(nn.Module):
    def __init__(self, u):
        super().__init__()
        self.u = u
        self.device_param = nn.Parameter(T.empty(0))

    def __iter__(self):
        return iter(self.u)

    def sample(self, n=1, device='cpu'):
        raise NotImplementedError()

    def forward(self, n=1):
        return self.sample(n=n)
