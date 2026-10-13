# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class ContrastiveBatchShuffle(nn.Module):
    def forward(self, features):
        permutation = torch.randperm(features.shape[0], device=features.device)
        return features[permutation]
