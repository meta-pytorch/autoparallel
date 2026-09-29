import torch
from torch import nn


class ContrastiveBatchShuffle(nn.Module):
    def forward(self, features):
        permutation = torch.randperm(features.shape[0], device=features.device)
        return features[permutation]
