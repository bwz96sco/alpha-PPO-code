from __future__ import annotations

from dataclasses import dataclass

import torch as th
import torch.nn as nn

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym  # type: ignore[no-redef]

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def _conv1x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)


class SimConvExtractor(BaseFeaturesExtractor):
    """Legacy-like CNN extractor used in the PPO baseline (paper section 5).

    Architecture:
      BN(C) -> Conv(32,k=1x3) -> BN -> ReLU -> Conv(64,k=1x3) -> BN -> ReLU
      -> Conv(4,k=1x3) -> Flatten -> FC(128) -> ReLU
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        c, h, w = observation_space.shape
        self.pre_bn = nn.BatchNorm2d(c)
        self.conv1 = _conv1x3(c, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = _conv1x3(32, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = _conv1x3(64, 4)

        self.relu = nn.ReLU()
        n_flatten = 4 * h * w
        self.fc = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.pre_bn(observations)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = th.flatten(x, start_dim=1)
        return self.fc(x)


def _conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = _conv3x3(channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = _conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + x)
        return out


class ResNetExtractor(BaseFeaturesExtractor):
    """Legacy-like residual tower extractor (paper GPSearch network backbone)."""

    def __init__(self, observation_space: gym.spaces.Box, *, filter_num: int = 128, resblock_num: int = 9):
        c, h, w = observation_space.shape
        input_size = h * w
        features_dim = 2 * input_size
        super().__init__(observation_space, features_dim)

        self.pre_bn = nn.BatchNorm2d(c)
        self.conv = nn.Conv2d(c, filter_num, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(filter_num)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList([_ResidualBlock(filter_num) for _ in range(int(resblock_num))])

        # policy-head-style projection to a compact (2 * H * W) representation
        self.act_conv = nn.Conv2d(filter_num, 2, kernel_size=1)
        self.act_bn = nn.BatchNorm2d(2)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.pre_bn(observations)
        x = self.relu(self.bn(self.conv(x)))
        for blk in self.blocks:
            x = blk(x)
        x = self.relu(self.act_bn(self.act_conv(x)))
        return th.flatten(x, start_dim=1)

