import torch
from typing import Tuple, List

from torch import nn


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        self.feature_extractor = nn.ModuleList([
            # First layer
            nn.Sequential(
                nn.Conv2d(in_channels=image_channels,
                          out_channels=32,
                          stride=(1, 1),
                          kernel_size=(3, 3),
                          padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=2),
                nn.Conv2d(in_channels=32,
                          out_channels=64,
                          stride=(1, 1),
                          kernel_size=(3, 3),
                          padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=2),
                nn.Conv2d(in_channels=64,
                          out_channels=64,
                          stride=(1, 1),
                          kernel_size=(3, 3),
                          padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64,
                          out_channels=output_channels[0],
                          stride=(2, 2),
                          kernel_size=(3, 3),
                          padding=1),
                nn.BatchNorm2d(output_channels[0]),
                nn.ReLU(),
            ),
            # Second layer
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels=output_channels[0],
                          out_channels=128,
                          stride=(1, 1),
                          kernel_size=(3, 3),
                          padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(in_channels=128,
                          out_channels=output_channels[1],
                          stride=(2, 2),
                          kernel_size=(3, 3),
                          padding=1),
                nn.BatchNorm2d(output_channels[1]),
                nn.ReLU()
            ),
            # Third layer
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels=output_channels[1],
                          out_channels=256,
                          stride=(1, 1),
                          kernel_size=(3, 3),
                          padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                          out_channels=output_channels[2],
                          stride=(2, 2),
                          kernel_size=(3, 3),
                          padding=1),
                nn.BatchNorm2d(output_channels[2]),
                nn.ReLU()
            ),
            # Fourth layer
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels=output_channels[2],
                          out_channels=128,
                          stride=(1, 1),
                          kernel_size=(3, 3),
                          padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(in_channels=128,
                          out_channels=output_channels[3],
                          stride=(2, 2),
                          kernel_size=(3, 3),
                          padding=1),
                nn.ReLU()
            ),
            # Fifth layer
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels=output_channels[3],
                          out_channels=128,
                          stride=(1, 1),
                          kernel_size=(3, 3),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128,
                          out_channels=output_channels[4],
                          stride=(2, 2),
                          kernel_size=(3, 3),
                          padding=1),
                nn.ReLU()
            ),
            # Sixth layer
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels=output_channels[4],
                          out_channels=128,
                          stride=(1, 1),
                          kernel_size=(3, 3),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128,
                          out_channels=output_channels[5],
                          stride=(1, 1),
                          kernel_size=(3, 3),
                          padding=0),
                nn.ReLU()
            ),
        ])

    def forward(self, x):
        """
        The forward function should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = [self.feature_extractor[0](x)]
        for i in range(1, 6):
            out_features.append(self.feature_extractor[i](out_features[-1]))

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

