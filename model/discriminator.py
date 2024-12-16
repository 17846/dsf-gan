import torch
from torch import nn
from torchvision import models
from model.block import DownsampleBlock,GaborDownsampleBlock,GaborConv2d
from torchvision.models import resnet18


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layer(x)

class EnhancedDiscriminator(nn.Module):
    def __init__(self, writer_count: int, character_count: int, image_size: int):
        super(EnhancedDiscriminator, self).__init__()
        self.feature_extractor = nn.Sequential(
            GaborDownsampleBlock(1, 64,2,1),
            DownsampleBlock(64, 128),
            DownsampleBlock(128, 256),
            DownsampleBlock(256, 512),
            DownsampleBlock(512, 512),
            nn.Flatten(),
        )
        
        feature_size = 512 * (image_size // 16) ** 2
        self.reality_classifier = MLP(feature_size,1,4096)
        self.writer_classifier = MLP(feature_size,writer_count,4096)

        self.character_classifier = MLP(feature_size,character_count,4096)

    def forward(self, input):
        feature = self.feature_extractor(input)
        reality = torch.sigmoid(self.reality_classifier(feature))
        # writer = self.writer_classifier(feature)
        # character = self.character_classifier(feature)

        return reality


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, writer_count: int, character_count: int, num_scales: int = 3, image_size: int = 64):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_scales = num_scales

        network_list = []
        for _ in range(num_scales):
            network_list.append(EnhancedDiscriminator(writer_count, character_count, image_size))
            image_size //= 2
        self.network = nn.ModuleList(network_list)

        self.downsample = nn.AvgPool2d(3, 2, 1, count_include_pad=False)

    def forward(self, input):
        output = []
        for i in range(self.num_scales):
            output.append(self.network[i](input))
            input = self.downsample(input)
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation (SE) block
class SELayer(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)

# 编码器块
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.se = SELayer(out_channels)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        return x

# 多尺度特征鉴别器网络
class MultiScaleDiscriminatorV2(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminatorV2, self).__init__()
        self.encoder1 = EncoderBlock(1, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc = nn.Linear(32768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x1 = self.encoder1(x)  # 64x64
        p1 = self.pool(x1)     # 32x32
        x2 = self.encoder2(p1) # 32x32
        p2 = self.pool(x2)     # 16x16
        x3 = self.encoder3(p2) # 16x16
        p3 = self.pool(x3)     # 8x8
        encoded = self.encoder4(p3) # 8x8
        
        encoded = encoded.view(encoded.size(0), -1) # Flatten the feature maps
        out = self.fc(encoded)
        out = self.sigmoid(out)
        return out

