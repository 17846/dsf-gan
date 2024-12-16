from torch import nn
import torchvision.models as models
from model.block import DownsampleBlock, UpsampleBlock, AttenDownsampleBlock, SelfAttention
from model.transform import *
from einops import rearrange, repeat
import torch


class StructureEncoder(nn.Module):
    def __init__(self):
        super(StructureEncoder, self).__init__()

        self.downsample_1 = DownsampleBlock(1, 32, kernel_size=1, stride=1, padding=0)
        self.downsample_2 = AttenDownsampleBlock(32, 64)
        self.downsample_3 = DownsampleBlock(64, 128)
        self.downsample_4 = DownsampleBlock(128, 256)
        self.downsample_5 = DownsampleBlock(256, 512)
        self.downsample_6 = DownsampleBlock(512, 512)

        self.upsample_1 = UpsampleBlock(512, 512)
        self.upsample_2 = UpsampleBlock(1024, 256)
        self.upsample_3 = UpsampleBlock(512, 128)
        self.upsample_4 = UpsampleBlock(256, 64)
        self.upsample_5 = UpsampleBlock(128, 32)
        self.upsample_6 = UpsampleBlock(64, 64)

    def forward(self, input):  # (batch_size, 1, 64, 64)
        feature_1 = self.downsample_1(input)  # (batch_size, 32, 64, 64)
        feature_2 = self.downsample_2(feature_1)  # (batch_size, 64, 32, 32)
        feature_3 = self.downsample_3(feature_2)  # (batch_size, 128, 16, 16)
        feature_4 = self.downsample_4(feature_3)  # (batch_size, 256, 8, 8)
        feature_5 = self.downsample_5(feature_4)  # (batch_size, 512, 4, 4)
        attribute_1 = self.downsample_6(feature_5)  # (batch_size, 512, 2, 2)

        attribute_2 = self.upsample_1(attribute_1, feature_5)  # (batch_size, 1024, 4, 4)
        attribute_3 = self.upsample_2(attribute_2, feature_4)  # (batch_size, 512, 8, 8)
        attribute_4 = self.upsample_3(attribute_3, feature_3)  # (batch_size, 256, 16, 16)
        attribute_5 = self.upsample_4(attribute_4, feature_2)  # (batch_size, 128, 32, 32)
        attribute_6 = self.upsample_5(attribute_5, feature_1)  # (batch_size, 64, 64, 64)
        attribute_7 = self.upsample_6(attribute_6)  # (batch_size, 64, 128, 128)

        return attribute_1, attribute_2, attribute_3, attribute_4, attribute_5, attribute_6, attribute_7


# 注意力定位网络
class AttentionNet(nn.Module):
    def __init__(self):
        super(AttentionNet, self).__init__()
        self.linear = nn.Linear(in_features=1024, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_weights = self.sigmoid(self.linear(x))
        return attention_weights * x


# MLP投影头
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=512):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()
        self.base_model = models.densenet121(pretrained=True).features
        original_first_layer = self.base_model.conv0
        self.base_model.conv0 = nn.Conv2d(1,
                                          original_first_layer.out_channels,
                                          kernel_size=original_first_layer.kernel_size,
                                          stride=original_first_layer.stride,
                                          padding=original_first_layer.padding,
                                          bias=False)
        self.attention_net = AttentionNet()
        self.base_model.classifier = nn.Identity()  # 移除原始全连接层，仅使用特征
        self.projection_head = ProjectionHead()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size, num_imgs, h, w = x.shape
        x = x.view(batch_size * num_imgs, 1, h, w)
        features = self.base_model(x)  # 提取特征
        features = self.relu(features)
        features = self.attention_net(features)  # 注意力加权
        temp = torch.mean(features, dim=(2, 3))
        temp = temp.view(batch_size, num_imgs, -1)
        _, c, H, W = features.shape
        features = features.view(batch_size, num_imgs, c, H, W)
        features = torch.mean(features, dim=1)
        return features, temp


class StyleEncoderV8(nn.Module):
    def __init__(self):
        super(StyleEncoderV8, self).__init__()
        densenet121 = models.densenet121(pretrained=True).features
        densenet121.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.Feat_Encoder = densenet121
        self.add_position = PositionalEncoding(dropout=0.1, dim=512)
        encoder_layer = TransformerEncoderLayer(512, 8, 2048, 0.1, "relu", True)
        self.base_encoder = TransformerEncoder(encoder_layer, 2, None)
        writer_norm = nn.LayerNorm(512)
        self.atten_block = AttentionNet()
        self.fc = nn.Linear(1024, 512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.writer_head = TransformerEncoder(encoder_layer, 2, writer_norm)
        self.pro_mlp_writer = nn.Sequential(
            nn.Linear(512, 4096), nn.GELU(), nn.Linear(4096, 256))

    def forward(self, input):
        # [B,R,C,H,W]
        batch_size, num_imgs, h, w = input.shape
        style_images = input.view(-1, 1, h, w)
        # print(style_images.shape) [1920, 1, 64, 64]
        x = self.Feat_Encoder(style_images)  # [32 512 4 4]
        style_feat = self.style(x, input)
        # print(x.shape) torch.Size([1920, 512, 2, 2])
        anchor_num = num_imgs // 2
        style_embe = x.view(batch_size * num_imgs, 512, -1).permute(2, 0, 1)
        FEAT_ST_ENC = self.add_position(style_embe)

        memory = self.base_encoder(FEAT_ST_ENC)
        writer_memory = self.writer_head(memory)
        # print(writer_memory.shape) torch.Size([4, 1920, 512])
        writer_memory = rearrange(writer_memory, 't (b p n) c -> t (p b) n c',
                                  b=batch_size, p=2, n=anchor_num)  # [4, 2*B, N, C]
        memory_fea = rearrange(writer_memory, 't b n c ->(t n) b c')  # [4*N, 2*B, C]
        compact_fea = torch.mean(memory_fea, 0)  # [2*B, C]
        pro_emb = self.pro_mlp_writer(compact_fea)
        query_emb = pro_emb[:batch_size, :]
        pos_emb = pro_emb[batch_size:, :]
        nce_emb = torch.stack((query_emb, pos_emb), 1)  # [B, 2, C]
        nce_emb = nn.functional.normalize(nce_emb, p=2, dim=2)

        # input the writer-wise & character-wise styles into the decoder
        writer_style = memory_fea[:, :batch_size, :]  # [4*N, B, C]
        writer_style = torch.mean(writer_style, dim=0)
        return writer_style, style_feat, nce_emb

    def random_double_sampling(self, x, ratio=0.25):
        """
        Sample the positive pair (i.e., o and o^+) within a character by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [L, B, N, D], sequence
        return o [B, N, 1, D], o^+ [B, N, 1, D]
        """
        L, B, N, D = x.shape  # length, batch, group_number, dim
        x = rearrange(x, "L B N D -> B N L D")
        noise = torch.rand(B, N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=2)

        anchor_tokens, pos_tokens = int(L * ratio), int(L * 2 * ratio)
        ids_keep_anchor, ids_keep_pos = ids_shuffle[:, :, :anchor_tokens], ids_shuffle[:, :, anchor_tokens:pos_tokens]
        x_anchor = torch.gather(
            x, dim=2, index=ids_keep_anchor.unsqueeze(-1).repeat(1, 1, 1, D))
        x_pos = torch.gather(
            x, dim=2, index=ids_keep_pos.unsqueeze(-1).repeat(1, 1, 1, D))
        return x_anchor, x_pos

    def style(self, input, source):
        batch_size, num_imgs, h, w = source.shape
        x = self.avgpool(input)
        x = x.view(batch_size, num_imgs, 1024)
        x = torch.mean(x, dim=1)
        x = self.atten_block(x)
        x = self.fc(x)
        return x
