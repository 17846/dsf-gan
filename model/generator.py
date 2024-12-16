import torch
from torch import nn
from model.encoder import StructureEncoder, StyleEncoderV8
import torch.nn.functional as F


class DenormalizationLayer(nn.Module):
    def __init__(self, input_channels: int, structure_channels: int, style_channels: int):
        super(DenormalizationLayer, self).__init__()
        self.norm = nn.BatchNorm2d(input_channels)
        self.structure_mu = nn.Conv2d(structure_channels, input_channels, 1, 1, 0)
        self.structure_sigma = nn.Conv2d(structure_channels, input_channels, 1, 1, 0)

        self.style_mu = nn.Linear(style_channels, input_channels)
        self.style_sigma = nn.Linear(style_channels, input_channels)
        self.input_mask = nn.Conv2d(input_channels, 1, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, structure, style):
        input = self.norm(input)

        structure_mu = self.structure_mu(structure)
        structure_sigma = self.structure_sigma(structure)
        structure_activation = structure_mu + structure_sigma * input

        target_shape = (input.shape[0], input.shape[1], 1, 1)
        style_mu = self.style_mu(style).reshape(target_shape).expand_as(input)
        style_sigma = self.style_sigma(style).reshape(target_shape).expand_as(input)
        style_activation = style_mu + style_sigma * input

        input_mask = self.sigmoid(self.input_mask(input))
        output = (1 - input_mask) * structure_activation + input_mask * style_activation
        return output


class DenormalizationBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, structure_channels: int, style_channels: int,
                 dropout_prob=0.2):
        super(DenormalizationBlock, self).__init__()
        self.transform = bool(input_channels != output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.denorm_1 = DenormalizationLayer(input_channels, structure_channels, style_channels)
        self.conv_1 = nn.Conv2d(input_channels, input_channels, 3, 1, 1)

        self.denorm_2 = DenormalizationLayer(input_channels, structure_channels, style_channels)
        self.conv_2 = nn.Conv2d(input_channels, output_channels, 3, 1, 1)

        if self.transform:
            self.denorm_t = DenormalizationLayer(input_channels, structure_channels, style_channels)
            self.conv_3 = nn.Conv2d(input_channels, output_channels, 3, 1, 1)

    def forward(self, input, attribute, identity):
        output = self.dropout(self.denorm_1(input, attribute, identity))
        output = self.conv_1(self.relu(output))

        output = self.dropout(self.denorm_2(output, attribute, identity))
        output = self.conv_2(self.relu(output))

        if self.transform:
            input = self.dropout(self.denorm_t(input, attribute, identity))
            input = self.conv_3(self.relu(input))

        output = output + input
        return output


class DenormalizationGenerator(nn.Module):
    def __init__(self):
        super(DenormalizationGenerator, self).__init__()
        self.num_blocks = 7

        self.boot = nn.ConvTranspose2d(512, 512, 2, 1, 0)
        self.tanh = nn.Tanh()

        result_channels = [512, 512, 512, 512, 256, 128, 64, 1]
        structure_channels = [512, 1024, 512, 256, 128, 64, 64]
        style_channels = [512, 512, 512, 512, 512, 512, 512]
        # self.adaptAtten = AdaptiveAttention()
        denorm_list = []
        for i in range(self.num_blocks):
            denorm_list.append(DenormalizationBlock(result_channels[i], result_channels[i + 1], structure_channels[i],
                                                    style_channels[i]))
        self.denorm = nn.ModuleList(denorm_list)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, structure, writer_style, glyph_style):
        # style = self.adaptAtten(writer_style,glyph_style)
        target_shape = (writer_style.shape[0], writer_style.shape[1], 1, 1)
        result = self.boot(writer_style.reshape(target_shape))
        for i in range(self.num_blocks):
            if i < 5:
                result = self.denorm[i](result, structure[i], writer_style)
            else:
                result = self.denorm[i](result, structure[i], glyph_style)

            if i < self.num_blocks - 1:
                result = self.upsample(result)
        result = F.interpolate(result, size=(64, 64), mode='bilinear')
        result = self.tanh(result)
        return result


class SynthesisGenerator(nn.Module):
    def __init__(self):
        super(SynthesisGenerator, self).__init__()
        self.structure_encoder = StructureEncoder()
        self.style_encoder = StyleEncoderV8()
        self.result_decoder = DenormalizationGenerator()

    def forward(self, reference, template):
        structure = self.structure_encoder(template)
        writer_style, glyph_style, nce_emb = self.style_encoder(reference)
        result = self.result_decoder(structure, writer_style, glyph_style)
        return result, structure, glyph_style, nce_emb, writer_style

    def structure(self, input):
        output = self.structure_encoder(input)
        return output

    def style(self, input):
        output = self.style_encoder.Feat_Encoder(input)
        output = self.style_encoder.style(output, input)
        return output
