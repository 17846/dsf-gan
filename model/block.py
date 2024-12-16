import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.residual(x)
class AdaptiveAttention(nn.Module):
    def __init__(self, feature_dim=512):
        super(AdaptiveAttention, self).__init__()
        self.feature_dim = feature_dim
        # 注意力权重生成器，根据两种风格的特征动态生成注意力权重
        self.attention_weight_generator = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),  # 输入维度是两种风格特征的拼接
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()  # 使用Sigmoid确保输出权重在0到1之间
        )
        
    def forward(self, writer_style, glyph_style):
        # 拼接两种风格的特征
        combined_features = torch.cat([writer_style, glyph_style], dim=-1)
        # 生成注意力权重
        attention_weights = self.attention_weight_generator(combined_features)
        # 根据权重融合两种风格的特征
        fused_style = writer_style * (1.5 - attention_weights) + glyph_style * attention_weights
        return fused_style

class CrossAttentionLayer(nn.Module):
    def __init__(self, input_channels, structure_channels, style_channels):
        super(CrossAttentionLayer, self).__init__()
        self.query_conv = nn.Conv2d(input_channels, input_channels // 8, 1)
        self.key_conv = nn.Conv2d(structure_channels, input_channels // 8, 1)
        self.value_conv = nn.Conv2d(style_channels, input_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, structure, style):
        batch_size, c, height, width = input.size()
        
        # Query is derived from the input feature map
        query = self.query_conv(input).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x (W*H) x C
        
        # Key is derived from the structure feature map
        key = self.key_conv(structure).view(batch_size, -1, width * height)  # B x C x (W*H)
        
        # Value is derived from the style feature map
        value = self.value_conv(style).view(batch_size, -1, width * height)  # B x C x (W*H)
        # Calculate the attention scores
        attention = self.softmax(torch.bmm(query, key) - torch.max(torch.bmm(query, key)))  # B x (W*H) x (W*H)
        
        # Apply the attention scores to the value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, c, height, width)
        
        return out



class GaborDownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        super(GaborDownsampleBlock, self).__init__()
        self.network = nn.Sequential(
            GaborConv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, input):
        output = self.network(input)
        return output

class GaborConv2d(_ConvNd):
 
    def __init__(self, in_channels, out_channels, kernel_size,  stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros',device="cuda"):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
    
        super(GaborConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                          _pair(0), groups, bias, padding_mode)
        self.freq = nn.Parameter(
            (3.14 / 2) * 1.41 ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor))
        self.theta = nn.Parameter((3.14 / 8) * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor))
        self.psi = nn.Parameter(3.14 * torch.rand(out_channels, in_channels))
        self.sigma = nn.Parameter(3.14 / self.freq)
        self.x0 = torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0]
        self.y0 = torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0]
        self.device = device
 
    def forward(self, input_image):
        y, x = torch.meshgrid([torch.linspace(-self.x0 + 1, self.x0, self.kernel_size[0]),
                               torch.linspace(-self.y0 + 1, self.y0, self.kernel_size[1])])
        x = x.to(self.device)
        y = y.to(self.device)
        weight = torch.empty(self.weight.shape, requires_grad=False).to(self.device)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                sigma = self.sigma[i, j].expand_as(y)
                freq = self.freq[i, j].expand_as(y)
                theta = self.theta[i, j].expand_as(y)
                psi = self.psi[i, j].expand_as(y)
 
                rotx = x * torch.cos(theta) + y * torch.sin(theta)
                roty = -x * torch.sin(theta) + y * torch.cos(theta)
 
                g = torch.zeros(y.shape)
 
                g = torch.exp(-0.5 * ((rotx ** 2 + roty ** 2) / (sigma + 1e-3) ** 2))
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * 3.14 * sigma ** 2)
                weight[i, j] = g
                self.weight.data[i, j] = g
        return F.conv2d(input_image, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class SelfAttention(nn.Module):

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, input):
        query = self.query_conv(input)
        key = self.key_conv(input)
        value = self.value_conv(input)
    
        query = query.view(query.shape[0], -1, query.shape[2]*query.shape[3])
        key = key.view(key.shape[0], -1, key.shape[2]*key.shape[3])
        value = value.view(value.shape[0], -1, value.shape[2]*value.shape[3])
        
        energy = torch.bmm(query.permute(0,2,1), key)
        attention = self.softmax(energy)
               
        out = torch.bmm(value, attention.permute(0,2,1))
        out = out.view(input.shape)
    
        return out

class AttenDownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        super(AttenDownsampleBlock, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            SelfAttention(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, input):
        output = self.network(input)
        return output


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        super(DownsampleBlock, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, input):
        output = self.network(input)
        return output
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        super(UpsampleBlock, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, input, injection=None):
        output = self.network(input)
        if injection is not None:
            output = torch.cat((output, injection), dim=1)
        return output


class MultilayerPerceptron(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 256, hidden_layers: int = 2):
        super(MultilayerPerceptron, self).__init__()
        self.network = [nn.Linear(in_channels, hidden_channels), nn.LeakyReLU(0.1, inplace=True)]
        for _ in range(hidden_layers - 1):
            self.network.append(nn.Linear(hidden_channels, hidden_channels))
            self.network.append(nn.LeakyReLU(0.1, inplace=True))
        self.network.append(nn.Linear(hidden_channels, out_channels))
        self.network = nn.Sequential(*self.network)

    def forward(self, input):
        output = self.network(input)
        return output
