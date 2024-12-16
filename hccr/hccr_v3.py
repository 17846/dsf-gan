import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.growth_rate = growth_rate

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels + growth_rate)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = torch.cat([x, out], 1)  # Concatenate the output of previous layers
        
        out = self.relu(self.bn2(self.conv2(out)))
        out = torch.cat([x, out], 1)  # Again, concatenate the output of all previous layers

        return out
class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient = False):
        super(DenseLayer,self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
 
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
 
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient
 
    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output
 
    def any_requires_grad(self, input):
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False
 
    # @torch.jit.unused
    # def call_checkpoint_bottleneck(self, input):
    #     def closure(*inputs):
    #         return self.bn_function(inputs)
 
    #     return cp.checkpoint(closure, *input)
 
    def forward(self, input):
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input
 
        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")
 
            # bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)
 
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features
class DenseBlock(nn.ModuleDict):
    def __init__(self,num_layers,num_input_features,bn_size,growth_rate,
                 drop_rate,memory_efficient = False,):
        super(DenseBlock,self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)
 
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)
class HandwritingRecognitionModel(nn.Module):
    def __init__(self, num_classes=3755,dropout_rate=0.1):
        super(HandwritingRecognitionModel, self).__init__()
        # self.gabor_block1 = GaborDownsampleBlock(in_channels=1, out_channels=16)
        self.gabor_block1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=4,stride=2,padding=1),
            nn.ReLU(inplace=True)
        )
        # self.gabor_block2 = GaborDownsampleBlock(in_channels=16, out_channels=32)
        self.gabor_block2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=4,stride=2,padding=1),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(dropout_rate)
        # Branch 1: 1 convolutional layer
        self.branch1_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 2 convolutional layers
        self.branch2_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch 3: 3 convolutional layers
        self.branch3_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.branch4_conv = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            nn.Conv2d(32, 64, 1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)  # 您可能还想在卷积层后加上ReLU激活函数
        )

        # Combine branches
        self.combine_conv = nn.Sequential(
            nn.Conv2d(64+128+256+64, 512, kernel_size=1),
            self.dropout  # Apply dropout here
        )

        # Residual blocks
        # self.residual_blocks = nn.Sequential(*[ResidualBlock(512) for _ in range(8)])
        self.residual_blocks = nn.Sequential(
            DenseBlock(12,512,4,32,0),
            DenseBlock(12,896,4,32,0),
            DenseBlock(12,1280,4,32,0),
            DenseBlock(12,1664,4,32,0),
        )

        # Adaptive Average Pooling
        self.adaptive_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # Final fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(2048, num_classes),
            self.dropout  # Apply dropout before the final classification layer
        )

    def forward(self, x):
        out = self.gabor_block1(x)
        out = self.gabor_block2(out)

        # Branches
        branch1_out = self.branch1_conv(out)
        branch2_out = self.branch2_conv(out)
        branch3_out = self.branch3_conv(out)
        branch4_out = self.branch4_conv(out)
        # Combine branches
        combined_out = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)
        combined_out = self.combine_conv(combined_out)

        # Residual blocks
        out = self.residual_blocks(combined_out)

        # Adaptive pool and fully connected layer
        out = self.adaptive_pooling(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1) # Flatten for the fully connected layer
        out = self.fc(out)

        return out