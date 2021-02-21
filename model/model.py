"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Mini-Xception for a real time Emotion Recognition 
"""
import sys

from torch.nn.modules.activation import ReLU
sys.path.insert(1, '../')

import torch
import torch.nn as nn
from depthwise_conv import SeparableConv2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class ResidualXceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super(ResidualXceptionBlock, self).__init__()

        self.depthwise_conv1 = SeparableConv2D(in_channels, out_channels, kernel)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.depthwise_conv2 = SeparableConv2D(out_channels, out_channels, kernel)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # self.padd = nn.ZeroPad2d(31)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # residual branch
        # self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=30, bias=False)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False)

        self.residual_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # residual branch
        residual = self.residual_conv(x)
        residual = self.residual_bn(residual)
        
        print('input',x.shape)
        # feature extraction branch
        x = self.depthwise_conv1(x)
        print('conv1',x.shape)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.depthwise_conv2(x)
        print('conv2',x.shape)
        x = self.bn2(x)
        # x = self.padd(x)
        x = self.maxpool(x)
        print('max_pooling',x.shape)
        print('res',residual.shape)
        return x + residual

class Mini_Xception(nn.Module):
    def __init__(self):
        super(Mini_Xception, self).__init__()

        self.conv1 = conv_bn_relu(1, 8, kernel_size=3, stride=1, padding=0)
        self.conv2 = conv_bn_relu(8, 8, kernel_size=3, stride=1, padding=0)
        self.residual_blocks = nn.ModuleList([
            ResidualXceptionBlock(8 , 16),
            ResidualXceptionBlock(16, 32),
            ResidualXceptionBlock(32, 64),
            ResidualXceptionBlock(64, 128)            
        ])

        self.conv3 = nn.Conv2d(128, 7, kernel_size=3, stride=1, padding=1)
        # self.global_avg_pool = nn.AvgPool2d(kernel_size=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        for block in self.residual_blocks:
            x = block(x)

        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.global_avg_pool(x)
        # x = self.softmax(x)

        return x


if __name__ == '__main__':
    x = torch.randn((2, 1, 64,64))
    conv1 = conv_bn_relu(1, 8, kernel_size=3, stride=1, padding=0)
    conv2 = conv_bn_relu(8, 8, kernel_size=3, stride=1, padding=0)
    print(f'x.shape {x.shape}')
    x = conv1(x)
    print(f'x.shape {x.shape}')
    x = conv2(x)
    print(f'x.shape {x.shape}')
    xception_block = ResidualXceptionBlock(8, 16)
    x = xception_block(x)
    print(f'\nx.shape {x.shape}')
    xception_block = ResidualXceptionBlock(16, 32)
    x = xception_block(x)
    print(f'\nx.shape {x.shape}')
    xception_block = ResidualXceptionBlock(32, 64)
    x = xception_block(x)
    print(f'\nx.shape {x.shape}')
    xception_block = ResidualXceptionBlock(64, 128)
    x = xception_block(x)
    print(f'\nx.shape {x.shape}')

    print('*'*50)
    x = torch.randn((2, 1, 48,48))
    model = Mini_Xception()
    y = model(x)
    print(f'y.shape {y.squeeze().shape}')


