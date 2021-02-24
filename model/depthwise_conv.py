"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Depth Wise Separable Conv2D used in mini-Xception 
"""

import torch
import torch.nn as nn

class SeparableConv2D(nn.Module):
    """
        Modified Depth wise Separable Convolution Block for mini-Xception
    """
    def __init__(self, in_channels, out_channels, kernel=3):
        super(SeparableConv2D, self).__init__()
        """
            Notes:
                - groups parameter: perform a groups of conv 
                https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                - Separates channel expantion from depth wise conv 
        """
        self.depth_wise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel, stride=1, groups=in_channels,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.point_wise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # depth wise
        x = self.depth_wise_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        # point wise
        x = self.point_wise_conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

if __name__ == "__main__":
    depth_conv = SeparableConv2D(3,10)
    print(depth_conv)
    x = torch.randn((4, 3,112,112)) # batch, Channels, W, H
    print(x.shape)
    y = depth_conv(x)
    print(y.shape)
