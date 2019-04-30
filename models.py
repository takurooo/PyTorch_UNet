# -------------------------------------------
# import
# -------------------------------------------
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision import models
# -------------------------------------------
# defines
# -------------------------------------------
CUR_PATH = os.path.join(os.path.dirname(__file__))

# -------------------------------------------
# private functions
# -------------------------------------------

# -------------------------------------------
# public functions
# -------------------------------------------


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.down_conv(x)
        return x


class Downsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.down(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, in_ch, out_ch, batchnorm=False, bilinear=False):
        super().__init__()
        self.bilinear = bilinear
        self.batchnorm = batchnorm
        if not self.bilinear:
            self.up = nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=2, stride=2)
            if self.batchnorm:
                self.batchnorm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        if self.bilinear:
            x = F.interpolate(x, scale_factor=2,
                              mode='bilinear', align_corners=True)
        else:
            x = self.up(x)
            if self.batchnorm:
                x = self.batchnorm(x)

        return x


class UNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = Conv(3, 64)
        self.conv2 = Conv(64, 128)
        self.conv3 = Conv(128, 256)
        self.conv4 = Conv(256, 512)
        self.down = Downsampling()

        self.conv5 = Conv(512, 1024)  # bottom

        self.up1 = Upsampling(1024, 512)
        self.conv6 = Conv(1024, 512)
        self.up2 = Upsampling(512, 256)
        self.conv7 = Conv(512, 256)
        self.up3 = Upsampling(256, 128)
        self.conv8 = Conv(256, 128)
        self.up4 = Upsampling(128, 64)
        self.conv9 = Conv(128, 64)

        self.conv10 = nn.Conv2d(
            64, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(self.down(x1))
        x3 = self.conv3(self.down(x2))
        x4 = self.conv4(self.down(x3))

        x5 = self.conv5(self.down(x4))

        x6 = torch.cat([x4, self.up1(x5)], dim=1)
        x6 = self.conv6(x6)
        x7 = torch.cat([x3, self.up2(x6)], dim=1)
        x7 = self.conv7(x7)
        x8 = torch.cat([x2, self.up3(x7)], dim=1)
        x8 = self.conv8(x8)
        x9 = torch.cat([x1, self.up4(x8)], dim=1)
        x9 = self.conv9(x9)

        x10 = self.conv10(x9)

        return x10

# -------------------------------------------
# main
# -------------------------------------------


if __name__ == '__main__':
    pass
