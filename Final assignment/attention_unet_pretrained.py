import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AttentionBlock(nn.Module):
    """ Attention Gate 
    https://www.mdpi.com/1424-8220/23/20/8589"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Upsample g to match x’s spatial dimensions
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)
            print(g)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=19):
        super(AttentionUNet, self).__init__()

        # Load pretrained ResNet34
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        self.input_layer = nn.Sequential(
            resnet.conv1,  # 64 channels
            resnet.bn1,
            resnet.relu
        )
        self.pool1 = resnet.maxpool  # Downsampling

        # Encoder layers from ResNet
        self.encoder1 = resnet.layer1  # 64
        self.encoder2 = resnet.layer2  # 128
        self.encoder3 = resnet.layer3  # 256
        self.encoder4 = resnet.layer4  # 512

        # Freeze encoder
        for param in resnet.parameters():
            param.requires_grad = False
        # Bottom
        self.center = self._conv_block(512, 1024)

        # Decoder
        # self.up4 = UpConv(1024, 512)
        # self.att4 = AttentionBlock(512, 512, 256)
        # self.up_conv4 = self._conv_block(1024, 512)

        # self.up3 = UpConv(512, 256)
        # self.att3 = AttentionBlock(256, 256, 128)
        # self.up_conv3 = self._conv_block(512, 256)

        # self.up2 = UpConv(256, 128)
        # self.att2 = AttentionBlock(128, 128, 64)
        # self.up_conv2 = self._conv_block(256, 128)

        # self.up1 = UpConv(128, 64)
        # self.att1 = AttentionBlock(64, 64, 32)
        # self.up_conv1 = self._conv_block(128, 64)
        self.up4 = UpConv(512, 256)
        self.att4 = AttentionBlock(256, 256, 128)
        self.up_conv4 = self._conv_block(512, 256)

        self.up3 = UpConv(256, 128)
        self.att3 = AttentionBlock(128, 128, 64)
        self.up_conv3 = self._conv_block(256, 128)

        self.up2 = UpConv(128, 64)
        self.att2 = AttentionBlock(64, 64, 32)
        self.up_conv2 = self._conv_block(128, 64)

        self.up1 = UpConv(64, 64)
        self.att1 = AttentionBlock(64, 64, 32)
        self.up_conv1 = self._conv_block(128, 64)
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

        self.output_refine = nn.Conv2d(n_classes, n_classes, kernel_size=3, padding=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x0 = self.input_layer(x)        
        x1 = self.encoder1(self.pool1(x0))  # (64x64)
        x2 = self.encoder2(x1)  # (32x32)
        x3 = self.encoder3(x2)  # (16x16)
        x4 = self.encoder4(x3)  # (8x8)

        # center = self.center(x4)  # (8x8)
        
        # # Decoder with attention
        # d4 = self.up4(center)  # (16x16)
        # x4 = self.att4(d4, x4)
        
        # # Ensure the spatial dimensions match before concatenating
        # if d4.shape[2:] != x4.shape[2:]:
        #     d4 = F.interpolate(d4, size=x4.shape[2:], mode="bilinear", align_corners=True)
        
        # d4 = torch.cat((x4, d4), dim=1)  # Concatenate along channel dimension
        # d4 = self.up_conv4(d4)  # (16x16) --> (32x32)

        # d3 = self.up3(d4)  # (64x64)
        # x3 = self.att3(d3, x3)
        
        # # Ensure the spatial dimensions match before concatenating
        # if d3.shape[2:] != x3.shape[2:]:
        #     d3 = F.interpolate(d3, size=x3.shape[2:], mode="bilinear", align_corners=True)
        
        # d3 = torch.cat((x3, d3), dim=1)
        # d3 = self.up_conv3(d3)  # (64x64) --> (128x128)

        # d2 = self.up2(d3)  # (128x128) --> (256x256)
        # x2 = self.att2(d2, x2)
        
        # # Ensure the spatial dimensions match before concatenating
        # if d2.shape[2:] != x2.shape[2:]:
        #     print(x2.shape[2:])
        #     d2 = F.interpolate(d2, size=x2.shape[2:], mode="bilinear", align_corners=True)
        
        # d2 = torch.cat((x2, d2), dim=1)
        # d2 = self.up_conv2(d2)

        # d1 = self.up1(d2)
        # x1 = self.att1(d1, x1)
        
        # # Ensure the spatial dimensions match before concatenating
        # if d1.shape[2:] != x1.shape[2:]:
        #     print(x1.shape[2:])
        #     d1 = F.interpolate(d1, size=x1.shape[2:], mode="bilinear", align_corners=True)
        
        # d1 = torch.cat((x1, d1), dim=1)
        # d1 = self.up_conv1(d1)

        d4 = self.up4(x4)
        x3 = self.att4(d4, x3)
        d4 = self.up_conv4(torch.cat([x3, d4], dim=1))

        d3 = self.up3(d4)
        x2 = self.att3(d3, x2)
        d3 = self.up_conv3(torch.cat([x2, d3], dim=1))

        d2 = self.up2(d3)
        x1 = self.att2(d2, x1)
        d2 = self.up_conv2(torch.cat([x1, d2], dim=1))

        d1 = self.up1(d2)
        x0 = self.att1(d1, x0)
        d1 = self.up_conv1(torch.cat([x0, d1], dim=1))

        out = self.final(d1)  # (256x256)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        # out = self.output_refine(out)
        return out