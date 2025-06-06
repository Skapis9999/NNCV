import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models import efficientnet_b0

"""
https://arxiv.org/abs/2301.04648
"""
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels + skip_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


class AFFormerTiny(nn.Module):
    def __init__(self, in_channels=3, n_classes=19):
        super().__init__()
        self.n_classes = n_classes

        # Use a pretrained EfficientNet-B0 as encoder
        backbone = efficientnet_b0(weights="IMAGENET1K_V1")
        self.input_layer = nn.Sequential(
            backbone.features[0],  # Conv+BN+SiLU
            backbone.features[1],  # Depthwise conv block
        )
        self.encoder1 = backbone.features[2]  # ↓
        self.encoder2 = backbone.features[3]  # ↓
        self.encoder3 = backbone.features[4]  # ↓
        self.encoder4 = backbone.features[5:7]  # ↓

        # Freeze all encoder parts initially
        for param in self.parameters():
            param.requires_grad = False

        # Decoder
        self.decoder4 = DecoderBlock(192, 80, 128)  
        self.decoder3 = DecoderBlock(128, 40, 96)   
        self.decoder2 = DecoderBlock(96, 24, 64) 
        self.decoder1 = DecoderBlock(64, 16, 32)
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.input_layer(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        # Decoder
        d4 = self.decoder4(x4, x3)
        d3 = self.decoder3(d4, x2)
        d2 = self.decoder2(d3, x1)
        d1 = self.decoder1(d2, x0)

        # Final convolution
        out = self.final_conv(d1)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)  # Upsample to input size
        
        return out
