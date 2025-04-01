import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """ Attention Gate 
    https://www.mdpi.com/1424-8220/23/20/8589"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UpConv(nn.Module):
    """ Upsampling followed by a convolution """
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.up(x)

class AttentionUNet(nn.Module):
    """ U-Net with Attention Mechanisms """
    def __init__(self, in_channels=3, n_classes=19):
        super(AttentionUNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down1 = self._conv_block(64, 128)
        self.down2 = self._conv_block(128, 256)
        self.down3 = self._conv_block(256, 512)
        self.down4 = self._conv_block(512, 1024)

        self.up4 = UpConv(1024, 512)
        self.att4 = AttentionBlock(512, 512, 256)
        self.up_conv4 = self._conv_block(1024, 512)
        
        self.up3 = UpConv(512, 256)
        self.att3 = AttentionBlock(256, 256, 128)
        self.up_conv3 = self._conv_block(512, 256)
        
        self.up2 = UpConv(256, 128)
        self.att2 = AttentionBlock(128, 128, 64)
        self.up_conv2 = self._conv_block(256, 128)
        
        self.up1 = UpConv(128, 64)
        self.att1 = AttentionBlock(64, 64, 32)
        self.up_conv1 = self._conv_block(128, 64)
        
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.pool(x1)
        x2 = self.down1(x2)
        x3 = self.pool(x2)
        x3 = self.down2(x3)
        x4 = self.pool(x3)
        x4 = self.down3(x4)
        x5 = self.pool(x4)
        x5 = self.down4(x5)
        
        d4 = self.up4(x5)
        x4 = self.att4(d4, x4)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.up_conv4(d4)
        
        d3 = self.up3(d4)
        x3 = self.att3(d3, x3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.up_conv3(d3)
        
        d2 = self.up2(d3)
        x2 = self.att2(d2, x2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.up_conv2(d2)
        
        d1 = self.up1(d2)
        x1 = self.att1(d1, x1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.up_conv1(d1)
        
        return self.final(d1)

# if __name__ == "__main__":
#     model = AttentionUNet(in_channels=3, n_classes=19)
#     x = torch.randn(1, 3, 256, 256)  # Example input
#     y = model(x)
#     print(y.shape)  # Expected output shape: (1, 19, 256, 256)
