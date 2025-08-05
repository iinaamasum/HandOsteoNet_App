import torch
import torch.nn as nn
from torchvision import models

class BoneAgeSegModel(nn.Module):
    def __init__(self, input_size=480):
        super().__init__()
        regnet = models.regnet_y_400mf(weights='IMAGENET1K_V1')
        self.encoder_stem = regnet.stem
        self.s1 = regnet.trunk_output[0]
        self.s2 = regnet.trunk_output[1]
        self.s3 = regnet.trunk_output[2]
        self.s4 = regnet.trunk_output[3]
        self.decoder = nn.ModuleList([
            # [B, 440, 15, 15] -> [B, 208, 30, 30]
            nn.Sequential(
                nn.ConvTranspose2d(440, 208, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(208),
                nn.ReLU(inplace=True)
            ),
            # [B, 416, 30, 30] -> [B, 104, 60, 60]
            nn.Sequential(
                nn.ConvTranspose2d(416, 104, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(104),
                nn.ReLU(inplace=True)
            ),
            # [B, 208, 60, 60] -> [B, 48, 120, 120]
            nn.Sequential(
                nn.ConvTranspose2d(208, 48, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True)
            ),
            # [B, 96, 120, 120] -> [B, 32, 240, 240]
            nn.Sequential(
                nn.ConvTranspose2d(96, 32, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            # [B, 32, 240, 240] -> [B, 1, 480, 480]
            nn.Sequential(
                nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
                nn.Sigmoid(),
                nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=False)
            )
        ])
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.decoder:
            for m in layer:
                if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.encoder_stem(x)
        x2 = self.s1(x1)
        x3 = self.s2(x2)
        x4 = self.s3(x3)
        x5 = self.s4(x4)
        x = self.decoder[0](x5)
        x = torch.cat([x, x4], dim=1)
        x = self.decoder[1](x)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder[2](x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder[3](x)
        x = self.decoder[4](x)
        return x 