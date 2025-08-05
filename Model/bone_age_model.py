import torch
import torch.nn as nn
from .attention import CBAM, SelfAttention

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.depthwise.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.pointwise.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class HandOsteoNetModel(nn.Module):
    def __init__(self, img_size=480):
        super(HandOsteoNetModel, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),
            CBAM(64),
            nn.MaxPool2d(2, 2)
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=1),
            CBAM(128),
            nn.MaxPool2d(2, 2)
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(128, 256, stride=1),
            SelfAttention(256),
            nn.MaxPool2d(2, 2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.gender_fc = nn.Sequential(
            nn.Linear(1, 8, bias=True),
            nn.ReLU(inplace=True),
            nn.LayerNorm(8)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + 8, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1, bias=True)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, gender):
        if x.shape[1:] != (4, 480, 480):
            raise ValueError(f"Expected image shape [B, 4, 480, 480], got {x.shape}")
        if gender.ndim == 1:
            gender = gender.unsqueeze(1)
        if gender.shape[1:] != (1,):
            raise ValueError(f"Expected gender shape [B, 1], got {gender.shape}")
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        gender = gender.float()
        gender_feat = self.gender_fc(gender)
        x = torch.cat([x, gender_feat], dim=1)
        out = self.classifier(x)
        return out 