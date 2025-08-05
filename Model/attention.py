import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_cat)
        return x * self.sigmoid(x_out)

class CBAM(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.qkv.weight, mode='fan_out', nonlinearity='linear')
        nn.init.constant_(self.qkv.bias, 0)
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='linear')
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.view(B, C, N).transpose(1, 2)
        q, k, v = self.qkv(x_flat).chunk(3, dim=-1)
        q = q.view(B, N, self.heads, C // self.heads).transpose(1, 2)
        k = k.view(B, N, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, N, self.heads, C // self.heads).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.clamp(attn, min=-10, max=10)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.proj(out)
        out = out.transpose(1, 2).view(B, C, H, W)
        return x + out 