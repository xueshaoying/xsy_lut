import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out) 
        return x * scale
class ITAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(ITAM, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),              
            nn.Conv2d(gate_channels, gate_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels // 4, 3, 1, bias=False),  
            nn.Sigmoid()
        )
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        b, c, _, _ = x.size()
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_1 = self.ChannelGateH(x_perm1)
        x_1 = x_1.permute(0,2,1,3).contiguous()
        x_2 = x.permute(0,3,2,1).contiguous()
        x_2 = self.ChannelGateW(x_2)
        x_2 = x_2.permute(0,3,2,1).contiguous()
        x_3=self.SpatialGate(x)
        x_out11=x_1*x_2
        x_out22=x_2*x_3
        x_out33=x_1*x_3
        x_out = (1/3)*(x_out11+x_out22+x_out33)
        return x_out  