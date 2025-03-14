import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=1, s=1, p=0, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu),
                )

    def forward(self, x):
        return self.conv(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6

class h_wish(nn.Module):
    def __init__(self, inplace=True):
        super(h_wish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

def channel_shuffle(x, groups):

    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # num_channels = groups * channels_per_group

    # grouping, 通道分组
    # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()
    # x.shape=(batchsize, channels_per_group, groups, height, width)
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class RDSM(nn.Module):
    def __init__(self, dim):
        super(RDSM, self).__init__()
        self.dim = dim
        self.soft = nn.Softmax(dim=1)
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim*2, int(dim*2 // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(dim*2 // 4)),
            nn.GELU(),
        )
        self.local_att = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(dim*2, int(dim*2 // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(dim*2 // 4)),
            nn.GELU(),
        )
        self.conv1 = DSConv3x3(dim//2, dim, 1, 1)
        self.conv = DSConv3x3(dim*2, dim, 1, 1)
        self.sig = nn.Sigmoid()




    def forward(self, r, d):
        cat = torch.cat([r, d], dim=1)
        Local = self.local_att(cat)
        Global = self.global_att(cat)
        att = Local + Global
        mask1 = self.sig(self.conv1(att))
        cat_shuffel = channel_shuffle(cat, 4)
        Local1 = self.local_att(cat_shuffel)
        Global1 = self.global_att(cat_shuffel)
        att1 = Local1 + Global1
        mask2 = self.sig(self.conv1(att1))
        cat1 = self.conv(cat)
        out = mask1 * cat1 + cat1 + mask2 * cat1

        return out

if __name__ == '__main__':
    a = torch.randn(5, 512, 15, 20)
    b = torch.randn(5, 512, 15, 20)
    out = RDSM(512)
    output = out(a, b)
    print(output.shape)