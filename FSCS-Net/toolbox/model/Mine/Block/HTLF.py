import torch
from torch import nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class HTLF(nn.Module):
    def __init__(self, H_channel, L_channel):
        super(HTLF, self).__init__()
        self.cur_channel = H_channel + L_channel
        self.conv1 = BasicConv2d(self.cur_channel, self.cur_channel, 1)
        self.conv3 = BasicConv2d(self.cur_channel, self.cur_channel, 3, padding=1)
        self.conv5 = BasicConv2d(self.cur_channel, self.cur_channel, 5, padding=2)
        # F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)

        self.convhtl = nn.Conv2d(H_channel, L_channel, kernel_size=1)
        self.convHtL = nn.Conv2d(self.cur_channel, H_channel, kernel_size=1)

    def forward(self, high, low):
        high_up = F.interpolate(high, scale_factor=2, mode='bilinear', align_corners=True)
        H_L_Mul = torch.cat((high_up, low), dim=1)
        H_L_1 = self.conv1(H_L_Mul)
        H_L_3 = self.conv3(H_L_Mul)
        H_L_5 = self.conv5(H_L_Mul)
        H_L_m = H_L_1 + H_L_3 + H_L_5

        H_L_m = self.convHtL(H_L_m)
        mask = H_L_m * high_up
        high_upc = self.convhtl(mask)

        return high_upc + low

if __name__ == '__main__':
    a = torch.randn(5, 3, 8, 8)
    b = torch.randn(5, 3, 16, 16)
    aat = HTLF(3, 3)
    output = aat(a, b)
    print(output.shape)