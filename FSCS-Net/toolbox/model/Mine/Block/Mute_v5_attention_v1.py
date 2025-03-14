import torch
from torch import nn
import torch.nn.functional as F

class Mute(nn.Module):
    def __init__(self, input_channels):
        super(Mute, self).__init__()
        self.query_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.key_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.value_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.scale = 1.0 / (input_channels ** 0.5)

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )


    def forward(self, fuse):
        B, C, H, W = fuse.size()

        fuse_Query = self.query_transform(fuse).view(B, C, -1)  # B C H*W
        fuse_Q_mask = fuse_Query.reshape(B*H, C*W)  # B*H CW

        fuse_Query_t = torch.transpose(fuse_Query, 1, 2).contiguous()  # B H*W C
        fuse_Key = self.key_transform(fuse).view(B, C, -1)  # B C H*W

        Fuse_Q_mask = torch.max(fuse_Q_mask, 0, keepdim=True).values  # 1 CW

        mask_KQ = torch.matmul(fuse_Query_t, fuse_Key) * self.scale  # B HW HW
        mask_KQ = F.softmax(mask_KQ, dim=0)  # B HW HW
        mask_KQ = torch.max(mask_KQ, 1).values  # B HW
        mask_KQ = mask_KQ.reshape(-1, W)   # BH W
        mask_KQ = torch.max(mask_KQ, -1, keepdim=True).values  # (BH, 1)

        mask = torch.matmul(mask_KQ, Fuse_Q_mask)
        mask = mask.reshape(B, C, H, W)
        fuse_Value = self.value_transform(fuse)
        fuse_mask = fuse_Value * mask
        out = fuse_Value + fuse_mask

        return out

if __name__ == '__main__':
    a = torch.randn(5, 512, 20, 15)
    b = torch.randn(5, 512, 20, 15)
    out = Mute(512)
    output = out(a)
    print(output.shape)