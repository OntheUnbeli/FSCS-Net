import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def apply_frequency_filter(feature_map, filter_type='low_pass', cutoff_freq=0.05):
    # 遍历特征图的每个通道
    filtered_feature_map = torch.zeros_like(feature_map)
    for i in range(feature_map.shape[1]):  # 遍历通道数
        channel = feature_map[:, i, :, :]  # 获取当前通道的特征图

        # 将张量转换为 NumPy 数组，并转换为灰度图像
        channel_numpy = channel.detach().cpu().numpy()[0]
        channel_numpy = np.abs(channel_numpy)  # 取绝对值
        channel_numpy = np.uint8(255 * (channel_numpy - np.min(channel_numpy)) / np.ptp(channel_numpy))

        # 傅里叶变换
        f_transform = np.fft.fft2(channel_numpy)
        f_shift = np.fft.fftshift(f_transform)

        rows, cols = channel_numpy.shape
        crow, ccol = rows // 2, cols // 2

        # 生成频域滤波器
        mask = np.zeros((rows, cols), np.uint8)
        if filter_type == 'high_pass':
            mask[crow - int(cutoff_freq * crow):crow + int(cutoff_freq * crow),
            ccol - int(cutoff_freq * ccol):ccol + int(cutoff_freq * ccol)] = 1
        elif filter_type == 'low_pass':
            mask[crow - int(cutoff_freq * crow):crow + int(cutoff_freq * crow),
            ccol - int(cutoff_freq * ccol):ccol + int(cutoff_freq * ccol)] = 0
            mask = 1 - mask

        # 应用滤波器
        f_shift = f_shift * mask

        # 逆傅里叶变换
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # 将 NumPy 数组转换回张量，并复制到结果特征图中的对应通道
        filtered_feature_map[:, i, :, :] = torch.from_numpy(img_back).unsqueeze(0)

    return filtered_feature_map


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

        F_Q = apply_frequency_filter(fuse, filter_type='high_pass')
        fuse_Query = self.query_transform(F_Q).view(B, C, -1)  # B C H*W
        fuse_Q_mask = fuse_Query.reshape(B*H, C*W)  # B*H CW
        fuse_Query_t = torch.transpose(fuse_Query, 1, 2).contiguous()  # B H*W C

        F_K = apply_frequency_filter(fuse, filter_type='low_pass')
        fuse_Key = self.key_transform(F_K).view(B, C, -1)  # B C H*W

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